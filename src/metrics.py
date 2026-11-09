"""신용평가 모형 검증 지표 (L1-9): 변별력, 보정, 부트스트랩 신뢰구간.

모든 함수는 y(0/1, 1=부도)와 p(부도확률 또는 점수, 클수록 위험)만 받는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, roc_auc_score

N_BINS = 10  # 십분위 lift·ECE 공통 구간 수
BOOTSTRAP_B = 1000


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """순위합 기반 AUC (동점은 평균순위). 부트스트랩 반복용."""
    r = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """부도·정상 누적분포의 최대 차이."""
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (len(ys) - ys.sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def quantile_bins(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """예측값이 높은 순으로 같은 인원씩 n_bins개 구간 (1=가장 위험). 동점은 순서대로 쪼갠다.

    정의: 위험도 내림차순 정렬 후 순위 i(0부터)를 floor(i * n_bins / N)+1 구간에 배정."""
    order = np.argsort(-p, kind="mergesort")
    bins = np.empty(len(p), dtype=int)
    bins[order] = (np.arange(len(p)) * n_bins // len(p)) + 1
    return bins


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """십분위(1=최고위험) 부도율, lift, 누적 포착률, 평균 예측확률."""
    d = pd.DataFrame({"y": y, "p": p, "bin": quantile_bins(p, n_bins)})
    g = d.groupby("bin").agg(n=("y", "size"), bads=("y", "sum"), mean_pred=("p", "mean"))
    g["bad_rate"] = g["bads"] / g["n"]
    g["lift"] = g["bad_rate"] / y.mean()
    g["cum_capture"] = g["bads"].cumsum() / g["bads"].sum()
    g["cum_bad_rate"] = g["bads"].cumsum() / g["n"].cumsum()
    return g.reset_index()


def monotonicity(tbl: pd.DataFrame) -> dict:
    """등급(십분위) 부도율이 위험 순서대로 단조감소하는지. 역전 개수와 최대 역전폭."""
    diff = np.diff(tbl["bad_rate"].to_numpy())
    inv = diff > 0
    return {
        "monotone": bool(not inv.any()),
        "inversions": int(inv.sum()),
        "max_inversion": float(diff.max()) if inv.any() else 0.0,
    }


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: 분위수(같은 인원) n_bins 구간에서
    sum_b (n_b / N) * |평균 예측확률_b - 실제 부도율_b|."""
    t = decile_table(y, p, n_bins)
    return float((t["n"] / t["n"].sum() * (t["mean_pred"] - t["bad_rate"]).abs()).sum())


def credit_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, p))
    mono = monotonicity(decile_table(y, p))
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_statistic(y, p),
        "top_decile_lift": float(decile_table(y, p).loc[0, "lift"]),
        "top30_capture": float(decile_table(y, p).loc[2, "cum_capture"]),
        "Brier": float(brier_score_loss(y, p)),
        "ECE_10q": ece(y, p),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y.mean()),
        "grade_monotone": mono["monotone"],
        "grade_inversions": mono["inversions"],
    }


PSI_EPS = 1e-4  # 비율이 0인 구간에 넣는 값


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = N_BINS) -> dict:
    """Hosmer-Lemeshow 적합도. 예측확률 기준 같은 인원 g구간 (십분위와 같은 구간 정의).

    통계량 = Σ (O_b − E_b)² / (E_b · (1 − E_b / n_b)), O 실제 부도 수, E 예측 부도 수 합.
    p값은 자유도 g (학습에 쓰지 않은 표본에 대한 표준 선택)와 g−2 (개발표본용) 두 가지를 모두 낸다.
    표본이 4만 건 이상이라 통계량이 작은 편차에도 커지므로 ECE·보정곡선과 함께 읽는다."""
    from scipy.stats import chi2

    d = pd.DataFrame({"y": y, "p": p, "bin": quantile_bins(p, g)})
    t = d.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    stat = float((((t["obs"] - t["exp"]) ** 2) / (t["exp"] * (1 - t["exp"] / t["n"]))).sum())
    return {
        "HL_stat": stat,
        "HL_p_df10": float(chi2.sf(stat, g)),
        "HL_p_df8": float(chi2.sf(stat, g - 2)),
    }


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = N_BINS, eps: float = PSI_EPS) -> float:
    """Population Stability Index. 구간은 expected(기준=학습셋)에서 정한다.

    - 수치형: expected의 분위수 n_bins구간 (중복 경계는 합침, 양 끝은 열어 둠). 결측은 별도 구간.
    - 범주형: expected에 있는 범주 + 없던 범주('기타') + 결측.
    - 비율이 0인 구간은 eps로 바꾼다 (재정규화 없음). PSI = Σ (a−e)·ln(a/e)."""
    expected = pd.Series(expected).reset_index(drop=True)
    actual = pd.Series(actual).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))

        def code(s):
            c = np.searchsorted(edges, s.to_numpy(dtype=float), side="left")
            return np.where(s.isna().to_numpy(), len(edges) + 1, c)

        k = len(edges) + 2
        ce, ca = code(expected), code(actual)
    else:
        cats = {v: i for i, v in enumerate(pd.unique(expected.dropna()))}
        k = len(cats) + 2

        def code(s):
            return np.array([len(cats) + 1 if pd.isna(v) else cats.get(v, len(cats)) for v in s])

        ce, ca = code(expected), code(actual)
    e = np.bincount(ce, minlength=k) / len(ce)
    a = np.bincount(ca, minlength=k) / len(ca)
    e = np.where(e == 0, eps, e)
    a = np.where(a == 0, eps, a)
    return float(((a - e) * np.log(a / e)).sum())


def calibration_method_table(y: np.ndarray, cands: dict[str, tuple[float, np.ndarray]]) -> pd.DataFrame:
    """보정 방법 비교. cands: 방법 → (검증셋 5-fold Brier, 테스트셋 보정 확률)."""
    rows = []
    for name, (cv_brier, p) in cands.items():
        rows.append(
            {
                "method": name,
                "valid_5fold_Brier": cv_brier,
                "test_Brier": float(brier_score_loss(y, p)),
                "test_ECE_10q": ece(y, p),
                "test_AUC": float(roc_auc_score(y, p)),
                "unique_values": int(len(np.unique(p))),
                "n_exact_zero": int((p == 0).sum()),
                "n_exact_one": int((p == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def metric_suite(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    baseline: str,
    score_splits: dict[str, dict[str, np.ndarray]] | None = None,
    feature_splits: dict[str, pd.DataFrame] | None = None,
    B: int = BOOTSTRAP_B,
) -> dict[str, pd.DataFrame]:
    """검증 지표를 한 번에 계산한다.

    y, preds: 테스트셋 라벨과 모형별 예측 (클수록 위험).
    score_splits: 모형 → {'train','valid','test': 점수} (점수 PSI용).
    feature_splits: {'train','valid','test': 피처 DataFrame} (변수 PSI용).
    반환: summary(KS·Gini·ECE·Brier 등), deciles, bootstrap, hl, psi_score, psi_variable."""
    out = {
        "summary": pd.DataFrame([{"model": k, **credit_metrics(y, p)} for k, p in preds.items()]),
        "deciles": pd.concat([decile_table(y, p).assign(model=k) for k, p in preds.items()], ignore_index=True),
        "bootstrap": bootstrap_auc(y, preds, baseline=baseline, B=B),
        "hl": pd.DataFrame([{"model": k, **hosmer_lemeshow(y, p)} for k, p in preds.items()]),
    }
    if score_splits:
        rows = []
        for m, s in score_splits.items():
            for other in ("valid", "test"):
                rows.append({"model": m, "comparison": f"train vs {other}", "PSI": psi(pd.Series(s["train"]), pd.Series(s[other]))})
        out["psi_score"] = pd.DataFrame(rows)
    if feature_splits:
        tr = feature_splits["train"]
        rows = [
            {"variable": c, "PSI_train_vs_valid": psi(tr[c], feature_splits["valid"][c]), "PSI_train_vs_test": psi(tr[c], feature_splits["test"][c])}
            for c in tr.columns
        ]
        out["psi_variable"] = pd.DataFrame(rows).sort_values("PSI_train_vs_test", ascending=False, ignore_index=True)
    return out


def paired_auc_diffs(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    pairs: list[tuple[str, str, str]],
    B: int = BOOTSTRAP_B,
    seed: int = 42,
) -> pd.DataFrame:
    """같은 부트스트랩 재표본으로 AUC 차이(a − b)의 95% percentile 구간을 계산한다.

    pairs: (설명, a 모형 이름, b 모형 이름). 모든 모형이 같은 재표본을 쓰므로 쌍 비교다."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = sorted({x for _, a, b in pairs for x in (a, b)})
    aucs = {k: np.empty(B) for k in names}
    for i in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for k in names:
            aucs[k][i] = fast_auc(yb, preds[k][idx])
    rows = []
    for label, a, b in pairs:
        d = aucs[a] - aucs[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(
            {
                "comparison": label,
                "a": a,
                "b": b,
                "AUC_a": float(roc_auc_score(y, preds[a])),
                "AUC_b": float(roc_auc_score(y, preds[b])),
                "diff_a_minus_b": float(roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b])),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_auc(y: np.ndarray, preds: dict[str, np.ndarray], baseline: str, B: int = BOOTSTRAP_B, seed: int = 42):
    """같은 재표본으로 모형별 AUC 95% 구간과 (모형 − baseline) AUC 차이 구간을 계산한다.

    percentile 방식, 재표본은 관측 단위 단순 복원추출 (층화 없음)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = {k: np.empty(B) for k in preds}
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for k, p in preds.items():
            aucs[k][b] = fast_auc(yb, p[idx])
    rows = []
    for k, p in preds.items():
        lo, hi = np.percentile(aucs[k], [2.5, 97.5])
        row = {"model": k, "AUC": float(roc_auc_score(y, p)), "AUC_ci_low": float(lo), "AUC_ci_high": float(hi)}
        if k != baseline:
            diff = aucs[k] - aucs[baseline]
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            row.update(
                {
                    "diff_vs_" + baseline: float(roc_auc_score(y, p) - roc_auc_score(y, preds[baseline])),
                    "diff_ci_low": float(dlo),
                    "diff_ci_high": float(dhi),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
