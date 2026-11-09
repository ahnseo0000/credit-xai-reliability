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
