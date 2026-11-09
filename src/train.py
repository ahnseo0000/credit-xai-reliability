"""T1-2 ~ T1-8: 전처리, 분할, 로지스틱 baseline, XGBoost, 확률 보정, 성능 비교, 저장.

실행: python -m src.train
출력:
  results/<MODEL_VERSION>/modeling/   성능 비교표, 신용 지표, 로지스틱 계수·Wald, VIF, 보정 곡선, 집단 경계값
  results/<MODEL_VERSION>/models/     저장된 모델 (Streamlit 재사용)
  results/<MODEL_VERSION>/raw/        분할 ID, 테스트셋 예측값 (git 제외)
"""
from __future__ import annotations

import json
import platform
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
import xgboost as xgb
from scipy.linalg import qr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C
from .data import add_groups, clean, income_cuts, load_raw, load_var_dict, model_features, split_labels
from .metrics import N_BINS, calibration_method_table, metric_suite
from .models import Calibrator, best_f1_threshold, evaluate, feature_to_variable, make_preprocessor

OUT = C.MODELING_DIR
C.set_korean_font()

XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    tree_method="hist",
    eval_metric="auc",
    early_stopping_rounds=150,
    random_state=C.SEED,
    n_jobs=C.N_JOBS,
)


def log(msg):
    print(f"[train] {msg}", flush=True)


# ---------------------------------------------------------------- 로지스틱 진단
def independent_columns(X: pd.DataFrame, tol: float = 1e-8) -> tuple[list[str], list[str]]:
    """피벗 QR로 선형종속 컬럼을 찾는다 (Wald 검정 전 특이행렬 방지)."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    _, R, piv = qr(A, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int((d > tol * d[0]).sum())
    keep_idx = sorted(i - 1 for i in piv[:rank] if i != 0)
    keep = [X.columns[i] for i in keep_idx]
    drop = [c for c in X.columns if c not in keep]
    return keep, drop


MIN_DUMMY_POSITIVES = 50


def logit_wald(
    X: pd.DataFrame, y: pd.Series, f2v: dict, kor: dict, num_cols: list[str]
) -> tuple[pd.DataFrame, dict]:
    """statsmodels 로지스틱으로 Wald 검정.
    - 1이 50건 미만인 더미는 뺀다 (완전분리로 계수가 발산).
    - 남은 컬럼 중 선형종속인 것은 피벗 QR로 뺀다."""
    sparse = [c for c in X.columns if c not in num_cols and (X[c] > 0).sum() < MIN_DUMMY_POSITIVES]
    X = X.drop(columns=sparse)
    keep, collinear = independent_columns(X, tol=1e-6)
    res = sm.Logit(y.to_numpy(), sm.add_constant(X[keep])).fit(method="newton", maxiter=100, disp=0)
    if not res.mle_retvals["converged"]:
        raise RuntimeError("로지스틱 Wald 모형이 수렴하지 않음")
    dropped = {"sparse": sparse, "collinear": collinear}
    tbl = pd.DataFrame(
        {
            "feature": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "z": res.tvalues.values,
            "p_value": res.pvalues.values,
            "odds_ratio": np.exp(res.params.values),
        }
    )
    tbl.insert(1, "variable", tbl["feature"].map(lambda f: f2v.get(f, "const")))
    tbl.insert(2, "korean_name", tbl["variable"].map(lambda v: kor.get(v, "")))
    tbl["direction"] = np.where(tbl["coef"] > 0, "+ (부도↑)", "- (부도↓)")
    tbl["significant_0.05"] = tbl["p_value"] < 0.05
    return tbl, dropped


def vif_table(X: pd.DataFrame, f2v: dict, n: int = 50000) -> pd.DataFrame:
    Xs = sm.add_constant(X.sample(n=min(len(X), n), random_state=C.SEED)).to_numpy()
    with warnings.catch_warnings():  # 완전 선형종속 컬럼은 VIF가 무한대로 나온다 (보고서에 표시)
        warnings.simplefilter("ignore")
        vifs = [variance_inflation_factor(Xs, i + 1) for i in range(X.shape[1])]
    return (
        pd.DataFrame({"feature": X.columns, "variable": [f2v[c] for c in X.columns], "VIF": vifs})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- 보정 방법 선택
def choose_calibration(p_raw: np.ndarray, y: np.ndarray) -> tuple[str, dict]:
    """검증셋 안에서 5-fold로 isotonic vs Platt의 Brier를 비교."""
    scores = {"isotonic": [], "platt": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(p_raw, y):
        for m in scores:
            cal = Calibrator(m).fit(p_raw[tr], y[tr])
            scores[m].append(brier_score_loss(y[te], cal.predict(p_raw[te])))
    mean = {m: float(np.mean(s)) for m, s in scores.items()}
    return min(mean, key=mean.get), mean


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ T1-2, T1-3 데이터
    vd = load_var_dict()
    kor = dict(zip(vd["variable"], vd["korean_name"]))
    feats = model_features(vd)
    df = clean(load_raw())
    split = split_labels(df)
    pd.DataFrame({C.ID_COL: df[C.ID_COL], "split": split}).to_csv(C.RAW_RESULTS_DIR / "splits.csv", index=False)
    tr, va, te = (df[split == s] for s in ("train", "valid", "test"))
    y_tr, y_va, y_te = (d[C.TARGET_COL].to_numpy() for d in (tr, va, te))
    log(f"split train/valid/test = {len(tr):,}/{len(va):,}/{len(te):,}, 부도율 {y_tr.mean():.4f}/{y_va.mean():.4f}/{y_te.mean():.4f}")

    cuts = income_cuts(tr)
    (OUT / "group_cuts.json").write_text(
        json.dumps({"income_q1": cuts[0], "income_q2": cuts[1], "age_bins": [35, 55]}, indent=2), encoding="utf-8"
    )
    log(f"소득 경계값 q1={cuts[0]:,.1f}, q2={cuts[1]:,.1f}")

    # ------------------------------------------------ T1-4 로지스틱 baseline
    log("로지스틱 baseline 학습")
    lr_pre = make_preprocessor(tr, feats, "lr")
    X_tr_lr = lr_pre.fit_transform(tr[feats])
    lr_f2v = feature_to_variable(list(X_tr_lr.columns), feats)
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(X_tr_lr, y_tr)
    lr_pipe = Pipeline([("pre", lr_pre), ("lr", lr)])
    p_va_lr = lr_pipe.predict_proba(va[feats])[:, 1]
    p_te_lr = lr_pipe.predict_proba(te[feats])[:, 1]

    log("Wald 검정 (statsmodels)")
    lr_num_cols = list(lr_pre.transformers_[0][2])
    wald, dropped = logit_wald(X_tr_lr, pd.Series(y_tr), lr_f2v, kor, lr_num_cols)
    wald.to_csv(OUT / "logit_coefficients.csv", index=False, encoding="utf-8-sig")
    log(f"Wald 검정에서 뺀 컬럼: {dropped}")

    log("VIF 계산")
    vif = vif_table(X_tr_lr, lr_f2v)
    vif.to_csv(OUT / "vif.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ T1-5 XGBoost
    log("XGBoost 학습")
    xgb_pre = make_preprocessor(tr, feats, "xgb")
    X_tr = xgb_pre.fit_transform(tr[feats])
    X_va = xgb_pre.transform(va[feats])
    X_te = xgb_pre.transform(te[feats])
    xgb_f2v = feature_to_variable(list(X_tr.columns), feats)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    model = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
    log(f"best_iteration={model.best_iteration}, valid AUC={model.best_score:.4f}, scale_pos_weight={spw:.2f}")
    p_va_raw = model.predict_proba(X_va)[:, 1]
    p_te_raw = model.predict_proba(X_te)[:, 1]

    # ------------------------------------------------ T1-6 확률 보정
    _, cv_brier = choose_calibration(p_va_raw, y_va)
    # Platt 고정: isotonic은 계단함수라 0%·100% 같은 극단값과 동점이 생기고
    # (고객 간 순위가 무너짐), Brier 차이는 미미하다. 수치는 보고서에 함께 남긴다.
    method = "platt"
    cal = Calibrator(method).fit(p_va_raw, y_va)
    p_va_xgb = cal.predict(p_va_raw)
    p_te_xgb = cal.predict(p_te_raw)
    log(f"보정 방법: {method} (검증셋 5-fold Brier {cv_brier})")

    cal_rows = []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="완벽 보정")
    for name, p in [("XGBoost 보정 전", p_te_raw), (f"XGBoost 보정 후 ({method})", p_te_xgb), ("로지스틱", p_te_lr)]:
        frac, mean_pred = calibration_curve(y_te, p, n_bins=N_BINS, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=3, label=name)
        cal_rows.append({"model": name, "test_Brier": brier_score_loss(y_te, p), "mean_pred": p.mean(), "observed_rate": y_te.mean()})
    ax.set_xlabel("예측 부도확률")
    ax.set_ylabel("실제 부도율")
    ax.set_title(f"보정 곡선 (테스트셋, 분위수 {N_BINS}구간)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_curve.png", dpi=150)
    plt.close(fig)
    cal_tbl = pd.DataFrame(cal_rows)
    cal_tbl.to_csv(OUT / "calibration_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ T1-7 성능 비교
    thr_lr = best_f1_threshold(y_va, p_va_lr)
    thr_xgb = best_f1_threshold(y_va, p_va_xgb)
    comp = pd.DataFrame(
        [
            {"model": "로지스틱 회귀", **evaluate(y_te, p_te_lr, thr_lr)},
            {"model": "XGBoost (보정 후)", **evaluate(y_te, p_te_xgb, thr_xgb)},
        ]
    )
    comp.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    log("\n" + comp.round(4).to_string(index=False))

    # ------------------------------------------------ L1-9 신용 검증 지표 (테스트셋)
    log("신용 지표·부트스트랩 (B=1000)")
    xgb_name = f"XGBoost 보정 후 ({method})"
    preds = {"로지스틱": p_te_lr, "XGBoost 보정 전": p_te_raw, xgb_name: p_te_xgb}
    # PSI용 점수: 모형의 최종 부도확률 (보정은 단조변환이라 보정 전 PSI는 구간이 같아 동일하다)
    p_tr_lr = lr_pipe.predict_proba(tr[feats])[:, 1]
    p_tr_xgb = cal.predict(model.predict_proba(X_tr)[:, 1])
    score_splits = {
        "로지스틱": {"train": p_tr_lr, "valid": p_va_lr, "test": p_te_lr},
        xgb_name: {"train": p_tr_xgb, "valid": p_va_xgb, "test": p_te_xgb},
    }
    feature_splits = {"train": tr[feats], "valid": va[feats], "test": te[feats]}
    suite = metric_suite(y_te, preds, baseline="로지스틱", score_splits=score_splits, feature_splits=feature_splits)
    cm, dec, boot = suite["summary"], suite["deciles"], suite["bootstrap"]
    cm.to_csv(OUT / "credit_metrics.csv", index=False, encoding="utf-8-sig")
    dec.to_csv(OUT / "decile_table.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "bootstrap_auc.csv", index=False, encoding="utf-8-sig")
    suite["hl"].to_csv(OUT / "hosmer_lemeshow.csv", index=False, encoding="utf-8-sig")
    suite["psi_score"].to_csv(OUT / "psi_score.csv", index=False, encoding="utf-8-sig")
    suite["psi_variable"].to_csv(OUT / "psi_variable.csv", index=False, encoding="utf-8-sig")
    plot_deciles(dec, y_te.mean())

    # isotonic vs Platt 비교 (채택은 Platt 고정, 근거를 표로 남긴다)
    cal_iso = Calibrator("isotonic").fit(p_va_raw, y_va)
    cal_cmp = calibration_method_table(
        y_te, {"isotonic": (cv_brier["isotonic"], cal_iso.predict(p_te_raw)), "platt": (cv_brier["platt"], p_te_xgb)}
    )
    cal_cmp.to_csv(OUT / "calibration_method_comparison.csv", index=False, encoding="utf-8-sig")
    log("\n" + cm.round(4).to_string(index=False))
    log("\n" + boot.round(4).to_string(index=False))
    log("\n" + suite["hl"].round(4).to_string(index=False))
    log("\n" + suite["psi_score"].round(5).to_string(index=False))
    log("\n" + cal_cmp.round(5).to_string(index=False))
    log("\n변수 PSI 상위 5\n" + suite["psi_variable"].head(5).round(5).to_string(index=False))

    # ------------------------------------------------ T1-8 저장
    bundle = {
        "preprocessor": xgb_pre,
        "model": model,
        "calibrator": cal,
        "features": feats,
        "feature_names": list(X_tr.columns),
        "feature_to_variable": xgb_f2v,
        "threshold_f1": thr_xgb,
        "income_cuts": cuts,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    joblib.dump(bundle, C.MODELS_DIR / "xgb_bundle.joblib", compress=3)
    model.save_model(C.MODELS_DIR / "xgb_model.json")
    joblib.dump({"pipeline": lr_pipe, "threshold_f1": thr_lr}, C.MODELS_DIR / "lr_pipeline.joblib", compress=3)

    groups = add_groups(te, cuts)
    pd.DataFrame(
        {
            C.ID_COL: te[C.ID_COL].to_numpy(),
            "target": y_te,
            "pd": p_te_xgb,
            "pd_raw": p_te_raw,
            "pd_lr": p_te_lr,
            **{g: groups[g].to_numpy() for g in groups},
        }
    ).to_csv(C.RAW_RESULTS_DIR / "test_predictions.csv", index=False)

    write_report(comp, cal_tbl, method, cv_brier, model, spw, wald, dropped, vif, cuts, len(tr), len(va), len(te))
    write_credit_report(suite, cal_cmp)
    log("완료")


def plot_deciles(dec: pd.DataFrame, base_rate: float):
    """십분위 부도율(막대)과 누적 포착률(선). 1=가장 위험한 십분위."""
    models = list(dec["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        d = dec[dec["model"] == m]
        axes[0].bar(d["bin"] + (i - (len(models) - 1) / 2) * w, d["bad_rate"], width=w, label=m)
        axes[1].plot(d["bin"], d["cum_capture"], marker="o", ms=3, label=m)
    axes[0].axhline(base_rate, color="k", ls="--", lw=1, label="전체 부도율")
    axes[0].set_xlabel("십분위 (1=최고위험)")
    axes[0].set_ylabel("실제 부도율")
    axes[0].set_title("십분위별 부도율")
    axes[0].set_xticks(range(1, N_BINS + 1))
    axes[0].legend(fontsize=7)
    axes[1].plot([0, N_BINS], [0, 1], "k--", lw=1, label="무작위")
    axes[1].set_xlabel("십분위 (누적)")
    axes[1].set_ylabel("누적 부도 포착률")
    axes[1].set_title("누적 포착률 (CAP)")
    axes[1].set_xticks(range(1, N_BINS + 1))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "decile_lift.png", dpi=150)
    plt.close(fig)


def write_credit_report(suite: dict, cal_cmp: pd.DataFrame):
    cm, dec, boot = suite["summary"], suite["deciles"], suite["bootstrap"]
    L = [
        "# 신용 검증 지표 (L1-9)",
        "",
        "`python -m src.train`으로 생성. 모두 테스트셋(46,127건) 기준.",
        "",
        "## 정의",
        "",
        "- **KS**: 부도·정상 누적분포의 최대 차이. **Gini** = 2·AUC − 1.",
        f"- **십분위**: 예측 위험도 내림차순으로 같은 인원씩 {N_BINS}개 구간 (1=최고위험). 동점은 정렬 순서대로 쪼갠다.",
        "- **lift** = 구간 부도율 / 전체 부도율. **누적 포착률** = 상위 k개 구간에 든 부도 수 / 전체 부도 수.",
        "- **등급 단조성**: 십분위 부도율이 위험 순서대로 줄어드는지. 역전 = 다음 구간 부도율이 더 높은 경우.",
        f"- **ECE** = Σ (n_b/N) · |평균 예측확률_b − 실제 부도율_b|, b는 위의 같은 인원 {N_BINS}구간 (분위수 구간).",
        "- 부트스트랩: 테스트셋 관측 단순 복원추출 B=1,000, 시드 42, percentile 95% 구간. 모형 간 차이는 같은 재표본으로 계산(대응).",
        "- XGBoost 보정 전은 `scale_pos_weight`로 학습해 확률 수준이 크게 높다. 순위 지표(AUC·KS·lift)는 보정 후와 같고 Brier·ECE만 다르다.",
        "",
        "## 요약",
        "",
        "| 모형 | AUC | Gini | KS | 1분위 lift | 상위 30% 포착률 | Brier | ECE | 등급 단조 | 역전 수 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in cm.itertuples():
        L.append(
            f"| {r.model} | {r.AUC:.4f} | {r.Gini:.4f} | {r.KS:.4f} | {r.top_decile_lift:.2f} | {r.top30_capture:.1%} "
            f"| {r.Brier:.4f} | {r.ECE_10q:.4f} | {'예' if r.grade_monotone else '아니오'} | {r.grade_inversions} |"
        )
    L += ["", "## 부트스트랩 AUC 95% 신뢰구간", "", "| 모형 | AUC | 95% CI | 로지스틱 대비 차이 | 차이 95% CI |", "| --- | --- | --- | --- | --- |"]
    for r in boot.itertuples():
        if pd.isna(getattr(r, "diff_ci_low", np.nan)):
            diff = "-"
            dci = "-"
        else:
            diff = f"{r[5]:+.4f}"
            dci = f"[{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]"
        L.append(f"| {r.model} | {r.AUC:.4f} | [{r.AUC_ci_low:.4f}, {r.AUC_ci_high:.4f}] | {diff} | {dci} |")
    L += [
        "",
        "## Hosmer-Lemeshow (테스트셋, 십분위 10구간)",
        "",
        "통계량 = Σ (O−E)² / (E·(1−E/n)). p값은 자유도 10(학습에 쓰지 않은 표본)과 8(개발표본용)을 모두 적는다.",
        "표본이 4.6만 건이라 작은 편차도 기각되기 쉬우므로 ECE·보정곡선과 함께 읽는다.",
        "",
        "| 모형 | HL 통계량 | p (df=10) | p (df=8) |",
        "| --- | --- | --- | --- |",
    ]
    for r in suite["hl"].itertuples():
        L.append(f"| {r.model} | {r.HL_stat:.2f} | {r.HL_p_df10:.4f} | {r.HL_p_df8:.4f} |")
    L += [
        "",
        "## 확률 보정 방법 비교: isotonic vs Platt",
        "",
        "Brier는 낮을수록 좋다. 검증셋 5-fold는 보정 방법 선택에, 테스트셋은 최종 확인에 쓴다.",
        "",
        "| 방법 | 검증 5-fold Brier | 테스트 Brier | 테스트 ECE | 테스트 AUC | 고유 확률값 수 | 정확히 0 | 정확히 1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in cal_cmp.itertuples():
        L.append(
            f"| {r.method} | {r.valid_5fold_Brier:.5f} | {r.test_Brier:.5f} | {r.test_ECE_10q:.5f} | {r.test_AUC:.4f} "
            f"| {r.unique_values:,} | {r.n_exact_zero:,} | {r.n_exact_one:,} |"
        )
    L += [
        "",
        "## PSI (Population Stability Index)",
        "",
        "구간은 학습셋 분위수 10구간(중복 경계는 합침, 양 끝 열림, 결측은 별도 구간)이고, 범주형은 학습셋 범주 + 기타 + 결측이다. "
        "비율이 0인 구간은 ε=1e-4로 바꾼다. 관례적 해석: 0.1 미만 안정, 0.1~0.25 주의, 0.25 초과 변화.",
        "",
        "> **한계: 시간 변수 부재.** Home Credit 데이터에는 신청일이 없고 분할은 TARGET 층화 *무작위* 분할이다. "
        "그래서 이 PSI는 분할 간 표본 변동만 재며 0 근처가 되는 것이 당연하다. 시점 간 모집단 변화(vintage·OOT)를 검증한 것이 아니다. "
        "이후 집단 간 PSI(성별·연령·소득)가 의미 있는 변동 확인이 된다.",
        "",
        "점수 PSI (최종 부도확률 기준):",
        "",
        "| 모형 | 비교 | PSI |",
        "| --- | --- | --- |",
    ]
    for r in suite["psi_score"].itertuples():
        L.append(f"| {r.model} | {r.comparison} | {r.PSI:.5f} |")
    pv = suite["psi_variable"]
    L += [
        "",
        f"변수 PSI (모형 입력 {len(pv)}개): 최대 {pv[['PSI_train_vs_valid', 'PSI_train_vs_test']].to_numpy().max():.5f}, "
        f"0.1 이상 {(pv[['PSI_train_vs_valid', 'PSI_train_vs_test']].max(axis=1) >= 0.1).sum()}개. 상위 10개 (전체: `psi_variable.csv`):",
        "",
        "| 변수 | train vs valid | train vs test |",
        "| --- | --- | --- |",
    ]
    for r in pv.head(10).itertuples():
        L.append(f"| `{r.variable}` | {r.PSI_train_vs_valid:.5f} | {r.PSI_train_vs_test:.5f} |")
    for m in dec["model"].unique():
        d = dec[dec["model"] == m]
        L += ["", f"## 십분위 표: {m}", "", "| 구간 | 인원 | 부도 | 부도율 | lift | 누적 포착률 | 평균 예측확률 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in d.itertuples():
            L.append(f"| {r.bin} | {r.n:,} | {r.bads:,} | {r.bad_rate:.2%} | {r.lift:.2f} | {r.cum_capture:.1%} | {r.mean_pred:.2%} |")
    L += ["", "그림: `decile_lift.png`, `calibration_curve.png`", ""]
    (OUT / "credit_metrics.md").write_text("\n".join(L), encoding="utf-8")


def write_report(comp, cal_tbl, method, cv_brier, model, spw, wald, dropped, vif, cuts, n_tr, n_va, n_te):
    L = ["# 모델링 결과 (T1-3 ~ T1-7)", "", "`python -m src.train`으로 생성.", ""]
    L += [
        "## 분할",
        "",
        f"- TARGET 층화 7:1.5:1.5, `random_state=42`: train {n_tr:,} / valid {n_va:,} / test {n_te:,}",
        f"- 소득 3분위 경계값(학습셋): q1 = {cuts[0]:,.1f}, q2 = {cuts[1]:,.1f}",
        "",
        "## 성능 비교 (테스트셋)",
        "",
        "컷오프는 검증셋에서 F1이 최대가 되는 값으로 정해 테스트셋에 적용했다. 부도율 8%라 0.5 컷오프는 쓰지 않는다.",
        "",
        "| 모델 | AUC | 정확도 | F1 | Brier | 컷오프 | TN | FP | FN | TP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in comp.itertuples():
        L.append(f"| {r.model} | {r.AUC:.4f} | {r.Accuracy:.4f} | {r.F1:.4f} | {r.Brier:.4f} | {r.threshold:.3f} | {r.TN:,} | {r.FP:,} | {r.FN:,} | {r.TP:,} |")
    L += [
        "",
        "## XGBoost",
        "",
        f"- `scale_pos_weight` = {spw:.2f} (학습셋 정상/부도 비), early stopping 150라운드 (검증셋 AUC)",
        f"- best iteration = {model.best_iteration}, 검증셋 AUC = {model.best_score:.4f}",
        f"- 하이퍼파라미터는 고정값 (5-fold 튜닝은 범위에서 제외): `{ {k: v for k, v in XGB_PARAMS.items() if k not in ('n_jobs', 'eval_metric')} }`",
        "- 결측은 XGBoost가 자체 처리, 범주형은 원-핫 (100건 미만 범주는 infrequent로 묶음)",
        "",
        "## 확률 보정 (T1-6)",
        "",
        f"- 검증셋 안 5-fold Brier: isotonic {cv_brier['isotonic']:.5f}, Platt {cv_brier['platt']:.5f}",
        f"- **{method}** 채택, 검증셋 전체로 다시 학습. isotonic은 계단함수라 부도확률이 정확히 0%·100%인 고객과 동점이 생긴다 (첫 실행에서 테스트셋 고유값 103개, 0%가 72명). 사용자·LLM에 보여줄 값으로 부적절하고 Brier 차이도 미미해 Platt을 쓴다.",
        "- 검증셋을 early stopping과 보정에 함께 썼다. 테스트셋은 어디에도 쓰지 않았다.",
        "- TreeSHAP은 보정 전 XGBoost의 log-odds 기준이다. 보정은 단조변환이라 고객 간 순위는 유지된다.",
        "",
        "| 모델 | 테스트 Brier | 평균 예측 | 실제 부도율 |",
        "| --- | --- | --- | --- |",
    ]
    for r in cal_tbl.itertuples():
        L.append(f"| {r.model} | {r.test_Brier:.5f} | {r.mean_pred:.4f} | {r.observed_rate:.4f} |")
    L += ["", "그림: `calibration_curve.png`", ""]

    w = wald[wald["variable"] != "const"].copy()
    w["abs_z"] = w["z"].abs()
    top = w.sort_values("abs_z", ascending=False).head(15)
    L += [
        "## 로지스틱 baseline (T1-4)",
        "",
        "- 전처리: 0.5%/99.5% 윈저화(고유값 50개 초과 연속형만) → 중앙값 대치 → 표준화, 범주형은 원-핫(첫 범주 제거). 계수는 표준화 단위.",
        f"- 유의(p<0.05) 계수 {int(w['significant_0.05'].sum())}개 / 전체 {len(w)}개. 전체 표: `logit_coefficients.csv`",
        f"- Wald 검정에서 뺀 컬럼 (예측용 모형에는 포함):",
        f"  - 1이 {MIN_DUMMY_POSITIVES}건 미만인 희소 더미 (완전분리로 계수 발산): {', '.join(f'`{d}`' for d in dropped['sparse']) or '없음'}",
        f"  - 완전 선형종속 (피벗 QR): {', '.join(f'`{d}`' for d in dropped['collinear']) or '없음'}",
        "",
        "|z| 상위 15개:",
        "",
        "| 컬럼 | 한국어 | 계수 | 오즈비 | z | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in top.itertuples():
        L.append(f"| `{r.feature}` | {r.korean_name} | {r.coef:+.3f} | {r.odds_ratio:.3f} | {r.z:+.1f} | {r.p_value:.1e} |")
    hi = vif[vif["VIF"] > 10]
    L += [
        "",
        "## VIF 진단",
        "",
        f"VIF > 10인 컬럼 {len(hi)}개 (5만 건 표본). 전체 표: `vif.csv`",
        "",
        "주요 원인 (구조적 중복이라 예상된 결과):",
        "",
        "- `ORGANIZATION_TYPE_XNA`, `FLAG_EMP_PHONE`, `NAME_INCOME_TYPE_Pensioner`, `DAYS_EMPLOYED`: 연금 수령·무직자를 서로 다른 방식으로 표시한 같은 정보",
        "- `CNT_FAM_MEMBERS` = `CNT_CHILDREN` + (기혼이면 2, 아니면 1): 혼인 상태 더미와 합쳐 완전 선형종속. Wald 검정에서는 피벗 QR로 종속 컬럼 하나를 자동으로 뺐다.",
        "- `OBS_30` ↔ `OBS_60`, `REGION_RATING_CLIENT` ↔ `_W_CITY`: 거의 같은 변수 쌍",
        "- `AMT_CREDIT` ↔ `AMT_GOODS_PRICE` ↔ 파생 비율 변수: 대출 금액과 상품 가격은 거의 같다. 그래서 로지스틱에서 두 계수 부호가 반대로 나온다 (개별 계수 해석 불가).",
        "- 범주형 더미: 기준 범주가 작을 때 생기는 더미 간 VIF",
        "",
        "로지스틱 계수는 이 변수들에 대해 개별 해석하지 않는다. XGBoost는 다중공선성에 예측 성능은 영향을 덜 받지만, SHAP 기여도가 상관된 변수끼리 나뉠 수 있다 (`docs/interface.md`의 `var_group` 참고).",
        "",
        "| 컬럼 | VIF |",
        "| --- | --- |",
    ]
    for r in hi.itertuples():
        shown = f"{r.VIF:,.1f}" if np.isfinite(r.VIF) and r.VIF < 1e6 else "∞ (완전 선형종속)"
        L.append(f"| `{r.feature}` | {shown} |")
    L.append("")
    (OUT / "modeling_summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
