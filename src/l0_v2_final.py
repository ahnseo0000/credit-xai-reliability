"""v2 확정 후: Platt 필요성 확인 → 로지스틱 챔피언 → test 최종 1회 → interface.md q70/q90 → 동결·태그.

test는 이 파일의 `evaluate_once`에서 단 한 번만 모형에 넣는다. 그 전에 같은 보고 코드를 valid 데이터로 먼저 돌려(dry run) 오류가 test 사용 뒤에 나지 않게 한다.
test 결과로 변수·모형·하이퍼파라미터·보정 여부를 다시 정하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from . import config as C
from .l0_run import age_proxy, commit, git, log
from .l0_tier2_run import Design
from .l0_v2 import L0, W, WRAW, calib_rows
from .metrics import bootstrap_auc, credit_metrics, decile_table
from .models import Calibrator, make_preprocessor
from .train import XGB_PARAMS

FROZEN = C.RESULTS_DIR / "v2_frozen"
PLATT_NEED = 0.001            # Platt 교차적합 ECE가 보정 전보다 이만큼 이상 낮을 때만 사용 (사전 고정)
C_GRID = (1e6, 1.0, 0.1, 0.01)


def add_missing_flags(df_lr: pd.DataFrame, feats: list[str], train_mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """train에서 결측이 있는 수치 변수마다 결측 표시 열. 결측 패턴이 같은 열은 하나로 합친다 (같은 테이블 변수들이 대부분 같은 패턴)."""
    seen: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    out = {}
    for f in feats:
        s = df_lr[f]
        if not pd.api.types.is_numeric_dtype(s) or not s.isna().any():
            continue
        if not s[train_mask].isna().any():
            continue
        flag = s.isna().to_numpy(np.uint8)
        key = hashlib.md5(flag.tobytes()).hexdigest()
        if key in seen:
            groups[seen[key]].append(f)
            continue
        name = f"MISS__{f}"
        seen[key] = name
        groups[name] = [f]
        out[name] = flag.astype(np.float32)
    return pd.concat([df_lr, pd.DataFrame(out, index=df_lr.index)], axis=1), groups


def report(y, preds: dict[str, np.ndarray], groups_df: pd.DataFrame, ids: np.ndarray, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """지표표(모형별), AUC 신뢰구간·차이, 십분위표. valid dry run과 test 최종이 같은 코드를 쓴다."""
    rows = []
    for k, p in preds.items():
        rows.append({"model": k, "split": label, **credit_metrics(y, p), **age_proxy(ids, p, groups_df)})
    met = pd.DataFrame(rows)
    ci = bootstrap_auc(y, preds, baseline=list(preds)[-1])
    dec = {k: decile_table(y, p) for k, p in preds.items()}
    return met, ci, dec


def fill_interface(q70: float, q90: float, pd_te: np.ndarray, gv: pd.DataFrame, model_desc: str):
    """docs/interface.md '추출 고정값' 표를 채운다 (q70/q90은 테스트셋 전체 최종 pd)."""
    g = gv.copy()
    g["pd"] = pd_te
    g["bin"] = np.where(g["pd"] >= q90, "고", np.where(g["pd"] >= q70, "경계", "저"))
    g = g[g["group_gender"].isin(["M", "F"])]
    tab = g.groupby(["bin", "group_gender", "group_age"]).size().rename("n").reset_index()
    tab["cell_id"] = tab["bin"] + "_" + tab["group_gender"] + "_" + tab["group_age"]
    tab = tab.sort_values("cell_id")
    p = C.DOCS_DIR / "interface.md"
    s = p.read_text(encoding="utf-8")
    start = s.index("### 추출 고정값")
    end = s.index("## `variable_dict.csv` 컬럼")
    cells = ["", "18칸 크기표 (테스트셋, 성별 XNA 제외):", "", "| cell_id | 칸 크기 |", "| --- | --- |"] + [f"| `{r.cell_id}` | {r.n:,} |" for r in tab.itertuples()]
    new = ("### 추출 고정값 (최종 모형 동결 후 채운다)\n\n| 항목 | 값 |\n| --- | --- |\n"
           f"| 추출에 쓴 모형 버전 | {model_desc} |\n"
           f"| `q70` (테스트셋 `pd` 70번째 백분위수) | {q70:.6f} |\n"
           f"| `q90` (90번째 백분위수) | {q90:.6f} |\n"
           f"| 18칸 크기표 | 아래 표 (합 {int(tab.n.sum()):,}) |\n" + "\n".join(cells) + "\n\n")
    p.write_text(s[:start] + new + s[end:], encoding="utf-8")
    tab.to_csv(FROZEN / "cell_sizes_18.csv", index=False, encoding="utf-8-sig")
    return tab


def phase_final(state, commit_enabled: bool = True):
    df, app_ids, split, base_feats_all, names, meta, E, _ = state
    cand_p = W / "features_v2_candidate.json"
    if not cand_p.exists():
        raise SystemExit("v2 후보가 없다: sweep 단계를 먼저 실행")
    cand = json.loads(cand_p.read_text(encoding="utf-8"))
    base_sel, l0_sel = cand["base_feats"], cand["l0"]
    cols = np.array([names.index(n) for n in l0_sel])
    feats = base_sel + l0_sel
    log(f"v2 후보 K={cand['K']}: 신청서 {len(base_sel)} + L0 {len(l0_sel)} = {len(feats)}개 입력")
    design = Design(df, base_sel, split)
    yv, yt = design.y["valid"], design.y["test"]
    FROZEN.mkdir(parents=True, exist_ok=True)
    (FROZEN / "models").mkdir(exist_ok=True)
    (FROZEN / "raw").mkdir(exist_ok=True)

    # ---- 모형 불러오기 + 무결성 점검 (valid)
    model = xgb.XGBClassifier()
    model.load_model(cand["model_path"])
    Xva = design.X("valid", E, cols)
    p_va = model.predict_proba(Xva)[:, 1]
    saved = np.load(WRAW / (f"valid_p_K{cand['K']}.npy" if cand["K"] != 190 else "valid_p_190.npy"))
    assert np.allclose(p_va, saved, atol=1e-6), "저장된 모형의 valid 예측이 학습 때와 다르다"
    log(f"저장된 모형 무결성 확인: valid 예측 일치, valid AUC {roc_auc_score(yv, p_va):.4f}")

    # ---- 1) Platt 보정 필요 여부 (valid)
    cr = calib_rows(yv, p_va)
    need = (cr["raw_ECE"] - cr["platt_crossfit_ECE"]) >= PLATT_NEED
    log(f"Platt 필요 여부(valid): 보정 전 ECE {cr['raw_ECE']:.4f}, Platt in-sample {cr['platt_insample_ECE']:.4f}, cross-fit {cr['platt_crossfit_ECE']:.4f}; "
        f"Brier {cr['raw_Brier']:.4f} → {cr['platt_crossfit_Brier']:.4f}. 규칙: cross-fit ECE가 {PLATT_NEED} 이상 낮을 때만 사용 → **{'사용' if need else '사용 안 함(원 확률 그대로)'}**", decision=True)
    calib = Calibrator("platt").fit(p_va, yv) if need else None
    pd_of = (lambda p: calib.predict(p)) if need else (lambda p: p)

    # ---- 2) 로지스틱 챔피언 (train 전처리, 결측 표시 포함, C는 valid로 선택)
    df_l0 = pd.DataFrame(E[:, cols], columns=l0_sel, index=df.index)
    df_lr = pd.concat([df[base_sel], df_l0], axis=1)
    tr_mask = design.m["train"]
    df_lr, flag_groups = add_missing_flags(df_lr, feats, tr_mask)
    feats_lr = list(df_lr.columns)
    pre = make_preprocessor(df_lr[tr_mask], feats_lr, "lr")
    Xtr = pre.fit_transform(df_lr.loc[tr_mask, feats_lr])
    Xva_lr = pre.transform(df_lr.loc[design.m["valid"], feats_lr])
    best = None
    for Cc in C_GRID:
        lr = LogisticRegression(C=Cc, max_iter=5000)
        lr.fit(Xtr, design.y["train"])
        a = roc_auc_score(yv, lr.predict_proba(Xva_lr)[:, 1])
        log(f"로지스틱 C={Cc:g}: valid AUC {a:.4f}")
        if best is None or a > best[0]:
            best = (a, Cc, lr)
    lr_auc, lr_C, lr = best
    lr_pipe = Pipeline([("pre", pre), ("lr", lr)])
    p_va_lr = lr_pipe.predict_proba(df_lr.loc[design.m["valid"], feats_lr])[:, 1]
    log(f"로지스틱 챔피언: C={lr_C:g}, valid AUC {lr_auc:.4f}, 입력 {len(feats)}개 + 결측 표시 {len(flag_groups)}개(패턴 중복 합침)", decision=True)

    # ---- 3) dry run: test에 쓸 보고 코드를 valid로 먼저 돌린다
    groups = pd.read_csv(L0 / "raw" / "group_vars.csv").set_index(C.ID_COL)
    ids_va = df.loc[design.m["valid"], C.ID_COL].to_numpy()
    report(yv, {"xgb_v2": pd_of(p_va), "logit": p_va_lr}, groups, ids_va, "valid(dry-run)")
    gv_test = groups.loc[df.loc[design.m["test"], C.ID_COL].to_numpy()]
    log("보고 코드 dry run(valid) 통과. 이제 test를 처음으로 최종 1회 평가한다. 이후 test로 아무 결정도 하지 않는다.", decision=True)

    # ---- 4) test 최종 1회
    ids_te = df.loc[design.m["test"], C.ID_COL].to_numpy()
    p_te_raw = model.predict_proba(design.X("test", E, cols))[:, 1]
    p_te = pd_of(p_te_raw)
    p_te_lr = lr_pipe.predict_proba(df_lr.loc[design.m["test"], feats_lr])[:, 1]
    pd.DataFrame({C.ID_COL: ids_te, "target": yt, "pd_xgb_raw": p_te_raw, "pd_xgb_v2": p_te, "pd_logit": p_te_lr}).to_csv(FROZEN / "raw" / "test_predictions.csv", index=False)
    out = FROZEN / "test_final"
    out.mkdir(exist_ok=True)
    met, ci, dec = report(yt, {"xgb_v2": p_te, "logit": p_te_lr}, groups, ids_te, "test(최종 1회)")
    met.to_csv(out / "test_metrics.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(out / "test_auc_ci.csv", index=False, encoding="utf-8-sig")
    for k, t in dec.items():
        t.to_csv(out / f"decile_{k}.csv", index=False, encoding="utf-8-sig")
    md = ["# v2 최종 test 평가 (1회)", "", "동결된 모형으로 test를 처음이자 마지막으로 평가한 결과. ablation.csv와 별도 파일이며, 이 결과로 변수·모형·보정·하이퍼파라미터를 다시 정하지 않았다.", "",
          "| 모형 | AUC | Gini | KS | 1분위 lift | 상위 30% 포착 | Brier | ECE(10분위) | PD–연령 순위상관 | 연령 3구간 AUC |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in met.itertuples():
        md.append(f"| {r.model} | {r.AUC:.4f} | {r.Gini:.4f} | {r.KS:.4f} | {r.top_decile_lift:.2f} | {r.top30_capture:.1%} | {r.Brier:.4f} | {r.ECE_10q:.4f} | {r.pd_age_spearman:+.3f} | {r.pd_age_group_auc:.3f} |")
    md += ["", "AUC 95% 구간(부트스트랩 B=1000, 시드 42)과 차이 (xgb_v2 − logit):", "", "| 모형 | AUC | 95% 구간 | 차이 (vs logit) | 차이 95% 구간 |", "| --- | --- | --- | --- | --- |"]
    for r in ci.itertuples():
        d = f"{r.diff_vs_logit:+.4f}" if hasattr(r, "diff_vs_logit") and pd.notna(getattr(r, "diff_vs_logit", np.nan)) else "-"
        dc = f"[{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]" if d != "-" else "-"
        md.append(f"| {r.model} | {r.AUC:.4f} | [{r.AUC_ci_low:.4f}, {r.AUC_ci_high:.4f}] | {d} | {dc} |")
    md += ["", f"- 보정: {'Platt (valid로 적합)' if need else '사용 안 함 (원 확률)'}. 로지스틱 C={lr_C:g}, 결측 표시 열 {len(flag_groups)}개.", ""]
    (out / "test_metrics.md").write_text("\n".join(md), encoding="utf-8")
    a_x, a_l = met.loc[met.model == "xgb_v2", "AUC"].iloc[0], met.loc[met.model == "logit", "AUC"].iloc[0]
    log(f"**test 최종 1회**: xgb_v2 AUC {a_x:.4f}, 로지스틱 AUC {a_l:.4f}", decision=True)

    # ---- 5) interface.md q70/q90
    q70, q90 = np.quantile(p_te, [0.7, 0.9])
    tab = fill_interface(float(q70), float(q90), p_te, gv_test, f"v2 (`results/v2_frozen/`, 태그 `v2-frozen`, XGBoost K={cand['K']}, 입력 {len(feats)}개)")
    log(f"interface.md 채움: q70 {q70:.6f}, q90 {q90:.6f}, 18칸 최소 크기 {tab.n.min()} (합 {tab.n.sum():,})")

    # ---- 6) 동결 파일
    model.save_model(str(FROZEN / "models" / "xgb_v2.json"))
    joblib.dump(design.pre, FROZEN / "models" / "xgb_preprocessor.joblib")
    joblib.dump(lr_pipe, FROZEN / "models" / "logit_champion.joblib")
    if need:
        joblib.dump(calib, FROZEN / "models" / "platt_calibrator.joblib")
    feat_json = {"model_version": "v2", "K": cand["K"], "application_features": base_sel, "l0_features": l0_sel, "n_inputs": len(feats),
                 "dropped_application_features": cand["dropped_base"], "logit_missing_flag_groups": flag_groups}
    (FROZEN / "features.json").write_text(json.dumps(feat_json, ensure_ascii=False, indent=1), encoding="utf-8")
    vd = pd.read_csv(C.VAR_DICT, encoding="utf-8-sig")
    vd.loc[vd["variable"].isin(cand["dropped_base"]), "use"] = "제외(v2): gain순위밖"
    vd.to_csv(FROZEN / "variable_dict_v2_application.csv", index=False, encoding="utf-8-sig")
    shutil.copy(C.DOCS_DIR / "variable_dict_l0.csv", FROZEN / "variable_dict_v2_l0.csv")
    for f in ("k_sweep.csv", "k_sweep.md"):
        shutil.copy(L0 / f, FROZEN / f)
    cfg = {"model_version": "v2", "seed": C.SEED, "n_jobs": C.N_JOBS, "split": "SK_ID_CURR seed-42 7:1.5:1.5 (results/v1_noage_noweight/raw/splits.csv와 동일 확인)",
           "xgb_params": {k: v for k, v in XGB_PARAMS.items()}, "scale_pos_weight": 1.0, "best_iteration": int(model.best_iteration),
           "calibration": "Platt(valid)" if need else "none", "logit": {"C": lr_C, "max_iter": 5000, "valid_AUC": lr_auc},
           "gender_age_income_in_model": False, "versions": {"python": sys.version.split()[0], "xgboost": xgb.__version__, "sklearn": sklearn.__version__,
                                                             "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()},
           "features_sha256": hashlib.sha256((FROZEN / "features.json").read_bytes()).hexdigest(),
           "safe_fallback": "git tag l0-tier2-complete (dd78ed7): Tier2 10단계 190개 모형. results/l0_v2/raw/model_190.json"}
    (FROZEN / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (FROZEN / "README.md").write_text(
        "# v2 동결\n\n- `models/xgb_v2.json` 모형, `models/xgb_preprocessor.joblib` 신청서 변수 전처리(train 적합), `models/logit_champion.joblib` 로지스틱 챔피언"
        f"{', `models/platt_calibrator.joblib` 보정기' if need else ''}\n- `features.json` 입력 변수(순서 고정), `variable_dict_v2_*.csv` 변수 사전, `config.json` 시드·하이퍼파라미터·버전\n"
        "- `test_final/` test 최종 1회 평가 (ablation과 별개). `k_sweep.*` 변수 수 결정 근거\n- `raw/`(git 제외) 예측 파일\n"
        "- L0 변수는 `results/l0/cand_*.parquet`, `features_stage*.parquet` 집계에서 만든다 (git 제외). 재생성: `python -m src.l0_features`/`src.l0_tier2_run --reuse`\n", encoding="utf-8")
    log("동결 파일 저장: results/v2_frozen/")
    paths = ["results/v2_frozen", "docs/interface.md", "docs/variable_dict_l0.csv", "src/config.py", "src/l0_v2.py", "src/l0_v2_sweep.py", "src/l0_v2_final.py", "results/l0/RUNLOG.md",
             "results/l0/k_sweep.csv", "results/l0/k_sweep.md"]
    commit(paths, f"v2 동결: K={cand['K']} 입력 {len(feats)}개, test 최종 1회 AUC {a_x:.4f} (로지스틱 {a_l:.4f}), interface.md q70/q90", commit_enabled)
    if commit_enabled:
        git("tag", "v2-frozen")
        log(f"태그 v2-frozen: {git('rev-parse', '--short', 'v2-frozen')}")
