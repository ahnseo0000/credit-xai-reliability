"""Phase 2 추가: 외부점수(EXT_SOURCE 계열) 제외 모델. v2 동결 파일은 읽기만 하고 수정하지 않는다. test는 읽지 않는다 (train 학습, valid 평가).

- 제외: v2 입력 100개 변수 중 EXT_SOURCE_1/2/3과 이들에서 만든 파생(MEAN/MIN/MAX/PROD) 7개.
- 학습: v2와 같은 하이퍼파라미터(results/v2_frozen/config.json의 xgb_params), 같은 시드 42, scale_pos_weight 1.0, 튜닝 없음, valid AUC 조기 종료 150 (v2와 같음).
- SHAP: v2 L2와 같은 방식(XGBoost pred_contribs 정확 TreeSHAP, 조기 종료 지점까지의 나무, 원-핫 열은 변수로 합산, family는 행별 합의 mean|.|).
  단, L2는 test에서 계산했으나 이번에는 test 사용 금지이므로 두 모형(v2 동결 모형, 새 모형) 모두 valid 46,127건에서 같은 방식으로 계산한다.
실행: python -m src.p2_ext_excluded
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from .l2_common import FROZEN, L2RAW, family_map, load_model
from .metrics import bootstrap_auc, credit_metrics
from .models import feature_to_variable
from .p2_woe import P2

OUT = P2 / "ext_excluded"
EXT_VARS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX", "EXT_SOURCE_PROD"]


def log(msg: str, decision: bool = False):
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def var_shap(booster, X, col_names, f2v, variables, ir):
    """정확한 TreeSHAP을 변수 단위로 합친 (n, 변수) 행렬. 가법성(합 + base = margin)을 확인한다."""
    dm = xgb.DMatrix(X, feature_names=col_names)
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=ir)
    idx = {v: [i for i, c in enumerate(col_names) if f2v[c] == v] for v in variables}
    S = np.stack([contrib[:, idx[v]].sum(axis=1) for v in variables], axis=1).astype(np.float32)
    margin = booster.predict(dm, output_margin=True, iteration_range=ir)
    assert np.allclose(S.sum(axis=1) + contrib[:, -1], margin, atol=1e-3), "SHAP 가법성 실패"
    return S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feat = json.loads((FROZEN / "features.json").read_text(encoding="utf-8"))
    variables = feat["application_features"] + feat["l0_features"]
    col_names = json.loads((L2RAW / "col_names.json").read_text(encoding="utf-8"))
    f2v = feature_to_variable(col_names, variables)
    fam = family_map(variables)
    kn = pd.read_csv(C.DOCS_DIR / "variable_dict_l0.csv", encoding="utf-8-sig").set_index("variable")["korean_name"].to_dict()

    # ---- 1. 제외 대상 찾기: 이름에 EXT_SOURCE가 들어간 변수 + 입력 열 이름 확인 + 변수 사전 설명의 외부 점수 파생
    by_name = [v for v in variables if "EXT_SOURCE" in v.upper()]
    d0 = pd.read_csv(C.DOCS_DIR / "variable_dict_l0.csv", encoding="utf-8-sig").set_index("variable")
    by_desc = [v for v in variables if v in d0.index and v not in by_name and any(k in str(d0.loc[v, "description"]) + str(d0.loc[v, "korean_name"]) for k in ("외부 신용평가", "외부 점수", "EXT_SOURCE"))]
    cols_ext = [c for c in col_names if "EXT_SOURCE" in c.upper()]
    print("이름에 EXT_SOURCE 포함 변수:", by_name)
    print("설명이 외부 점수를 쓰는 다른 변수:", by_desc or "없음")
    print("모형 입력 열 중 EXT_SOURCE 포함:", cols_ext)
    assert sorted(by_name) == sorted(EXT_VARS) and not by_desc, "제외 목록이 사전 정의와 다르다 - 확인 필요"
    kind = {"EXT_SOURCE_1": "원본 신청서 변수", "EXT_SOURCE_2": "원본 신청서 변수", "EXT_SOURCE_3": "원본 신청서 변수"}
    ex = pd.DataFrame({"variable": EXT_VARS, "korean_name": [kn.get(v, "") for v in EXT_VARS], "family_final": [fam[v] for v in EXT_VARS],
                       "kind": [kind.get(v, "EXT_SOURCE_1~3에서 만든 L0 집계 (평균/최소/최대/곱)") for v in EXT_VARS],
                       "reason": ["이름에 EXT_SOURCE 포함"] * len(EXT_VARS)})
    ex.to_csv(OUT / "excluded_vars.csv", index=False, encoding="utf-8-sig")
    keep = [v for v in variables if v not in EXT_VARS]

    log("외부점수 제외 모델 시작 (v2 동결 파일 수정 안 함, test 미사용): 이 단계 기록은 모두 results/phase2/ext_excluded/ 에 저장")
    log("**[해석 기준, 결과를 보기 전에 기록]** valid AUC 하락폭, 상위 SHAP 요인과 family 변화를 기술적으로 보고. 우열 판정이나 가설 검정은 하지 않음.", decision=True)
    log(f"제외 대상 {len(EXT_VARS)}개 (v2 입력 {len(variables)}개 중): {EXT_VARS}. 이름에 EXT_SOURCE가 들어간 변수가 전부이고, 변수 사전 설명에서 외부 점수를 재료로 쓴 다른 변수(비율/상호작용)는 없음. "
        f"남는 변수 {len(keep)}개 (신청서 {len([v for v in feat['application_features'] if v not in EXT_VARS])}개 + L0 {len([v for v in feat['l0_features'] if v not in EXT_VARS])}개). 모형 입력 열 중 EXT 포함 {len(cols_ext)}개 모두 제외", decision=True)
    log("학습 조건: v2 config.json의 xgb_params 그대로(n_estimators 3000, lr 0.03, depth 5, min_child_weight 20, subsample 0.8, colsample 0.7, reg_lambda 5, 조기 종료 150 on valid AUC, 시드 42, n_jobs 4), "
        "scale_pos_weight 1.0, 보정 없음, 튜닝 없음. 조기 종료에 valid를 쓰는 것도 v2와 같다", decision=True)
    log("SHAP 방식 결정: v2 L2는 test에서 계산했지만 이번 작업은 test 사용 금지라, v2 동결 모형과 새 모형 모두 valid 46,127건에서 같은 방식(정확한 TreeSHAP pred_contribs, 조기 종료 지점까지 나무, 원-핫은 변수로 합산)으로 계산해 비교한다. "
        "L2의 test 기준 v2 순위는 참고로만 인용(이미 저장된 결과 파일, 데이터를 다시 읽지 않음)", decision=True)

    # ---- 2. 학습 (train / valid만)
    Xtr, ytr = np.load(L2RAW / "X_train.npy", mmap_mode="r"), np.load(L2RAW / "y_train.npy")
    Xva, yva = np.load(L2RAW / "X_valid.npy", mmap_mode="r"), np.load(L2RAW / "y_valid.npy")
    cidx = [i for i, c in enumerate(col_names) if f2v[c] in keep]
    cn_new = [col_names[i] for i in cidx]
    Xtr_n, Xva_n = np.asarray(Xtr[:, cidx], dtype=np.float32), np.asarray(Xva[:, cidx], dtype=np.float32)
    cfg = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    params = dict(cfg["xgb_params"])
    assert params["random_state"] == 42
    model = xgb.XGBClassifier(scale_pos_weight=cfg["scale_pos_weight"], **params)
    model.fit(Xtr_n, ytr, eval_set=[(Xva_n, yva)], verbose=0)
    ir_new = (0, int(model.best_iteration) + 1)
    p_new = model.predict_proba(Xva_n, iteration_range=ir_new)[:, 1]
    (P2 / "raw").mkdir(exist_ok=True)
    np.save(P2 / "raw" / "valid_p_ext_excluded.npy", p_new)
    log(f"새 모형 학습 끝: 입력 열 {len(cidx)}, best_iteration {model.best_iteration} (v2는 {cfg['best_iteration']}), valid AUC {roc_auc(yva, p_new):.4f}")

    # ---- v2 동결 모형 (읽기 전용): 예측 재현 확인
    v2 = load_model()
    ir_v2 = (0, int(v2.best_iteration) + 1)
    p_v2 = v2.predict_proba(np.asarray(Xva, dtype=np.float32), iteration_range=ir_v2)[:, 1]
    saved = np.load(C.RESULTS_DIR / "l0_v2" / "raw" / "valid_p_K100.npy")
    assert np.allclose(p_v2, saved, atol=1e-5), "v2 동결 모형의 valid 예측이 저장값과 다르다"
    log(f"v2 동결 모형 valid 예측 재현 확인 통과 (최대 차이 {np.abs(p_v2 - saved).max():.1e}), best_iteration {v2.best_iteration}")

    # ---- 3. 지표 표 + 쌍 부트스트랩
    M = {"v2 (100변수)": credit_metrics(yva, p_v2), f"외부점수 제외 ({len(keep)}변수)": credit_metrics(yva, p_new)}
    k_v2, k_new = list(M)
    bt = bootstrap_auc(yva, {k_v2: p_v2, k_new: p_new}, baseline=k_v2).set_index("model")
    r = bt.loc[k_new]
    diff, dlo, dhi = r["diff_vs_" + k_v2], r["diff_ci_low"], r["diff_ci_high"]

    # ---- 4. SHAP (valid, 두 모형)
    S_v2 = var_shap(v2.get_booster(), np.asarray(Xva, dtype=np.float32), col_names, f2v, variables, ir_v2)
    f2v_new = {c: f2v[c] for c in cn_new}
    S_new = var_shap(model.get_booster(), Xva_n, cn_new, f2v_new, keep, ir_new)
    imp_v2 = pd.Series(np.abs(S_v2).mean(0), index=variables)
    imp_new = pd.Series(np.abs(S_new).mean(0), index=keep)
    rk_v2, rk_new = imp_v2.rank(ascending=False, method="first").astype(int), imp_new.rank(ascending=False, method="first").astype(int)

    def top_table(imp, rk, n=15):
        return [(v, rk[v], imp[v], fam[v]) for v in imp.sort_values(ascending=False).index[:n]]

    rows = []
    for v in imp_v2.sort_values(ascending=False).index[:15]:
        rows.append({"model": "v2", "rank": int(rk_v2[v]), "variable": v, "korean_name": kn.get(v, ""), "family_final": fam[v], "mean_abs_shap": imp_v2[v], "rank_in_other_model": int(rk_new[v]) if v in rk_new.index else np.nan,
                     "excluded": "Y" if v in EXT_VARS else ""})
    for v in imp_new.sort_values(ascending=False).index[:15]:
        rows.append({"model": "ext_excluded", "rank": int(rk_new[v]), "variable": v, "korean_name": kn.get(v, ""), "family_final": fam[v], "mean_abs_shap": imp_new[v], "rank_in_other_model": int(rk_v2[v]), "excluded": ""})
    top = pd.DataFrame(rows)
    top.to_csv(OUT / "shap_top15.csv", index=False, encoding="utf-8-sig")

    # family: 행별 합의 mean|.| (L2 family_importance와 같은 정의) + 변수별 mean|SHAP|의 합
    def fam_tab(S, vs):
        fs = {}
        for v in vs:
            fs.setdefault(fam[v], []).append(vs.index(v))
        t = pd.DataFrame({"mean_abs_shap_of_family_sum": {f: float(np.abs(S[:, ix].sum(axis=1)).mean()) for f, ix in fs.items()},
                          "sum_of_var_mean_abs_shap": {f: float(np.abs(S[:, ix]).mean(0).sum()) for f, ix in fs.items()},
                          "n_vars": {f: len(ix) for f, ix in fs.items()}})
        t["share_family_sum"] = t["mean_abs_shap_of_family_sum"] / t["mean_abs_shap_of_family_sum"].sum()
        t["rank"] = t["mean_abs_shap_of_family_sum"].rank(ascending=False, method="first").astype(int)
        return t

    ft_v2, ft_new = fam_tab(S_v2, variables), fam_tab(S_new, keep)
    fam_all = sorted(set(ft_v2.index) | set(ft_new.index), key=lambda f: -ft_v2["mean_abs_shap_of_family_sum"].get(f, 0))
    fr = []
    for f in fam_all:
        a, b = ft_v2.loc[f] if f in ft_v2.index else None, ft_new.loc[f] if f in ft_new.index else None
        fr.append({"family_final": f, "n_vars_v2": int(a.n_vars) if a is not None else 0, "n_vars_new": int(b.n_vars) if b is not None else 0,
                   "v2_mean_abs_shap_family_sum": a.mean_abs_shap_of_family_sum if a is not None else np.nan, "v2_share": a.share_family_sum if a is not None else np.nan, "v2_rank": int(a["rank"]) if a is not None else np.nan,
                   "new_mean_abs_shap_family_sum": b.mean_abs_shap_of_family_sum if b is not None else np.nan, "new_share": b.share_family_sum if b is not None else np.nan, "new_rank": int(b["rank"]) if b is not None else np.nan,
                   "v2_sum_of_var_mean_abs": a.sum_of_var_mean_abs_shap if a is not None else np.nan, "new_sum_of_var_mean_abs": b.sum_of_var_mean_abs_shap if b is not None else np.nan})
    fdf = pd.DataFrame(fr)
    fdf["change_in_family_sum"] = fdf["new_mean_abs_shap_family_sum"] - fdf["v2_mean_abs_shap_family_sum"]
    fdf.to_csv(OUT / "family_shap.csv", index=False, encoding="utf-8-sig")

    # 상위 10 겹침
    top10_v2 = list(imp_v2.sort_values(ascending=False).index[:10])
    top10_new = list(imp_new.sort_values(ascending=False).index[:10])
    stay = [v for v in top10_v2 if v in top10_new]
    excl_in10 = [v for v in top10_v2 if v in EXT_VARS]
    l2 = pd.read_csv(C.RESULTS_DIR / "l2" / "global_importance.csv", encoding="utf-8-sig").sort_values("mean_abs_shap", ascending=False)
    l2_top10 = list(l2.variable[:10])
    ov_l2 = len(set(l2_top10) & set(top10_v2))

    # ---- 보고서
    n_tr = len(ytr)
    md = ["# Phase 2 추가: 외부점수 제외 모델 (valid)", "",
          "- **해석 기준(결과 보기 전에 RUNLOG에 기록): valid AUC 하락폭, 상위 SHAP 요인과 family 변화를 기술적으로 보고한다. 우열 판정이나 가설 검정은 하지 않는다.**",
          f"- train {n_tr:,}건으로 학습, valid {len(yva):,}건으로 평가. **test는 읽지 않았다.** v2 동결 파일(모델, 변수 목록, 하이퍼파라미터)은 읽기만 했다.",
          f"- 제외한 변수 {len(EXT_VARS)}개: " + ", ".join(EXT_VARS) + f" (남은 변수 {len(keep)}개). 이름에 EXT_SOURCE가 들어간 변수가 전부이며, 외부 점수를 재료로 쓰는 비율·상호작용 변수는 v2 100개 안에 없었다 (`excluded_vars.csv`).",
          f"- 새 모형은 v2와 같은 하이퍼파라미터·시드 42·scale_pos_weight 1.0·조기 종료 150(valid AUC), 튜닝 없음. best_iteration {model.best_iteration} (v2 {v2.best_iteration}).", "",
          "## 1. valid 지표", "", "| 모형 | 변수 | AUC | Gini | KS | Brier | ECE(10분위) | 1분위 lift |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, nvar in ((k_v2, 100), (k_new, len(keep))):
        m = M[k]
        md.append(f"| {k} | {nvar} | {m['AUC']:.4f} | {m['Gini']:.4f} | {m['KS']:.4f} | {m['Brier']:.4f} | {m['ECE_10q']:.4f} | {m['top_decile_lift']:.2f} |")
    m0, m1 = M[k_v2], M[k_new]
    md.append(f"| 차이 (제외 − v2) | | {m1['AUC'] - m0['AUC']:+.4f} | {m1['Gini'] - m0['Gini']:+.4f} | {m1['KS'] - m0['KS']:+.4f} | {m1['Brier'] - m0['Brier']:+.4f} | {m1['ECE_10q'] - m0['ECE_10q']:+.4f} | {m1['top_decile_lift'] - m0['top_decile_lift']:+.2f} |")
    md += ["", f"- **AUC 차이(제외 − v2) = {diff:+.4f}, 95% 구간 [{dlo:+.4f}, {dhi:+.4f}]** (같은 재표본 쌍 부트스트랩, B=1000, 시드 42). AUC 하락 {-diff:.4f}는 v2 AUC의 {-diff / m0['AUC']:.1%}, 모형 AUC 구간: v2 [{bt.loc[k_v2, 'AUC_ci_low']:.4f}, {bt.loc[k_v2, 'AUC_ci_high']:.4f}], 제외 [{bt.loc[k_new, 'AUC_ci_low']:.4f}, {bt.loc[k_new, 'AUC_ci_high']:.4f}].",
           f"- 참고(스코어카드·22변수 GBM과의 위치): 외부점수를 쓰는 22변수 GBM AUC 0.7749, 스코어카드 0.7544 (compare_valid_v2.md). 외부점수 제외 모형의 AUC는 {m1['AUC']:.4f}.", "",
           "## 2. SHAP 상위 15개 변수 (valid, 정확한 TreeSHAP, mean|SHAP| = log-odds 단위)", "",
           "SHAP은 L2와 같은 방식이지만 test 대신 valid 46,127건에서 계산했다 (두 모형 모두). 변수별 값은 원-핫 열을 합친 것이다.", "",
           "| 순위 | v2 변수 | family | mean abs SHAP | 새 모형 순위 | | 순위 | 새 모형 변수 | family | mean abs SHAP | v2 순위 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in range(15):
        a, b = top_table(imp_v2, rk_v2)[i], top_table(imp_new, rk_new)[i]
        ra = rk_new.get(a[0]) if a[0] in rk_new.index else None
        md.append(f"| {a[1]} | {a[0]}{' (제외됨)' if a[0] in EXT_VARS else ''} | {a[3]} | {a[2]:.4f} | {ra if ra is not None else '제외'} | | {b[1]} | {b[0]} | {b[3]} | {b[2]:.4f} | {rk_v2[b[0]]} |")
    md += ["", f"- **v2 상위 10위 변수 중 새 모형에서도 상위 10위 안: {len(stay)}개 / 10개.** v2 상위 10위 중 제외된(외부점수) 변수 {len(excl_in10)}개({', '.join(excl_in10) or '없음'})는 당연히 빠졌고, 남은 {10 - len(excl_in10)}개 중 새 상위 10위에 남은 변수는 {len(stay)}개: {', '.join(stay) or '없음'}.",
           f"- v2 상위 10위 중 새 모형에서 10위 밖으로 간 변수(제외된 변수 제외): " + (", ".join(f"{v}({int(rk_v2[v])}위→{int(rk_new[v])}위)" for v in top10_v2 if v not in EXT_VARS and v not in top10_new) or "없음"),
           f"- 새 모형 상위 10위에 새로 들어온 변수(v2 상위 10위 밖): " + (", ".join(f"{v}({int(rk_v2[v])}위→{int(rk_new[v])}위)" for v in top10_new if v not in top10_v2) or "없음"),
           f"- 참고: valid로 다시 계산한 v2 상위 10위와 L2(test 기준) v2 상위 10위의 겹침은 {ov_l2}/10 (L2 상위 10위: {', '.join(l2_top10)}).",
           f"- 새 모형 mean|SHAP| 합 {imp_new.sum():.3f}, v2 {imp_v2.sum():.3f}. v2에서 외부점수 7개 변수의 mean|SHAP| 합은 {imp_v2[EXT_VARS].sum():.3f} ({imp_v2[EXT_VARS].sum() / imp_v2.sum():.1%}), `EXT_SOURCE_MEAN` 단독 {imp_v2['EXT_SOURCE_MEAN']:.3f}.", "",
           "## 3. family별 평균 |SHAP| (valid, 행별 family 합의 mean|.|; L2 family_importance와 같은 정의)", "",
           "| family | 변수 수 (v2→새) | v2 값 | v2 비중 | v2 순위 | 새 값 | 새 비중 | 새 순위 | 값 변화 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in fdf.sort_values("v2_mean_abs_shap_family_sum", ascending=False).itertuples():
        nz = lambda x, f: ("-" if pd.isna(x) else f.format(x))
        md.append(f"| {r.family_final} | {r.n_vars_v2}→{r.n_vars_new} | {nz(r.v2_mean_abs_shap_family_sum, '{:.4f}')} | {nz(r.v2_share, '{:.1%}')} | {nz(r.v2_rank, '{:.0f}')} | {nz(r.new_mean_abs_shap_family_sum, '{:.4f}')} | {nz(r.new_share, '{:.1%}')} | {nz(r.new_rank, '{:.0f}')} | {nz(r.change_in_family_sum, '{:+.4f}')} |")
    md += ["", "- 값은 SHAP이 log-odds 척도라 합이 1이 아니다. 비중은 family 합 값의 비중이다. 외부점수 family는 새 모형에서 사라졌다 (변수 7→0). 다른 family의 값은 외부점수가 하던 일을 다른 변수가 나눠 받은 몫을 포함한다.",
           "- 변수별 mean|SHAP|의 단순 합(상쇄 없음) 열은 `family_shap.csv`에 같이 저장했다.", "",
           "## 4. 한계", "",
           "- 이 문서는 기술 보고이다. 우열 판정이나 가설 검정은 하지 않았다.",
           "- 단일 학습·단일 valid 분할의 결과이고 튜닝은 하지 않았다 (v2와 같은 조건 유지). 조기 종료에 valid를 쓰는 점도 v2와 같다.",
           "- SHAP 순위는 두 모형의 학습 방식 차이(외부점수가 없을 때 다른 변수가 대신 쓰이는 정도)를 포함한다. 순위 변화가 곧 해당 변수의 인과적 중요도 변화는 아니다.",
           "- SHAP은 test가 아니라 valid에서 계산했다 (test 사용 금지). L2의 test 기준 값과 완전히 같지는 않다.", ""]
    (OUT / "compare_valid.md").write_text("\n".join(md), encoding="utf-8")
    log(f"결과: valid AUC v2 {m0['AUC']:.4f} → 제외 {m1['AUC']:.4f} (차이 {diff:+.4f}, 95% [{dlo:+.4f}, {dhi:+.4f}]), KS {m0['KS']:.4f} → {m1['KS']:.4f}, Brier {m0['Brier']:.4f} → {m1['Brier']:.4f}, ECE {m0['ECE_10q']:.4f} → {m1['ECE_10q']:.4f}. "
        f"v2 상위 10위 중 새 상위 10위에 남은 변수 {len(stay)}/10 (제외된 것 {len(excl_in10)}개 포함). 저장: ext_excluded/compare_valid.md, excluded_vars.csv, shap_top15.csv, family_shap.csv. test 미사용, 커밋 전")
    return md


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


if __name__ == "__main__":
    main()
