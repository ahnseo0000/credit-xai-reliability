"""L2-1~5: test 전체 TreeSHAP, mean|SHAP|, XGBoost 내장 FI, Permutation Importance, 기법 간 Spearman, family 합산.

동결 모형(v2_frozen)만 읽는다. test는 '설명'에만 쓰고 모형·변수·하이퍼파라미터를 바꾸지 않는다.
원-핫 더미는 원래 변수로 합친다 (SHAP은 가법). Permutation은 원-핫 열을 한 변수로 묶어 같이 섞는다.
산출: results/l2/global_importance.csv, family_importance.csv, rank_corr.csv, l2_global_summary.md, raw/shap_test.npy(변수 단위)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import config as C
from .l2_common import FROZEN, L2, L2RAW, family_map, load_cache, load_model, log

PERM_REPEATS = 5
import sys
REUSE_PERM = '--reuse-perm' in sys.argv


def aggregate(contrib: np.ndarray, col_names, variables, f2v):
    """(n, 열+1)의 pred_contribs를 변수 단위 (n, 변수) + base value로 합친다."""
    base = contrib[:, -1]
    idx = {v: [i for i, c in enumerate(col_names) if f2v[c] == v] for v in variables}
    S = np.stack([contrib[:, idx[v]].sum(axis=1) for v in variables], axis=1).astype(np.float32)
    return S, base, idx


def main():
    d, col_names, variables, f2v = load_cache()
    model = load_model()
    booster = model.get_booster()
    ir = (0, int(model.best_iteration) + 1)   # 조기 종료 지점까지의 나무만 쓴다 (동결 예측과 동일)
    X, y = d["test"]["X"], d["test"]["y"]
    ids = d["test"]["ids"]
    fam = family_map(variables)

    # ---- 재현 확인: 동결 예측과 일치하는가
    pr = pd.read_csv(FROZEN / "raw" / "test_predictions.csv").set_index(C.ID_COL).loc[ids]
    dm = xgb.DMatrix(X, feature_names=col_names)
    p = booster.predict(dm, iteration_range=ir)
    assert np.allclose(p, pr["pd_xgb_raw"].to_numpy(), atol=1e-5), "캐시 입력으로 계산한 예측이 동결 test 예측과 다르다"
    log(f"L2-1 시작: 캐시 입력 재현 확인 통과 (최대 차이 {np.abs(p - pr['pd_xgb_raw'].to_numpy()).max():.2e}), 입력 열 {len(col_names)} → 변수 {len(variables)}")

    # ---- L2-1 TreeSHAP (test 전체)
    t0 = time.time()
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=ir)
    S, base, idx = aggregate(contrib, col_names, variables, f2v)
    margin = booster.predict(dm, output_margin=True, iteration_range=ir)
    assert np.allclose(S.sum(axis=1) + base, margin, atol=1e-3), "SHAP 합이 log-odds와 다르다"
    np.save(L2RAW / "shap_test.npy", S)
    np.save(L2RAW / "shap_base_test.npy", base)
    np.save(L2RAW / "shap_variables.npy", np.array(variables))
    log(f"L2-1 TreeSHAP 완료: {S.shape}, {time.time() - t0:.0f}초. 가법성 확인(SHAP 합 + 기준값 = log-odds) 통과, 기준값 {float(base[0]):.4f}")
    del contrib

    # ---- L2-2/3 mean|SHAP|, 내장 FI
    msh = np.abs(S).mean(axis=0)
    gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")
    key = {c: (c if c in gain or c in weight else f"f{i}") for i, c in enumerate(col_names)}   # 저장 모형은 특성 이름이 없으면 f0.. 로 기록
    assert any(k in gain for k in key.values()), "내장 FI 키를 못 찾음"
    g = pd.Series({v: sum(gain.get(key[c], 0.0) for c in col_names if f2v[c] == v) for v in variables})
    w = pd.Series({v: sum(weight.get(key[c], 0.0) for c in col_names if f2v[c] == v) for v in variables})

    # ---- L2-4 Permutation Importance (test AUC 하락, 변수 단위, 원-핫은 같이 섞음)
    base_auc = roc_auc_score(y, p)
    old = L2 / 'global_importance.csv'
    if REUSE_PERM and old.exists():
        o = pd.read_csv(old, encoding='utf-8-sig').set_index('variable')
        perm = {v: (float(o.loc[v, 'perm_auc_drop']), float(o.loc[v, 'perm_auc_drop_sd'])) for v in variables}
        log('L2-4 Permutation 결과 재사용 (이전 실행값, 내장 FI 열 이름 수정 후 표만 다시 만듦)')
    else:
        rng = np.random.default_rng(C.SEED)
        t0 = time.time()
        perm = {}
        Xp = X.copy()
        for k, v in enumerate(variables):
            cols_i = idx[v]
            drops = []
            for r in range(PERM_REPEATS):
                order = rng.permutation(len(X))
                Xp[:, cols_i] = X[order][:, cols_i]
                pp = booster.inplace_predict(Xp, iteration_range=ir, validate_features=False)
                drops.append(base_auc - roc_auc_score(y, pp))
            Xp[:, cols_i] = X[:, cols_i]
            perm[v] = (float(np.mean(drops)), float(np.std(drops)))
            if (k + 1) % 20 == 0:
                log(f"L2-4 Permutation {k + 1}/{len(variables)} ({time.time() - t0:.0f}초)")
    pm = pd.Series({v: a for v, (a, _) in perm.items()})
    ps = pd.Series({v: s for v, (_, s) in perm.items()})

    tab = pd.DataFrame({"variable": variables, "family": [fam[v] for v in variables], "mean_abs_shap": msh, "xgb_total_gain": g.values,
                        "xgb_weight": w.values, "perm_auc_drop": pm.values, "perm_auc_drop_sd": ps.values})
    tab["shap_share"] = tab["mean_abs_shap"] / tab["mean_abs_shap"].sum()
    tab["gain_share"] = tab["xgb_total_gain"] / tab["xgb_total_gain"].sum()
    for c in ("mean_abs_shap", "xgb_total_gain", "perm_auc_drop"):
        tab[f"rank_{c}"] = tab[c].rank(ascending=False, method="min").astype(int)
    tab = tab.sort_values("rank_mean_abs_shap")
    tab.to_csv(L2 / "global_importance.csv", index=False, encoding="utf-8-sig")

    # ---- L2-5 Spearman (기법 간)
    pairs = [("mean_abs_shap", "xgb_total_gain"), ("mean_abs_shap", "perm_auc_drop"), ("xgb_total_gain", "perm_auc_drop")]
    rc = []
    for a, b in pairs:
        rho = spearmanr(tab[a], tab[b]).statistic
        top = tab.nlargest(20, a)
        rho20 = spearmanr(top[a], top[b]).statistic
        rc.append({"기법 A": a, "기법 B": b, "Spearman(전체 100)": rho, "Spearman(A 상위20)": rho20})
    rc = pd.DataFrame(rc)
    rc.to_csv(L2 / "rank_corr.csv", index=False, encoding="utf-8-sig")
    tab["rank_gap"] = (tab[["rank_mean_abs_shap", "rank_xgb_total_gain", "rank_perm_auc_drop"]].max(axis=1)
                       - tab[["rank_mean_abs_shap", "rank_xgb_total_gain", "rank_perm_auc_drop"]].min(axis=1))
    div = tab.sort_values("rank_gap", ascending=False).head(10)

    # ---- family 합산
    Sf = pd.DataFrame(S, columns=variables).T.groupby(fam).sum().T          # 행 단위 합 (가법)
    f_abs_sum = pd.Series(np.abs(S).mean(axis=0), index=variables).groupby(fam).sum()   # 변수별 mean|SHAP|의 합
    f_net = Sf.abs().mean()                                                # family 합 SHAP의 mean|.| (상쇄 반영)
    ftab = pd.DataFrame({"n_vars": pd.Series(fam).value_counts(), "mean_abs_shap_of_family_sum": f_net, "sum_of_var_mean_abs_shap": f_abs_sum,
                         "xgb_total_gain": g.groupby(fam).sum(), "perm_auc_drop_sum": pm.groupby(fam).sum()})
    ftab["share_family_sum"] = ftab["mean_abs_shap_of_family_sum"] / ftab["mean_abs_shap_of_family_sum"].sum()
    ftab["gain_share"] = ftab["xgb_total_gain"] / ftab["xgb_total_gain"].sum()
    ftab = ftab.sort_values("mean_abs_shap_of_family_sum", ascending=False)
    ftab.index.name = "family"
    ftab.to_csv(L2 / "family_importance.csv", encoding="utf-8-sig")

    # ---- 요약 md
    md = ["# L2-1~5 전역 중요도 (v2 동결 모형, test 46,127건)", "",
          f"- 모형 base AUC(test) {base_auc:.4f}. SHAP은 log-odds 단위, 원-핫은 원 변수로 합침(가법). Permutation: 변수별 {PERM_REPEATS}회 섞어 test AUC 하락 평균 (시드 42).",
          "- 이 표는 설명용이다. 이 결과로 변수·모형을 바꾸지 않는다 (v2 동결).", "",
          "## 상위 20개 (mean|SHAP| 순)", "", "| 순위 | 변수 | family | mean|SHAP| | 비중 | 내장FI 순위 | Perm AUC↓ | Perm 순위 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in tab.head(20).itertuples():
        md.append(f"| {r.rank_mean_abs_shap} | {r.variable} | {r.family} | {r.mean_abs_shap:.4f} | {r.shap_share:.1%} | {r.rank_xgb_total_gain} | {r.perm_auc_drop:.4f} | {r.rank_perm_auc_drop} |")
    md += ["", "## 기법 간 Spearman 순위상관", "", "| 기법 A | 기법 B | 전체 100개 | A 상위 20개 |", "| --- | --- | --- | --- |"]
    md += [f"| {r['기법 A']} | {r['기법 B']} | {r['Spearman(전체 100)']:.3f} | {r['Spearman(A 상위20)']:.3f} |" for _, r in rc.iterrows()]
    md += ["", "## 기법 간 순위가 가장 엇갈리는 변수 10개 (원인 메모는 사람이 보고 채움)", "", "| 변수 | family | SHAP 순위 | 내장FI 순위 | Perm 순위 | 순위 차 |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.variable} | {r.family} | {r.rank_mean_abs_shap} | {r.rank_xgb_total_gain} | {r.rank_perm_auc_drop} | {r.rank_gap} |" for r in div.itertuples()]
    md += ["", "## family 단위 SHAP 합산", "", "family 안 변수의 SHAP을 행마다 더한 값의 mean|·|이다 (같은 family 안에서 서로 상쇄되는 효과 반영). 비교용으로 변수별 mean|SHAP| 단순 합도 함께 둔다.", "",
           "| family | 변수 수 | mean|family SHAP| | 비중 | 변수별 합 | 내장FI 비중 | Perm AUC↓ 합 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {i} | {int(r.n_vars)} | {r.mean_abs_shap_of_family_sum:.4f} | {r.share_family_sum:.1%} | {r.sum_of_var_mean_abs_shap:.4f} | {r.gain_share:.1%} | {r.perm_auc_drop_sum:.4f} |" for i, r in ftab.iterrows()]
    (L2 / "l2_global_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"L2-2~5 완료: 1위 {tab.iloc[0].variable} ({tab.iloc[0].shap_share:.1%}), Spearman SHAP–내장FI {rc.iloc[0, 2]:.3f}, SHAP–Perm {rc.iloc[1, 2]:.3f}, 내장FI–Perm {rc.iloc[2, 2]:.3f}, "
        f"family 1위 {ftab.index[0]} ({ftab.iloc[0].share_family_sum:.1%})")


if __name__ == "__main__":
    main()
