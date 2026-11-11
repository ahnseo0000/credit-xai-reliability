"""K-sweep: 190개 모형의 gain 중요도 곡선으로 변수 수 K를 정한다. test는 쓰지 않는다 (train 중요도, valid AUC).

1) 190개 모형(train으로 적합)의 total_gain을 변수 단위로 합산(원-핫은 원래 변수로 묶음)해 정렬.
2) 같은 family 안(L0 변수끼리)에서 순위상관 0.95 초과 중복을 먼저 제거 (gain 높은 쪽 유지).
3) 상위 K개(K=40,70,100; 190은 이미 학습한 모형 재사용)로 같은 분할·같은 하이퍼파라미터·같은 시드로 재학습.
   190개 모형 1회가 1시간을 넘었다면 K=60,100만.
4) valid AUC가 최고치와 0.002 이내인 가장 작은 K를 v2 후보로 확정한다.
결과: results/l0/k_sweep.csv (+ k_sweep.md), results/l0_v2/ (작업 파일).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from .l0_run import commit, log
from .l0_tier2_run import Design
from .l0_v2 import DROP_LIMIT, DUP_CORR, L0, W, WRAW, calib_rows, fit_valid, rank_matrix
from .models import feature_to_variable

K_DEFAULT = (40, 70, 100)
K_SLOW = (60, 100)       # 190개 모형 1회 학습이 1시간을 넘었을 때


def gain_by_variable(design: Design, names: list[str], base_feats: list[str], model_path) -> pd.DataFrame:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    total = booster.get_score(importance_type="total_gain")
    avg = booster.get_score(importance_type="gain")
    pre_cols = list(design.pre.get_feature_names_out())
    f2v = feature_to_variable(pre_cols, base_feats)
    all_cols = pre_cols + names
    rows = {}
    for i, col in enumerate(all_cols):
        var = f2v[col] if i < len(pre_cols) else col
        r = rows.setdefault(var, {"variable": var, "source": "신청서" if i < len(pre_cols) else "L0", "total_gain": 0.0, "n_cols": 0})
        r["total_gain"] += total.get(f"f{i}", 0.0)
        r["n_cols"] += 1
    g = pd.DataFrame(rows.values()).sort_values("total_gain", ascending=False, ignore_index=True)
    g["gain_share"] = g["total_gain"] / g["total_gain"].sum()
    g["cum_share"] = g["gain_share"].cumsum()
    return g


def phase_sweep(state, commit_enabled: bool = True):
    df, app_ids, split, base_feats, names, meta, E, design = state
    col = {n: j for j, n in enumerate(names)}
    tim = pd.read_csv(W / "timing_190.csv", encoding="utf-8-sig").iloc[0].to_dict()
    ks = K_SLOW if tim["minutes"] > 60 else K_DEFAULT
    log(f"K-sweep 시작: 190개 모형 1회 {tim['minutes']}분 → K = {list(ks)} + 190(재사용). test는 쓰지 않는다.", decision=True)

    # 1) gain 중요도
    g = gain_by_variable(design, names, base_feats, WRAW / "model_190.json")
    g.to_csv(W / "gain_importance_190.csv", index=False, encoding="utf-8-sig")
    log(f"gain 중요도(train, total_gain 변수 단위): 상위 5 " + ", ".join(f"{r.variable} {r.gain_share:.1%}" for r in g.head(5).itertuples())
        + f". gain 0인 변수 {(g.total_gain == 0).sum()}개, 상위 40개가 {g.cum_share.iloc[39]:.1%}, 70개 {g.cum_share.iloc[69]:.1%}, 100개 {g.cum_share.iloc[99]:.1%}")

    # 2) 같은 family 안 상관 0.95 초과 중복 제거 (L0 변수끼리, gain 높은 쪽 유지)
    gain = dict(zip(g.variable, g.total_gain))
    tr_idx = np.flatnonzero((split == "train").to_numpy())
    samp = np.sort(np.random.default_rng(C.SEED).choice(tr_idx, size=100_000, replace=False))
    removed: dict[str, str] = {}
    fam: dict[str, list[str]] = {}
    for n in names:
        fam.setdefault(meta[n]["family"], []).append(n)
    for f, members in fam.items():
        if len(members) < 2:
            continue
        R, ix = rank_matrix({n: E[samp, col[n]] for n in members})
        corr = (R.T @ R) / len(samp)
        kept = []
        for n in sorted(members, key=lambda n: -gain[n]):
            hit = [k for k in kept if abs(corr[ix[n], ix[k]]) > DUP_CORR]
            if hit:
                best = max(hit, key=lambda k: abs(corr[ix[n], ix[k]]))
                removed[n] = f"같은 family({f}) 안 상관 {corr[ix[n], ix[best]]:+.2f} > 0.95 중복(대상 {best}, gain 높은 쪽 유지)"
            else:
                kept.append(n)
    log(f"family 안 상관 중복 제거: L0 {len(names)}개 중 {len(removed)}개 (family별 {pd.Series({n: meta[n]['family'] for n in removed}).value_counts().to_dict()})")
    ranking = [v for v in g.variable if v not in removed]
    log(f"순위 대상 변수 {len(ranking)}개 (신청서 {sum(v in base_feats for v in ranking)} + L0 {sum(v in col for v in ranking)})")

    # 3) K별 재학습 (valid만)
    WRAW.mkdir(parents=True, exist_ok=True)
    rows = []
    r190 = {"K": 190, "n_inputs": int(tim["n_inputs"]), "n_base": len(base_feats), "n_L0": len(names), "valid_AUC": tim["valid_AUC"], "best_iter": int(tim["best_iter"]),
            "minutes": tim["minutes"], "raw_ECE": tim["raw_ECE"], "raw_Brier": tim["raw_Brier"], "platt_insample_ECE": tim["platt_insample_ECE"],
            "platt_insample_Brier": tim["platt_insample_Brier"], "platt_crossfit_ECE": tim["platt_crossfit_ECE"], "platt_crossfit_Brier": tim["platt_crossfit_Brier"],
            "note": "기존 190개 모형 재사용(family 중복 제거 없음)"}
    rows.append(r190)
    for K in ks:
        if K >= len(ranking):
            log(f"K={K}는 순위 대상({len(ranking)})보다 크거나 같아 건너뜀")
            continue
        top = set(ranking[:K])
        base_sub = [b for b in base_feats if b in top]
        l0_sub = [n for n in names if n in top]
        d = Design(df, base_sub, split)
        cols = np.array([col[n] for n in l0_sub])
        model, p, mins, ncols = fit_valid(d, E, cols)
        model.save_model(str(WRAW / f"model_K{K}.json"))
        np.save(WRAW / f"valid_p_K{K}.npy", p)
        (W / f"features_K{K}.json").write_text(json.dumps({"K": K, "base_feats": base_sub, "l0": l0_sub}, ensure_ascii=False), encoding="utf-8")
        row = {"K": K, "n_inputs": ncols, "n_base": len(base_sub), "n_L0": len(l0_sub), "best_iter": int(model.best_iteration), "minutes": round(mins, 1),
               **calib_rows(design.y["valid"], p), "note": ""}
        rows.append(row)
        log(f"K={K}: 변수 {len(base_sub)}+{len(l0_sub)}, 학습 {mins:.1f}분, valid AUC {row['valid_AUC']:.4f}, 보정 전 ECE {row['raw_ECE']:.4f}")
        del d, model
        pd.DataFrame(rows).sort_values("K").to_csv(L0 / "k_sweep.csv", index=False, encoding="utf-8-sig")  # 중간에 멈춰도 쓸 수 있게

    # 4) 확정: 최고 valid AUC와 0.002 이내인 가장 작은 K
    sw = pd.DataFrame(rows).sort_values("K", ignore_index=True)
    best = sw["valid_AUC"].max()
    sw["gap_to_best"] = best - sw["valid_AUC"]
    sw["within_0.002"] = sw["gap_to_best"] <= DROP_LIMIT
    kstar = int(sw.loc[sw["within_0.002"], "K"].min())
    sw["chosen"] = sw["K"] == kstar
    sw.to_csv(L0 / "k_sweep.csv", index=False, encoding="utf-8-sig")
    md = ["# K-sweep (gain 중요도 상위 K개, valid)", "",
          "test는 쓰지 않았다. 중요도는 190개 모형(train으로 적합)의 변수 단위 total_gain, 같은 family 안 상관 0.95 초과 중복은 먼저 제거했다 (K=190은 중복 제거 없는 기존 모형). "
          "valid는 조기 종료에도 쓰여 AUC가 약간 낙관적이다 (모든 K에 같은 방식).", "",
          "| K | 신청서 | L0 | valid AUC | 최고와의 차이 | best iter | 학습(분) | 보정 전 ECE | 보정 전 Brier | Platt cross-fit ECE | 확정 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sw.itertuples():
        md.append(f"| {r.K} | {r.n_base} | {r.n_L0} | {r.valid_AUC:.4f} | {r.gap_to_best:.4f} | {r.best_iter} | {r.minutes} | {r.raw_ECE:.4f} | {r.raw_Brier:.4f} | "
                  f"{r.platt_crossfit_ECE:.4f} | {'**예**' if r.chosen else ''} |")
    md += ["", f"- 최고 valid AUC {best:.4f}. 0.002 이내인 가장 작은 K = **{kstar}**."]
    (L0 / "k_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"K 선택(valid만): 최고 valid AUC {best:.4f}, 0.002 이내 가장 작은 K = {kstar}", decision=True)

    # 후보 변수 목록 확정 + 변수 사전 use 열 기록
    if kstar == 190:
        base_sel, l0_sel, model_path = base_feats, names, WRAW / "model_190.json"
        reason = {}
    else:
        f = json.loads((W / f"features_K{kstar}.json").read_text(encoding="utf-8"))
        base_sel, l0_sel, model_path = f["base_feats"], f["l0"], WRAW / f"model_K{kstar}.json"
        top = set(ranking[:kstar])
        reason = {n: removed.get(n, f"gain 순위 K={kstar} 밖(total_gain 점유 {g.set_index('variable').loc[n, 'gain_share']:.2%})") for n in names if n not in l0_sel}
    (W / "features_v2_candidate.json").write_text(json.dumps(
        {"K": kstar, "base_feats": base_sel, "l0": l0_sel, "model_path": str(model_path), "dropped_base": [b for b in base_feats if b not in base_sel]}, ensure_ascii=False), encoding="utf-8")
    dp = C.DOCS_DIR / "variable_dict_l0.csv"
    d = pd.read_csv(dp, encoding="utf-8-sig")
    m = d["variable"].isin(list(reason))
    d.loc[m, "use"] = d.loc[m, "variable"].map(lambda v: "제외(v2): " + ("family중복" if "family" in reason[v] else "gain순위밖"))
    d.loc[m, "note"] = d.loc[m, "variable"].map(lambda v: "v2 정리 제외 — " + reason[v])
    d.to_csv(dp, index=False, encoding="utf-8-sig")
    pd.DataFrame({"variable": list(reason), "reason": list(reason.values())}).to_csv(W / "v2_dropped_l0.csv", index=False, encoding="utf-8-sig")
    log(f"v2 후보: K={kstar}, 신청서 {len(base_sel)} + L0 {len(l0_sel)} = {len(base_sel) + len(l0_sel)}개. L0 제외 {len(reason)}개는 변수 사전 use 열에 사유 기록(지우지 않음). 제외된 신청서 변수 {[b for b in base_feats if b not in base_sel]}")
    commit(["results/l0/k_sweep.csv", "results/l0/k_sweep.md", "results/l0_v2/gain_importance_190.csv", "results/l0_v2/vif_ext_source.csv", "results/l0_v2/v2_dropped_l0.csv",
            "results/l0_v2/timing_190.csv", "results/l0_v2/features_190.json", "results/l0_v2/features_v2_candidate.json", "docs/variable_dict_l0.csv", "src/l0_v2.py", "src/l0_v2_sweep.py",
            "results/l0/RUNLOG.md"] + [f"results/l0_v2/features_K{k}.json" for k in ks],
           f"L0 v2: gain 중요도 K-sweep, valid 기준 K={kstar} 후보 확정 (test 미사용)", commit_enabled)
