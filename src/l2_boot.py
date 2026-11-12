"""L2-7: SHAP 부트스트랩 (학습 데이터 변동에 따른 전역 SHAP 중요도의 불확실성).

반복마다 train을 복원추출 -> 같은 하이퍼파라미터로 재학습(나무 수는 동결 모형의 1406개로 고정, valid·test 사용 안 함)
-> test 전체 TreeSHAP -> 변수별 mean|SHAP|, family 합 SHAP의 mean|.| 저장. B=50 구간을 먼저 만들고 시간이 허락하면 100까지.
이어하기 가능 (results/l2/raw/boot_*.npy). test는 설명 계산에만 쓰고 모형 결정에는 쓰지 않는다.
실행: python -m src.l2_boot [--b-max 100] [--hours-cap 12]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from .l2_common import L2, L2RAW, family_map, load_cache, load_model, log
from .train import XGB_PARAMS

B_FIRST = 50


def summarize(vm: np.ndarray, fm: np.ndarray, variables, fams, B: int, ref: pd.DataFrame):
    """부트스트랩 B회 요약: 변수별 mean|SHAP| 평균·95% 구간(percentile)·순위 구간."""
    vm, fm = vm[:B], fm[:B]
    rk = (-vm).argsort(axis=1).argsort(axis=1) + 1
    rows = pd.DataFrame({"variable": variables, "mean_abs_shap_full": ref.set_index("variable").loc[variables, "mean_abs_shap"].to_numpy(),
                         "boot_mean": vm.mean(0), "boot_sd": vm.std(0, ddof=1), "ci_low": np.percentile(vm, 2.5, axis=0), "ci_high": np.percentile(vm, 97.5, axis=0),
                         "rank_boot_median": np.median(rk, axis=0), "rank_ci_low": np.percentile(rk, 2.5, axis=0), "rank_ci_high": np.percentile(rk, 97.5, axis=0)})
    rows["ci_width"] = rows["ci_high"] - rows["ci_low"]
    rows["ci_width_rel"] = rows["ci_width"] / rows["boot_mean"].replace(0, np.nan)
    rows["top10_freq"] = (rk <= 10).mean(axis=0)
    rows["top20_freq"] = (rk <= 20).mean(axis=0)
    rows = rows.sort_values("boot_mean", ascending=False)
    frows = pd.DataFrame({"family": fams, "boot_mean": fm.mean(0), "ci_low": np.percentile(fm, 2.5, axis=0), "ci_high": np.percentile(fm, 97.5, axis=0)}).sort_values("boot_mean", ascending=False)
    return rows, frows


def write_outputs(vm, fm, variables, fams, B):
    ref = pd.read_csv(L2 / "global_importance.csv", encoding="utf-8-sig")
    rows, frows = summarize(vm, fm, variables, fams, B, ref)
    rows.to_csv(L2 / f"shap_bootstrap_ci_B{B}.csv", index=False, encoding="utf-8-sig")
    frows.to_csv(L2 / f"shap_bootstrap_family_ci_B{B}.csv", index=False, encoding="utf-8-sig")
    from scipy.stats import spearmanr
    rho = spearmanr(rows["boot_mean"], rows["ci_width"]).statistic
    rho_rel = spearmanr(rows["boot_mean"], rows["ci_width_rel"], nan_policy="omit").statistic
    md = [f"# L2-7 SHAP 부트스트랩 (B={B})", "",
          "- 반복마다 train을 복원추출해 재학습(나무 1406개 고정, 동결 하이퍼파라미터), 고정된 test 46,127건에서 TreeSHAP → 변수별 mean|SHAP|. 95% 구간은 percentile.",
          f"- L2-8 참고: 중요도와 CI 폭의 Spearman {rho:.3f} (절대 폭), {rho_rel:.3f} (상대 폭 = 폭/평균). 중요한 변수일수록 절대 폭은 큰 경향이 있으나 상대 폭은 작아진다면 상위 순위는 안정적이다.", "",
          "| 순위 | 변수 | mean|SHAP| (동결 모형) | 부트스트랩 평균 | 95% 구간 | 상대 폭 | 순위 95% 구간 | 상위10 비율 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(rows.head(25).itertuples(), 1):
        md.append(f"| {i} | {r.variable} | {r.mean_abs_shap_full:.4f} | {r.boot_mean:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.ci_width_rel:.1%} | {r.rank_ci_low:.0f}~{r.rank_ci_high:.0f} | {r.top10_freq:.0%} |")
    md += ["", "family 합 SHAP mean|·| 95% 구간:", "", "| family | 평균 | 95% 구간 |", "| --- | --- | --- |"]
    md += [f"| {r.family} | {r.boot_mean:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |" for r in frows.itertuples()]
    (L2 / f"shap_bootstrap_B{B}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-max", type=int, default=100)
    ap.add_argument("--hours-cap", type=float, default=12.0)
    args = ap.parse_args(argv)
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # 절전 방지
    except Exception:
        pass
    d, col_names, variables, f2v = load_cache()
    ref_model = load_model()
    n_trees = int(ref_model.best_iteration) + 1
    fam = family_map(variables)
    fams = sorted(set(fam.values()))
    idx_v = [[i for i, c in enumerate(col_names) if f2v[c] == v] for v in variables]
    Xtr, ytr, Xte = d["train"]["X"], d["train"]["y"], d["test"]["X"]
    dte = xgb.DMatrix(Xte, feature_names=col_names)
    fam_idx = [[variables.index(v) for v in variables if fam[v] == f] for f in fams]
    f_vm, f_fm = L2RAW / "boot_var_meanabs.npy", L2RAW / "boot_family_meanabs.npy"
    vm = list(np.load(f_vm)) if f_vm.exists() else []
    fm = list(np.load(f_fm)) if f_fm.exists() else []
    done0 = len(vm)
    params = {k: v for k, v in XGB_PARAMS.items() if k not in ("early_stopping_rounds", "n_estimators")}
    log(f"L2-7 부트스트랩 시작: 목표 B={args.b_max} (B={B_FIRST}에서 1차 요약), 나무 {n_trees}개 고정, 이미 완료 {done0}회, 시간 상한 {args.hours_cap}시간")
    t_start = time.time()
    n_tr = len(ytr)
    for b in range(done0, args.b_max):
        t0 = time.time()
        rs = np.random.default_rng(C.SEED + 1000 + b)
        rows = rs.integers(0, n_tr, n_tr)
        m = xgb.XGBClassifier(n_estimators=n_trees, scale_pos_weight=1.0, **{**params, "random_state": C.SEED + b})
        m.fit(Xtr[rows], ytr[rows], verbose=0)
        c = m.get_booster().predict(dte, pred_contribs=True)
        S = np.stack([c[:, ix].sum(axis=1) for ix in idx_v], axis=1)
        del c, m
        vm.append(np.abs(S).mean(axis=0))
        fm.append(np.array([np.abs(S[:, ix].sum(axis=1)).mean() for ix in fam_idx]))
        np.save(f_vm, np.array(vm))
        np.save(f_fm, np.array(fm))
        el = (time.time() - t0) / 60
        total_h = (time.time() - t_start) / 3600
        log(f"L2-7 부트스트랩 {b + 1}/{args.b_max} 완료 ({el:.1f}분, 이번 실행 누적 {total_h:.2f}시간, 1위 변수 mean|SHAP| {vm[-1].max():.4f})")
        if b + 1 == B_FIRST:
            write_outputs(np.array(vm), np.array(fm), variables, fams, B_FIRST)
            log(f"L2-7 B={B_FIRST} 1차 요약 저장 (shap_bootstrap_B{B_FIRST}.md). 계속해서 B={args.b_max}까지 진행", decision=True)
        if total_h * (b + 2 - done0) / max(b + 1 - done0, 1) > args.hours_cap and b + 1 >= B_FIRST:
            log(f"시간 상한({args.hours_cap}시간) 때문에 B={b + 1}에서 중단", decision=True)
            break
    B = len(vm)
    write_outputs(np.array(vm), np.array(fm), variables, fams, B)
    log(f"L2-7 완료: B={B} 요약 저장 (shap_bootstrap_B{B}.md)", decision=True)


if __name__ == "__main__":
    main()
