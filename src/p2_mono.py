"""Phase 2-4b: 스코어카드와 같은 22개 변수로 단조 제약 XGBoost. train 5-fold CV로 Optuna 튜닝(40 trial), valid로 평가. test 미사용.

제약 방향 근거: 스코어카드 구간화(woe_bins.csv)에서 결측을 뺀 구간의 train 부도율 추세. 스코어카드 계수가 전부 음수(WoE가 클수록 PD 낮음)이므로
'부도율이 구간 번호와 함께 오르면 +1(변수가 클수록 PD 증가), 내리면 -1'로 변수 값 방향에 매핑한다.
범주형(NAME_EDUCATION_TYPE, OCCUPATION_TYPE 등)은 스코어카드 범주 묶음의 bin 번호(부도율 낮은 순)를 서수로 넣고 +1을 건다. 결측은 NaN.
연령·성별 제외, scale_pos_weight 없음.
실행: PYTHONPATH=<optuna 설치 폴더> python -m src.p2_mono --mode mono|free [--trials 20]
  mono: 단조 제약 GBM, free: 같은 입력·같은 탐색·같은 시드의 제약 없는 GBM(22변수, 튜닝)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .metrics import credit_metrics
from .p2_scorecard import MAX_BINS_CAT, MIN_SHARE_CAT
from .p2_woe import P2, Data, assign_categorical, categorical_bins
from .train import XGB_PARAMS

TUNE_LR, TUNE_TREES, TUNE_ES = 0.1, 1000, 50     # 탐색은 큰 학습률 + 조기 종료
FINAL_LR = 0.03                                   # 최종 재학습은 낮은 학습률 (v2와 같은 0.03), 조기 종료는 valid AUC 150


def log(msg: str, decision: bool = False):
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def build(variables):
    tr, va = Data("train"), Data("valid")
    bins = pd.read_csv(P2 / "woe_bins.csv", encoding="utf-8-sig")
    Xtr, Xva, cons, rows = [], [], [], []
    for v in variables:
        g = bins[(bins.variable == v) & (bins.bin_id >= 0)].sort_values("bin_id")
        rate = g.bad_rate_train.to_numpy()
        up, down = bool(np.all(np.diff(rate) >= -1e-9)), bool(np.all(np.diff(rate) <= 1e-9))
        assert up or down, f"{v}: 구간 부도율이 단조가 아님"
        x_tr, x_va = tr.var(v), va.var(v)
        if tr.is_cat[v]:
            mp = categorical_bins(x_tr, tr.y, min_share=MIN_SHARE_CAT, max_bins=MAX_BINS_CAT)
            f_tr, f_va = assign_categorical(x_tr, mp).astype(float), assign_categorical(x_va, mp).astype(float)
            f_tr[f_tr < 0], f_va[f_va < 0] = np.nan, np.nan
            assert up and not down or up, f"{v}: bin 번호가 부도율 오름차순이 아님"
            sign, kind = 1, "범주형(bin 서수)"
        else:
            f_tr, f_va = x_tr, x_va
            sign, kind = (1 if up and rate[-1] > rate[0] else -1), "수치형"
        Xtr.append(f_tr), Xva.append(f_va), cons.append(sign)
        rows.append({"variable": v, "type": kind, "constraint": sign, "bad_rate_first_bin": rate[0], "bad_rate_last_bin": rate[-1], "n_nonmissing_bins": len(rate),
                     "meaning": "값이 클수록 PD 증가" if sign > 0 else "값이 클수록 PD 감소"})
    return tr, va, np.column_stack(Xtr).astype(np.float32), np.column_stack(Xva).astype(np.float32), tuple(cons), pd.DataFrame(rows)


def main(n_trials: int = 20, mode: str = "mono"):
    tag = "mono" if mode == "mono" else "free22"
    label = "단조 제약 GBM" if mode == "mono" else "제약 없는 GBM(22, 튜닝)"
    sc = json.loads((P2 / "scorecard_model.json").read_text(encoding="utf-8"))
    variables = sc["variables"]
    assert all(c < 0 for c in sc["coef"].values()), "스코어카드 계수는 전부 음수여야 한다"
    tr, va, Xtr, Xva, cons_all, ctab = build(variables)
    ctab.to_csv(P2 / "mono_constraints.csv", index=False, encoding="utf-8-sig")
    cons = cons_all if mode == "mono" else None
    log(f"4b[{tag}] 시작: {label}, 변수 {len(variables)}개(스코어카드와 동일), 입력 열 {Xtr.shape[1]}, train {len(tr.y):,} / valid {len(va.y):,} (test 미사용)")
    if mode == "mono":
      log(f"4b 제약 방향 규칙(학습 전 고정): 스코어카드 구간화에서 결측을 뺀 구간의 train 부도율 추세가 오르면 +1, 내리면 -1. 스코어카드 계수 22개 전부 음수라 WoE 방향과 일치. "
        f"범주형은 스코어카드 범주 묶음 번호(부도율 낮은 순)를 서수로 넣고 +1. 결과 +1 {sum(c > 0 for c in cons_all)}개, -1 {sum(c < 0 for c in cons_all)}개 (mono_constraints.csv)", decision=True)
    else:
      log("4b[free22] 제약 없는 GBM(22): 단조 GBM과 입력 행렬(22열, 범주형은 bin 서수)·fold·탐색 범위·시드가 완전히 같고 monotone_constraints만 없다. 단조 제약 효과만 분리하기 위함", decision=True)
    y = tr.y
    skf = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xtr, y))
    log(f"4b[{tag}] 튜닝 설정(사용자 지시로 40→20 trial, 탐색 lr 0.1 + 조기 종료): Optuna TPE sampler seed {C.SEED}, fold 분할 StratifiedKFold(5, shuffle, seed {C.SEED}), XGBoost seed {C.SEED}, {n_trials} trial, 목적 = train 5-fold CV 평균 AUC (valid 미사용), 탐색 중 lr {TUNE_LR}, 최대 {TUNE_TREES}그루, fold 내 조기 종료 {TUNE_ES}(fold 검증 AUC). "
        f"탐색: max_depth 2~6, min_child_weight 5~100(log), subsample 0.6~1, colsample_bytree 0.5~1, reg_lambda 0.1~30(log), reg_alpha 1e-3~5(log), gamma 0~2. "
        f"trial 0은 v2 설정(depth 5, mcw 20, subsample 0.8, colsample 0.7, lambda 5)을 넣어 기준선으로 쓴다", decision=True)
    trials_log = []

    def objective(trial):
        p = dict(max_depth=trial.suggest_int("max_depth", 2, 6), min_child_weight=trial.suggest_float("min_child_weight", 5, 100, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True), gamma=trial.suggest_float("gamma", 0, 2))
        t0, aucs, iters = time.time(), [], []
        for a, b in skf:
            m = xgb.XGBClassifier(n_estimators=TUNE_TREES, learning_rate=TUNE_LR, tree_method="hist", eval_metric="auc", early_stopping_rounds=TUNE_ES, random_state=C.SEED,
                                  n_jobs=C.N_JOBS, monotone_constraints=cons, **p)
            m.fit(Xtr[a], y[a], eval_set=[(Xtr[b], y[b])], verbose=0)
            aucs.append(roc_auc_score(y[b], m.predict_proba(Xtr[b], iteration_range=(0, m.best_iteration + 1))[:, 1]))
            iters.append(m.best_iteration)
        trial.set_user_attr("iters", iters)
        row = {"trial": trial.number, **p, "cv_auc": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs)), "mean_best_iter": float(np.mean(iters)), "sec": round(time.time() - t0, 1)}
        trials_log.append(row)
        pd.DataFrame(trials_log).to_csv(P2 / f"{tag}_optuna_trials.csv", index=False, encoding="utf-8-sig")
        log(f"4b[{tag}] trial {trial.number}: CV AUC {row['cv_auc']:.5f} (sd {row['cv_auc_sd']:.5f}), depth {p['max_depth']}, mcw {p['min_child_weight']:.1f}, 평균 트리 {row['mean_best_iter']:.0f}, {row['sec']}초")
        return row["cv_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.enqueue_trial(dict(max_depth=5, min_child_weight=20.0, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.001, gamma=0.0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    log(f"4b[{tag}] 튜닝 끝: 최고 CV AUC {study.best_value:.5f} (trial {study.best_trial.number}, v2 설정 trial 0은 {trials_log[0]['cv_auc']:.5f}), 최적 {json.dumps(best)}", decision=True)

    # ---- 최종: 다른 GBM과 같은 방식 (lr 0.03, 최대 3000그루, valid AUC 조기 종료 150)
    prm = {k: v for k, v in XGB_PARAMS.items() if k not in ("max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda")}
    log(f"4b[{tag}] 최종 재학습: 찾은 파라미터 + lr {FINAL_LR}(범위 0.03~0.05 중 v2와 같은 값), 최대 3000그루, valid AUC 조기 종료 150, scale_pos_weight 없음", decision=True)
    final = xgb.XGBClassifier(scale_pos_weight=1.0, monotone_constraints=cons, **prm, **best)
    t0 = time.time()
    final.fit(Xtr, y, eval_set=[(Xva, va.y)], verbose=0)
    p = final.predict_proba(Xva, iteration_range=(0, final.best_iteration + 1))[:, 1]
    m = credit_metrics(va.y, p)
    (P2 / "raw").mkdir(exist_ok=True)
    np.save(P2 / "raw" / f"valid_p_{tag}.npy", p)
    # 단조 검증: 변수 하나씩 분위수 격자로 올리며 평균 예측이 제약 방향으로만 움직이는지 (valid 500행 부분집합)
    rng = np.random.default_rng(C.SEED)
    sub = Xva[rng.choice(len(Xva), 500, replace=False)]
    viol = []
    for j, v in enumerate(variables if mode == "mono" else []):
        col = Xva[:, j]
        grid = np.unique(np.nanpercentile(col, np.linspace(1, 99, 15)))
        prev, bad = None, 0
        for gval in grid:
            z = sub.copy()
            z[:, j] = gval
            mp = final.predict_proba(z, iteration_range=(0, final.best_iteration + 1))[:, 1]
            if prev is not None:
                bad += int(np.any((mp - prev) * cons[j] < -1e-7))
            prev = mp
        viol.append(bad)
    (P2 / f"gbm_{tag}.json").write_text(json.dumps({"variables": variables, "constraints": dict(zip(variables, cons_all)) if mode == "mono" else None, "final_lr": FINAL_LR, "tune_lr": TUNE_LR, "n_trials": n_trials, "seed": C.SEED, "best_params": best, "cv_auc": study.best_value, "best_iteration": int(final.best_iteration),
                                                  "final_fit_minutes": round((time.time() - t0) / 60, 1), "valid_metrics": m, "monotone_violations_in_check": int(sum(viol))}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"4b[{tag}] 최종: best_iter {final.best_iteration}, valid AUC {m['AUC']:.4f}, KS {m['KS']:.4f}, ECE {m['ECE_10q']:.4f}, Brier {m['Brier']:.4f}. 단조 위반 점검(mono만; 변수 22개 × 격자 15점 × 500행): 위반 {sum(viol) if mode == 'mono' else '해당 없음'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--mode", choices=["mono", "free"], default="mono")
    a = ap.parse_args()
    main(a.trials, a.mode)
