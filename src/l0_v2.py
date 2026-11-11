"""v2 동결 준비: 190개 모형 시간 점검 → gain 중요도 곡선(K-sweep)으로 변수 수 결정 → (확정 시) 보정·로지스틱·test 1회·동결.

실행: python -m src.l0_v2 --phase timing | sweep | final
- timing, sweep 단계는 test 데이터를 모형에 넣지 않는다 (test 예측·지표 계산 없음).
- final 단계에서 test를 처음으로 최종 1회 평가하고, 그 뒤 test로 아무 결정도 하지 않는다.
push 안 함, Co-Authored-By 없음, 이전 결과 폴더(v0_baseline, v1_*, comparison_2x2, l0의 기존 파일) 수정 안 함.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import l0_features as F
from . import l0_tier2 as T
from .data import clean, load_raw, split_labels
from .l0_family import family_of
from .l0_run import BASE_DIR, commit, git, log
from .l0_tier2_run import Design
from .metrics import credit_metrics
from .models import Calibrator
from .train import XGB_PARAMS

L0 = C.RESULTS_DIR / "l0"
W = C.RESULTS_DIR / "l0_v2"          # 정리 단계의 작업·결과 폴더 (새 폴더)
WRAW = W / "raw"                      # 큰 중간 산출물 (git 제외)
DUP_CORR = 0.95
DROP_LIMIT = 0.002                    # K 선택: 최고 valid AUC와의 허용 차이


def keep_awake():
    """학습 도중 PC가 절전으로 들어가 시간이 늘어나는 일(Tier2 10단계 335분)을 막는다."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        log("절전 방지 설정 (SetThreadExecutionState)")
    except Exception as e:
        log(f"절전 방지 설정 실패: {e}")


# ---------------------------------------------------------------- 상태 불러오기
def load_state():
    df = clean(load_raw())
    app_ids = pd.Index(df[C.ID_COL])
    split = split_labels(df)
    saved = pd.read_csv(BASE_DIR / "raw" / "splits.csv").set_index(C.ID_COL)["split"]
    assert (saved.reindex(app_ids).to_numpy() == split.to_numpy()).all(), "기존 분할과 다르다"
    cfg = json.loads((BASE_DIR / "modeling" / "run_config.json").read_text(encoding="utf-8"))
    base_feats = cfg["features"]
    assert not {"DAYS_BIRTH", "EMPLOYED_AGE_RATIO", "CODE_GENDER"} & set(base_feats)

    t1 = pd.concat([pd.read_parquet(L0 / f"features_stage{s}_{F.STAGE_NAMES[s]}.parquet") for s in (1, 2, 3, 4)], axis=1).reindex(app_ids).astype(np.float32)
    ivt = pd.read_csv(L0 / "iv_table.csv", encoding="utf-8-sig").set_index("variable")
    meta, names, arrays = {}, [], [t1.to_numpy()]
    for c in t1.columns:
        stage = next(f.stage for f in F.REGISTRY if f.name == c)
        src = next(f.source for f in F.REGISTRY if f.name == c)
        meta[c] = {"table": f"tier1:{src}", "stage": stage, "slice": "", "measure": c, "stat": "", "IV": float(ivt.loc[c, "IV"]),
                   "valid_delta": np.nan, "family": family_of(c)}
        names.append(c)
    for t in T.TABLE_ORDER:
        ev = pd.read_csv(L0 / f"candidates_eval_{t}.csv", encoding="utf-8-sig")
        ev = ev[ev["selected"]]
        fr = pd.read_parquet(L0 / f"cand_{t}.parquet", columns=ev["variable"].tolist()).reindex(app_ids)
        arrays.append(fr.to_numpy(np.float32))
        for r in ev.itertuples():
            meta[r.variable] = {"table": t, "stage": T.SPECS[t].stage, "slice": r.slice, "measure": r.measure, "stat": r.stat, "IV": float(r.IV),
                                "valid_delta": float(r.incr_AUC_valid_delta), "family": family_of(r.variable)}
            names.append(r.variable)
        del fr
    E = np.hstack(arrays)
    assert E.shape[1] == len(names) == 139, (E.shape, len(names))
    design = Design(df, base_feats, split)
    return df, app_ids, split, base_feats, names, meta, E, design


def fit_valid(design: Design, E: np.ndarray, cols):
    """train으로 적합, valid로 조기 종료. test는 쓰지 않는다."""
    Xtr, Xva = design.X("train", E, cols), design.X("valid", E, cols)
    model = xgb.XGBClassifier(scale_pos_weight=1.0, **XGB_PARAMS)
    t0 = time.time()
    model.fit(Xtr, design.y["train"], eval_set=[(Xva, design.y["valid"])], verbose=0)
    mins = (time.time() - t0) / 60
    p = model.predict_proba(Xva)[:, 1]
    n_cols = Xtr.shape[1]
    del Xtr, Xva
    gc.collect()
    return model, p, mins, n_cols


def calib_rows(y, p) -> dict:
    """valid 위의 보정 전후 비교. 'Platt 자체 적합(in-sample)'과 'Platt 교차적합(5-fold, 접힌 부분만 평가)' 둘 다."""
    raw = credit_metrics(y, p)
    cal = Calibrator("platt").fit(p, y)
    ins = credit_metrics(y, cal.predict(p))
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(p, y):
        oof[b] = Calibrator("platt").fit(p[a], y[a]).predict(p[b])
    cv = credit_metrics(y, oof)
    return {"valid_AUC": raw["AUC"], "raw_ECE": raw["ECE_10q"], "raw_Brier": raw["Brier"],
            "platt_insample_ECE": ins["ECE_10q"], "platt_insample_Brier": ins["Brier"],
            "platt_crossfit_ECE": cv["ECE_10q"], "platt_crossfit_Brier": cv["Brier"]}


def rank_matrix(cols: dict[str, np.ndarray]):
    names = list(cols)
    R = np.zeros((len(next(iter(cols.values()))), len(names)), np.float32)
    for j, nm in enumerate(names):
        v = cols[nm].astype(np.float64)
        v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0.0, v)
        r = pd.Series(v).rank().to_numpy(np.float32)
        sd = r.std()
        R[:, j] = (r - r.mean()) / sd if sd > 0 else 0.0
    return R, {nm: i for i, nm in enumerate(names)}


# ---------------------------------------------------------------- phases
def phase_timing(state):
    df, app_ids, split, base_feats, names, meta, E, design = state
    log(f"190개 모형(신청서 {len(base_feats)} + L0 {len(names)}) 정상 속도 점검: 학습 1회, valid만 평가 (test 미사용)")
    model, p, mins, ncols = fit_valid(design, E, slice(0, E.shape[1]))
    WRAW.mkdir(parents=True, exist_ok=True)
    model.save_model(str(WRAW / "model_190.json"))
    (W / "features_190.json").write_text(json.dumps({"base_feats": base_feats, "l0": names}, ensure_ascii=False), encoding="utf-8")
    np.save(WRAW / "valid_p_190.npy", p)
    row = {"model": "190개(Tier2 10단계)", "n_inputs": ncols, "best_iter": int(model.best_iteration), "minutes": round(mins, 1), **calib_rows(design.y["valid"], p)}
    pd.DataFrame([row]).to_csv(W / "timing_190.csv", index=False, encoding="utf-8-sig")
    log(f"학습 1회 {mins:.1f}분 (best_iter {model.best_iteration}, 입력 열 {ncols}), valid AUC {row['valid_AUC']:.4f} "
        f"(Tier2 10단계 기록 valid 0.7908과 비교). 모형은 {WRAW / 'model_190.json'}에 저장")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["timing", "sweep", "final"])
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args(argv)
    keep_awake()
    with open(L0 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n\n# L0 v2 정리·동결 ({args.phase}, {datetime.now():%Y-%m-%d %H:%M})\n\n")
    W.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if args.phase == "timing":
        phase_timing(state)
    elif args.phase == "sweep":
        from .l0_v2_sweep import phase_sweep
        phase_sweep(state)
    else:
        from .l0_v2_final import phase_final
        phase_final(state, commit_enabled=not args.no_commit)


if __name__ == "__main__":
    main()
