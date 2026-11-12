"""Phase 2-3: shortlist 변수로 GBM을 v2와 같은 설정으로 학습 (train 적합, valid 조기 종료/평가, test 미사용).

v2 설정: src.train.XGB_PARAMS (n_estimators 3000, lr 0.03, depth 5, min_child_weight 20, subsample 0.8, colsample 0.7, reg_lambda 5,
early_stopping_rounds 150 on valid AUC, seed 42, n_jobs 4), scale_pos_weight 없음, 연령·성별 제외.
실행: python -m src.p2_gbm                  (shortlist 전체)
      python -m src.p2_gbm --from-scorecard (스코어카드 최종 변수만; 변수 축소와 모형 형태 비용 분리용 보조 실험)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import xgboost as xgb

from . import config as C
from .metrics import credit_metrics
from .p2_woe import P2, Data
from .train import XGB_PARAMS


def log(msg: str):
    line = f"- `{datetime.now():%H:%M:%S}` {msg}"
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def run(variables: list[str], name: str):
    tr, va = Data("train"), Data("valid")
    idx = [i for v in variables for i in tr.cols[v]]
    Xtr, Xva = np.asarray(tr.X[:, idx], dtype=np.float32), np.asarray(va.X[:, idx], dtype=np.float32)
    model = xgb.XGBClassifier(scale_pos_weight=1.0, **XGB_PARAMS)
    t0 = time.time()
    model.fit(Xtr, tr.y, eval_set=[(Xva, va.y)], verbose=0)
    mins = (time.time() - t0) / 60
    p = model.predict_proba(Xva)[:, 1]
    m = credit_metrics(va.y, p)
    (P2 / "raw").mkdir(exist_ok=True)
    np.save(P2 / "raw" / f"valid_p_{name}.npy", p)
    (P2 / f"gbm_{name}.json").write_text(json.dumps({"variables": variables, "n_input_cols": len(idx), "best_iteration": int(model.best_iteration), "minutes": round(mins, 1),
                                                     "valid_metrics": m, "params": {k: v for k, v in XGB_PARAMS.items()}}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"2-3 GBM({name}): 변수 {len(variables)}개 (입력 열 {len(idx)}), best_iter {model.best_iteration}, {mins:.1f}분, valid AUC {m['AUC']:.4f}, KS {m['KS']:.4f}, ECE {m['ECE_10q']:.4f}, Brier {m['Brier']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-scorecard", action="store_true")
    a = ap.parse_args()
    if a.from_scorecard:
        vars_ = json.loads((P2 / "scorecard_model.json").read_text(encoding="utf-8"))["variables"]
        run(vars_, "scorecard_vars")
    else:
        vars_ = json.loads((P2 / "shortlist_vars.json").read_text(encoding="utf-8"))["final"]
        run(vars_, "shortlist")
