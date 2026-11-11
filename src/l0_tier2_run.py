"""L0 Tier 2 실행기: 테이블 하나씩 타깃 EDA → 후보 생성 → (테이블 안) 상관 중복 제거 → 증분 AUC → 변수 선택 → 누적 모형 ablation.

실행: python -m src.l0_tier2_run --tables installments_payments [--iv-top K] [--reuse] [--no-commit]
테이블마다 결과를 저장하고 로컬 커밋한다 (push 안 함, Co-Authored-By 없음, v0_baseline/v1_*/comparison_2x2 수정 안 함).
이어서 돌릴 때는 앞선 테이블의 선택 결과(candidates_eval_<table>.csv)와 test_preds_stage<N>.csv를 읽는다.

라벨 사용 규칙:
- EDA, IV, 단독 AUC, 상관 중복 제거, 증분 AUC, 변수 선택은 train/valid만 쓴다 (valid는 증분 AUC의 평가에만).
- test는 선택이 끝난 변수 묶음의 성능을 ablation.csv에 보고하는 데만 쓴다. test 결과를 보고 변수를 다시 고르지 않는다.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
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
from .l0_run import BASE_AUC_EXPECTED, BASE_DIR, LEAK_AUC, ablation_row, commit, git, iv_band, iv_for, log, univariate_auc
from .metrics import credit_metrics, paired_auc_diffs
from .models import Calibrator, make_preprocessor
from .train import XGB_PARAMS

L0 = T.L0
TIME_LIMIT_H = 4.0
DUP_CORR = 0.95
IV_MIN = 0.02
SELECTION_RULE = ("선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND "
                  "train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.")


# ---------------------------------------------------------------- 모델 입력 조립 (float32)
class Design:
    """신청서 변수(전처리: train에서만 적합) + 숫자 추가 변수 행렬을 조립한다."""

    def __init__(self, df: pd.DataFrame, base_feats: list[str], split: pd.Series):
        self.df, self.base_feats = df, base_feats
        self.m = {s: (split == s).to_numpy() for s in ("train", "valid", "test")}
        self.pre = make_preprocessor(df[self.m["train"]], base_feats, "xgb")
        self.Xb = {"train": self.pre.fit_transform(df.loc[self.m["train"], base_feats]).to_numpy(np.float32)}
        for s in ("valid", "test"):
            self.Xb[s] = self.pre.transform(df.loc[self.m[s], base_feats]).to_numpy(np.float32)
        self.y = {s: df.loc[self.m[s], C.TARGET_COL].to_numpy() for s in self.m}

    def X(self, split: str, E: np.ndarray | None, cols=None):
        xb = self.Xb[split]
        if E is None or cols is None:
            return xb
        k = (cols.stop - cols.start) if isinstance(cols, slice) else len(cols)
        rows = np.flatnonzero(self.m[split])
        out = np.empty((len(rows), xb.shape[1] + k), np.float32)
        out[:, :xb.shape[1]] = xb
        for r0 in range(0, len(rows), 30_000):
            r = rows[r0:r0 + 30_000]
            out[r0:r0 + len(r), xb.shape[1]:] = E[r][:, cols] if isinstance(cols, slice) else E[np.ix_(r, cols)]
        return out


def fit_big(design: Design, E: np.ndarray | None, cols=None) -> dict:
    X_tr, X_va, X_te = (design.X(s, E, cols) for s in ("train", "valid", "test"))
    model = xgb.XGBClassifier(scale_pos_weight=1.0, **XGB_PARAMS)
    model.fit(X_tr, design.y["train"], eval_set=[(X_va, design.y["valid"])], verbose=0)
    p_va, p_te = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
    cal = Calibrator("platt").fit(p_va, design.y["valid"])
    ids = design.df.loc[design.m["test"], C.ID_COL].to_numpy()
    out = {"ids": ids, "y": design.y["test"], "p_raw": p_te, "p_cal": cal.predict(p_te), "best_iter": int(model.best_iteration),
           "n_cols": X_tr.shape[1], "valid_auc": float(roc_auc_score(design.y["valid"], p_va))}
    del X_tr, X_va, X_te, model
    gc.collect()
    return out


# ---------------------------------------------------------------- 증분 AUC (train 3-fold + valid)
_BASE_PARAMS = dict(XGB_PARAMS, n_estimators=400, learning_rate=0.06, early_stopping_rounds=None)


def base_logits(design: Design, E: np.ndarray, cols, folds) -> tuple[np.ndarray, np.ndarray]:
    """기준 모델(신청서 + Tier1 + 앞서 선택된 변수)의 train OOF 로짓과, train 전체로 적합한 모형의 valid 로짓."""
    Xtr, Xva, y = design.X("train", E, cols), design.X("valid", E, cols), design.y["train"]
    oof = np.zeros(len(y))
    for a, b in folds:
        mdl = xgb.XGBClassifier(**_BASE_PARAMS)
        mdl.fit(Xtr[a], y[a], verbose=0)
        oof[b] = mdl.predict(Xtr[b], output_margin=True)
    full = xgb.XGBClassifier(**_BASE_PARAMS)
    full.fit(Xtr, y, verbose=0)
    return oof, full.predict(Xva, output_margin=True)


def _small():
    return xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, min_child_weight=20, tree_method="hist",
                             n_jobs=C.N_JOBS, random_state=C.SEED)


def _cv_auc(Z, y, folds) -> float:
    p = np.zeros(len(y))
    for a, b in folds:
        m = _small().fit(Z[a], y[a], verbose=0)
        p[b] = m.predict_proba(Z[b])[:, 1]
    return float(roc_auc_score(y, p))


def _valid_auc(Ztr, ytr, Zva, yva) -> float:
    m = _small().fit(Ztr, ytr, verbose=0)
    return float(roc_auc_score(yva, m.predict_proba(Zva)[:, 1]))


def _rank_matrix(cols: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = list(cols)
    n = len(next(iter(cols.values())))
    R = np.zeros((n, len(names)), np.float32)
    for j, nm in enumerate(names):
        v = cols[nm].astype(np.float64)
        v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0.0, v)
        r = pd.Series(v).rank().to_numpy(np.float32)
        sd = r.std()
        R[:, j] = (r - r.mean()) / sd if sd > 0 else 0.0
    return R, names


def select_table(design: Design, E_base: np.ndarray, table: str, Fr: np.ndarray, names: list[str], reg: pd.DataFrame,
                 iv_top: int | None) -> pd.DataFrame:
    """train/valid만 사용. 1) IV·단독 AUC 2) 테이블 안 상관 중복 제거 3) 증분 AUC 4) 선택."""
    t0 = time.time()
    tr_idx, va_idx = np.flatnonzero(design.m["train"]), np.flatnonzero(design.m["valid"])
    ytr, yva = design.y["train"], design.y["valid"]
    rows = []
    for j, nm in enumerate(names):
        x = Fr[tr_idx, j].astype(np.float64)
        iv, nb = iv_for(pd.Series(x), ytr)
        med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
        xf = np.where(np.isnan(x), med, x)
        a = roc_auc_score(ytr, xf) if np.nanstd(xf) > 0 else 0.5
        rows.append({"variable": nm, "table": table, "train_missing_rate": float(np.isnan(x).mean()), "n_bins": nb, "IV": iv,
                     "IV_band": iv_band(iv), "single_AUC_train": max(a, 1 - a)})
    ev = pd.DataFrame(rows)
    ev["removed_reason"] = ""
    # (선택) IV 상위 K만 남기기: 사용자가 승인한 경우에만
    if iv_top:
        out_rank = ev["IV"].rank(ascending=False, method="first") > iv_top
        ev.loc[out_rank, "removed_reason"] = f"IV 순위 밖(상위 {iv_top}개만)"
    # 테이블 안 상관 0.95 초과 중복: IV 높은 쪽을 남긴다 (train 10만 행 표본 순위상관)
    rng = np.random.default_rng(C.SEED)
    samp = np.sort(rng.choice(tr_idx, size=min(100_000, len(tr_idx)), replace=False))
    alive = ev.loc[ev["removed_reason"] == "", "variable"].tolist()
    R, rn = _rank_matrix({nm: Fr[samp, names.index(nm)] for nm in alive})
    corr = (R.T @ R) / len(samp)
    ix = {nm: i for i, nm in enumerate(rn)}
    kept, dup_of, dup_r = [], {}, {}
    for nm in ev[ev["variable"].isin(alive)].sort_values("IV", ascending=False)["variable"]:
        hit = [k for k in kept if abs(corr[ix[nm], ix[k]]) > DUP_CORR]
        if hit:
            best = max(hit, key=lambda k: abs(corr[ix[nm], ix[k]]))
            dup_of[nm], dup_r[nm] = best, float(corr[ix[nm], ix[best]])
        else:
            kept.append(nm)
    ev["corr_dup_of"] = ev["variable"].map(dup_of)
    ev["corr_dup_r"] = ev["variable"].map(dup_r)
    ev.loc[ev["corr_dup_of"].notna(), "removed_reason"] = "테이블 안 상관>0.95 중복"
    log(f"{table}: 후보 {len(ev)}개 → 상관 중복 제거 {len(dup_of)}개 → 증분 AUC 대상 {len(kept)}개 (IV는 train)")
    # 증분 AUC
    folds = list(StratifiedKFold(3, shuffle=True, random_state=C.SEED).split(np.zeros(len(ytr)), ytr))
    oof, vlog = base_logits(design, E_base, slice(0, E_base.shape[1]), folds)
    ref_cv = _cv_auc(oof[:, None], ytr, folds)
    ref_va = _valid_auc(oof[:, None], ytr, vlog[:, None], yva)
    log(f"{table}: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 {design.Xb['train'].shape[1] + E_base.shape[1]}열) train 3-fold {ref_cv:.4f}, valid {ref_va:.4f}")
    res = {}
    for nm in kept:
        j = names.index(nm)
        xtr, xva = Fr[tr_idx, j].astype(np.float64), Fr[va_idx, j].astype(np.float64)
        try:
            cv = _cv_auc(np.column_stack([oof, xtr]), ytr, folds)
            va = _valid_auc(np.column_stack([oof, xtr]), ytr, np.column_stack([vlog, xva]), yva)
        except Exception as e:
            cv = va = np.nan
            log(f"증분 AUC 계산 실패 {nm}: {type(e).__name__}")
        res[nm] = (cv - ref_cv, va - ref_va)
    ev["incr_AUC_trainCV_delta"] = ev["variable"].map(lambda v: res.get(v, (np.nan, np.nan))[0])
    ev["incr_AUC_valid_delta"] = ev["variable"].map(lambda v: res.get(v, (np.nan, np.nan))[1])
    ev["incr_AUC_trainCV_base"], ev["incr_AUC_valid_base"] = ref_cv, ref_va
    ok = (ev["removed_reason"] == "") & (ev["IV"] >= IV_MIN) & (ev["incr_AUC_trainCV_delta"] > 0) & (ev["incr_AUC_valid_delta"] > 0)
    ev["selected"] = ok
    ev["not_selected_reason"] = np.select(
        [ev["removed_reason"] != "", ev["IV"] < IV_MIN, ~(ev["incr_AUC_trainCV_delta"] > 0), ~(ev["incr_AUC_valid_delta"] > 0)],
        [ev["removed_reason"], "IV<0.02", "train 증분 AUC<=0", "valid 증분 AUC<=0"], default="")
    ev["flag_IV_lt_0.02"] = ev["IV"] < IV_MIN
    ev["suspect_leak_single_AUC_gt_0.75"] = ev["single_AUC_train"] > 0.75
    ev = ev.merge(reg[["variable", "slice", "measure", "stat", "korean_name"]], on="variable", how="left")
    log(f"{table}: 선택 {int(ev['selected'].sum())}개 / 후보 {len(ev)}개 ({(time.time() - t0) / 60:.1f}분). {SELECTION_RULE}")
    return ev.sort_values("IV", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- 산출물
def rebuild_dict():
    """variable_dict_l0.csv: Tier1 행 유지 + 처리된 테이블의 모든 후보 (선택=사용, 나머지=제외 + 사유)."""
    d_old = pd.read_csv(C.DOCS_DIR / "variable_dict_l0.csv", encoding="utf-8-sig")
    d_old = d_old[d_old["stage"] < 6]
    cols = list(d_old.columns)
    parts = []
    for t in T.TABLE_ORDER:
        rp, ep = L0 / f"cand_registry_{t}.csv", L0 / f"candidates_eval_{t}.csv"
        if not rp.exists():
            continue
        reg = pd.read_csv(rp, encoding="utf-8-sig")
        if ep.exists():
            ev = pd.read_csv(ep, encoding="utf-8-sig").set_index("variable")
            sel = reg["variable"].map(ev["selected"]).fillna(False).astype(bool)
            why = reg["variable"].map(ev["not_selected_reason"]).fillna("")
            reg["use"] = np.where(sel, "사용", "제외")
            reg["note"] = np.where(sel, "L0 Tier2 선택 (" + t + ")", "L0 Tier2 후보, 미선택: " + why.astype(str) + " (" + t + ")")
        else:
            reg["use"], reg["note"] = "사용", "L0 Tier2 후보 (" + t + ", 선택 전)"
        parts.append(reg[cols])
    from .l0_family import add_family
    d = add_family(pd.concat([d_old[cols]] + parts, ignore_index=True))
    d.to_csv(C.DOCS_DIR / "variable_dict_l0.csv", index=False, encoding="utf-8-sig")


def write_rules():
    rows = []
    for t in T.TABLE_ORDER:
        p = L0 / f"summary_{t}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    L = ["# L0 Tier 2 후보 생성·선택 규칙", "",
         "Tier 1은 사전에 정한 목록을 구현한 것이었다. Tier 2는 *train 라벨로 본 타깃 EDA*(`target_eda_<테이블>.md`)에서 신호가 확인된 축을 중심으로 후보를 대량 생성하고, train/valid로만 선택한다. test는 선택이 끝난 묶음의 성능 보고에만 쓴다.", "",
         "## 후보 생성 규칙", "",
         "후보 = **집계대상(측정값) x 통계 x 시간창 x 상태분리**. 한 행(대출·신청·납부회차·월)에서 측정값을 만들고, 슬라이스(시간창·상태·유형)로 거른 뒤 고객(SK_ID_CURR)별로 집계한다. 집계에는 라벨을 쓰지 않는다.",
         "",
         "- **시간창**: 최근 3/6/12/24개월 (일 단위 시간축은 30일=1개월). 월 단위 테이블은 MONTHS_BALANCE 기준.",
         "- **통계**: 평균·합계·최댓값·최솟값 중 측정값 성격에 맞는 것만. 슬라이스 종류별로 측정값 목록을 줄여 후보 폭증을 막았다.",
         "- **상태분리**: 계약 상태·상품 유형·거절 사유·연체/부족 납부 회차 등. 타깃 EDA에서 신호가 있는 구간만 유지 (부도율 배수가 1에서 10% 이상 벗어나고 구간 고객 500명 이상, 또는 구간 내 불량 이벤트 고객 100명 이상의 배수가 10% 이상 벗어남). 전체 기간과 시간창 4개는 항상 만든다.",
         "- **건수·결측**: 슬라이스마다 `*_CNT`(기록 건수, 없으면 0). 불량 플래그 합계는 테이블에 기록이 있을 때만 값, 나머지 통계는 해당 슬라이스에 기록이 없으면 결측.",
         "- 비율형 측정값은 분모가 0 이하이면 결측, 극단값은 상한으로 절단 (잔액/한도 비율 5, 승인/신청 5, 납부 비율 5, 카드 최소납부 대비 10).",
         "- 한국어 설명은 규칙으로 생성했다 (`docs/variable_dict_l0.csv`, 변수마다 한 문장).",
         "", "## 선택 절차 (테이블마다, train/valid만)", "",
         "1. train에서 IV·단독 AUC를 계산한다.",
         f"2. **테이블 안 상관 {DUP_CORR} 초과 중복을 먼저 제거**한다 (train 10만 행 표본의 순위 상관, 남기는 쪽은 IV가 높은 변수).",
         "3. 남은 후보에 대해 증분 AUC를 계산한다: 기준 모델(신청서+Tier1+앞서 선택된 변수)의 train 3-fold OOF 로짓 위에 후보 1개를 더한 작은 XGBoost의 AUC 차이. "
         "train 3-fold CV 값과, train으로 적합해 valid에서 잰 값 두 가지.",
         f"4. {SELECTION_RULE}",
         "5. 선택된 묶음을 누적해 전체 모형을 학습하고 test 성능을 `ablation.csv`에 보고한다. **test 결과를 보고 변수를 다시 고르지 않는다.**",
         "", "## 테이블별 후보 수", "",
         "| 단계 | 테이블 | 슬라이스(정의/유지) | 후보 | 상관 중복 제거 | 증분 AUC 대상 | 선택 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['stage']} | {r['table']} | {r['n_slices']}/{r['n_keep']} | {r['n_cand']} | {r.get('n_dup', '-')} | {r.get('n_eval', '-')} | {r.get('n_selected', '-')} |")
    (L0 / "candidates_rules.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def elapsed_h(t0):
    return (time.time() - t0) / 3600


def load_preds(path, te_ids):
    p = pd.read_csv(path)
    assert (p[C.ID_COL].to_numpy() == te_ids).all()
    return p["pd"].to_numpy()


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=",".join(T.TABLE_ORDER))
    ap.add_argument("--reuse", action="store_true", help="저장된 후보 parquet이 있으면 다시 집계하지 않는다")
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--iv-top", type=int, default=None, help="테이블마다 IV 상위 K개 후보만 평가 (승인 시에만)")
    ap.add_argument("--nrows", type=int, default=None, help="점검용")
    args = ap.parse_args(argv)
    do_commit = not args.no_commit
    t_all = time.time()
    tables = [t for t in args.tables.split(",") if t]

    with open(L0 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n\n# L0 Tier 2 실행 로그 ({datetime.now():%Y-%m-%d %H:%M})\n\n테이블: {', '.join(tables)}"
                f"{'  (IV 상위 ' + str(args.iv_top) + '개만)' if args.iv_top else ''}\n\n")
    log("push 안 함, Co-Authored-By 없음, v0_baseline/v1_*/comparison_2x2 결과 수정 안 함. 라벨 계산은 train/valid만, test는 선택이 끝난 묶음의 보고용.")
    log("순서: installments_payments(6) → credit_card_balance(7) → POS_CASH_balance(8) → bureau(9) → previous_application(10). "
        "bureau_balance는 Tier1(4단계)에 이미 있어 Tier2에서 제외한다. bureau/previous_application은 Tier1에 없던 시간창·상태분리 슬라이스를 추가하는 후보 생성이다.", decision=True)
    log(SELECTION_RULE, decision=True)

    df = clean(load_raw())
    app_ids = pd.Index(df[C.ID_COL])
    if not app_ids.is_unique:
        log("신청서 키 중복: 중단", decision=True)
        return
    split = split_labels(df)
    saved = pd.read_csv(BASE_DIR / "raw" / "splits.csv").set_index(C.ID_COL)["split"]
    assert (saved.reindex(app_ids).to_numpy() == split.to_numpy()).all(), "기존 분할과 다르다"
    y_all = df[C.TARGET_COL].to_numpy()
    train = (split == "train").to_numpy()
    cfg = json.loads((BASE_DIR / "modeling" / "run_config.json").read_text(encoding="utf-8"))
    base_feats = cfg["features"]
    assert not {"DAYS_BIRTH", "EMPLOYED_AGE_RATIO", "CODE_GENDER"} & set(base_feats)
    log(f"분할 확인 (train/valid/test = {train.sum():,}/{(split == 'valid').sum():,}/{(split == 'test').sum():,}), 기준 입력 {len(base_feats)}개")

    # Tier1 변수 + 앞서 선택된 테이블
    t1 = pd.concat([pd.read_parquet(L0 / f"features_stage{s}_{F.STAGE_NAMES[s]}.parquet") for s in (1, 2, 3, 4)], axis=1).reindex(app_ids).astype(np.float32)
    t1_names = list(t1.columns)
    sel_names, sel_arrays = [], []
    first = T.TABLE_ORDER.index(tables[0])
    prior_stage, prior_key = None, None
    for t in T.TABLE_ORDER[:first]:
        ep = L0 / f"candidates_eval_{t}.csv"
        if not ep.exists():
            continue
        ev0 = pd.read_csv(ep, encoding="utf-8-sig")
        sel = ev0.loc[ev0["selected"], "variable"].tolist()
        if sel:
            fr = pd.read_parquet(L0 / f"cand_{t}.parquet", columns=sel).reindex(app_ids)
            sel_names += sel
            sel_arrays.append(fr.to_numpy(np.float32))
        if (L0 / "raw" / f"test_preds_stage{T.SPECS[t].stage}.csv").exists():
            prior_stage = T.SPECS[t].stage
        log(f"앞선 테이블 {t}: 선택된 변수 {len(sel)}개를 불러옴")
    design = Design(df, base_feats, split)
    E_base = np.hstack([t1.to_numpy()] + sel_arrays)
    del t1
    gc.collect()

    groups = pd.read_csv(L0 / "raw" / "group_vars.csv").set_index(C.ID_COL)
    ab_path = L0 / "ablation.csv"
    abl = pd.read_csv(ab_path, encoding="utf-8-sig")
    base_p = pd.read_csv(BASE_DIR / "raw" / "test_predictions.csv")
    te_ids = df.loc[split == "test", C.ID_COL].to_numpy()
    assert (base_p[C.ID_COL].to_numpy() == te_ids).all()
    assert abs(credit_metrics(base_p["target"].to_numpy(), base_p["pd"].to_numpy())["AUC"] - BASE_AUC_EXPECTED) < 5e-4
    preds = {"s0": base_p["pd"].to_numpy(), "s4": load_preds(L0 / "raw" / "test_preds_stage4.csv", te_ids)}
    rows_new: list[dict] = []
    done_stages: list[str] = []

    def save_rows():
        keep = abl[~abl["stage"].astype(str).isin(done_stages)]
        pd.concat([keep, pd.DataFrame(rows_new)], ignore_index=True).to_csv(ab_path, index=False, encoding="utf-8-sig")

    # 4r: Tier1 입력을 이 단계의 조립 방식으로 재학습한 기준 (한 번만)
    p4r = L0 / "raw" / "test_preds_stage4r.csv"
    n_t1 = len(t1_names)
    if p4r.exists():
        preds["s4r"] = load_preds(p4r, te_ids)
    else:
        t0 = time.time()
        r4 = fit_big(design, np.hstack([E_base[:, :n_t1]]), slice(0, n_t1))
        preds["s4r"] = r4["p_cal"]
        pd.DataFrame({C.ID_COL: r4["ids"], "target": r4["y"], "pd": r4["p_cal"], "pd_raw": r4["p_raw"]}).to_csv(p4r, index=False)
        a4r, a4 = credit_metrics(r4["y"], r4["p_cal"])["AUC"], credit_metrics(r4["y"], preds["s4"])["AUC"]
        row = ablation_row("4r", "Tier1 재학습(조립 방식 점검)", len(base_feats) + n_t1, 0, r4, preds, "s4", groups,
                           {"minutes": round((time.time() - t0) / 60, 1), "note": "기존 4단계와 같은 입력, float32 배열로 재학습", "valid_AUC": r4["valid_auc"]})
        done_stages.append("4r")
        rows_new.append(row)
        save_rows()
        log(f"조립 방식 점검: Tier1 입력을 새 방식으로 재학습한 test AUC {a4r:.4f} (기존 4단계 {a4:.4f}, 차이 {a4r - a4:+.4f}). 6단계부터 이 4r과 짝 비교한다.", decision=True)
    prev_key = f"s{prior_stage}" if prior_stage else "s4r"
    if prior_stage:
        preds[prev_key] = load_preds(L0 / "raw" / f"test_preds_stage{prior_stage}.csv", te_ids)
    prev_feats = len(base_feats) + E_base.shape[1]
    halted = False

    for name in tables:
        spec = T.SPECS[name]
        st = spec.stage
        if elapsed_h(t_all) > TIME_LIMIT_H - 0.5:
            log(f"{name} 이후는 4시간 상한이라 건너뜀", decision=True)
            break
        pq = L0 / f"cand_{name}.parquet"
        t0 = time.time()
        # ---- 1) EDA + 후보 생성
        try:
            if args.reuse and pq.exists() and (L0 / f"cand_registry_{name}.csv").exists():
                feat = pd.read_parquet(pq)
                reg = pd.read_csv(L0 / f"cand_registry_{name}.csv", encoding="utf-8-sig")
                n_keep = reg["slice"].nunique()
                log(f"{name}: 저장된 후보 {feat.shape[1]}개를 읽음")
            else:
                for attempt in (1, 2):
                    try:
                        md, keep, jd = T.eda_pass(spec, app_ids, y_all, train, nrows=args.nrows)
                        break
                    except MemoryError:
                        spec.chunk_rows //= 2
                        gc.collect()
                        log(f"{name} EDA 메모리 부족, 청크를 {spec.chunk_rows:,}행으로 줄여 재시도 ({attempt}/2)")
                else:
                    raise MemoryError("EDA 2회 실패")
                (L0 / f"target_eda_{name}.md").write_text(md, encoding="utf-8")
                if len(jd):
                    jd.to_csv(L0 / f"target_eda_slices_{name}.csv", index=False, encoding="utf-8-sig")
                log(f"{name}: 타깃 EDA 저장 (슬라이스 {len(spec.slices)}개 중 유지 {len(keep)}개: {', '.join(keep)})")
                dropped = [s.name for s in spec.slices if s.name not in keep]
                if dropped:
                    log(f"{name}: EDA에서 신호가 약해 후보를 만들지 않은 슬라이스: {', '.join(dropped)}")
                for attempt in (1, 2):
                    try:
                        feat, reg = T.build_candidates(spec, app_ids, keep, nrows=args.nrows)
                        break
                    except MemoryError:
                        spec.chunk_rows //= 2
                        gc.collect()
                        log(f"{name} 후보 집계 메모리 부족, 청크를 {spec.chunk_rows:,}행으로 줄여 재시도 ({attempt}/2)")
                else:
                    raise MemoryError("후보 집계 2회 실패")
                if feat.index.has_duplicates or len(feat) != len(app_ids):
                    raise RuntimeError(f"{name}: 집계 키 중복 또는 행 수 불일치")
                feat.to_parquet(pq)
                reg.to_csv(L0 / f"cand_registry_{name}.csv", index=False, encoding="utf-8-sig")
                n_keep = len(keep)
        except RuntimeError as e:
            log(f"{name} **중단**: {e}", decision=True)
            return
        except Exception as e:
            log(f"{name} 예상 밖 오류로 건너뜀: {type(e).__name__}: {e}", decision=True)
            log("```\n" + traceback.format_exc()[-1500:] + "\n```")
            continue
        summ = {"stage": st, "table": name, "n_slices": len(spec.slices), "n_keep": n_keep, "n_cand": feat.shape[1]}
        (L0 / f"summary_{name}.json").write_text(json.dumps(summ), encoding="utf-8")
        log(f"{name}: 후보 {feat.shape[1]}개 생성, 키 {len(feat):,}개 (중복 없음), {(time.time() - t0) / 60:.1f}분")
        rebuild_dict()
        write_rules()
        commit([f"results/l0/target_eda_{name}.md", f"results/l0/target_eda_slices_{name}.csv", f"results/l0/cand_registry_{name}.csv",
                f"results/l0/summary_{name}.json", "results/l0/candidates_rules.md", "docs/variable_dict_l0.csv", "results/l0/RUNLOG.md"],
               f"L0 Tier2 {st}단계({name}): 타깃 EDA, 후보 {feat.shape[1]}개 생성", do_commit)

        # ---- 2) 상관 중복 제거 → 증분 AUC → 선택 (train/valid만)
        names = list(feat.columns)
        Fr = feat.to_numpy(np.float32)
        del feat
        gc.collect()
        try:
            ev = select_table(design, E_base, name, Fr, names, reg, args.iv_top)
        except Exception as e:
            log(f"{name} 선택 단계 오류로 건너뜀: {type(e).__name__}: {e}", decision=True)
            log("```\n" + traceback.format_exc()[-1500:] + "\n```")
            continue
        ev.to_csv(L0 / f"candidates_eval_{name}.csv", index=False, encoding="utf-8-sig")
        allp = [pd.read_csv(L0 / f"candidates_eval_{t}.csv", encoding="utf-8-sig") for t in T.TABLE_ORDER if (L0 / f"candidates_eval_{t}.csv").exists()]
        pd.concat(allp, ignore_index=True).to_csv(L0 / "candidates_eval.csv", index=False, encoding="utf-8-sig")
        summ.update({"n_dup": int((ev["removed_reason"] == "테이블 안 상관>0.95 중복").sum()), "n_eval": int(ev["incr_AUC_trainCV_delta"].notna().sum()),
                     "n_selected": int(ev["selected"].sum())})
        (L0 / f"summary_{name}.json").write_text(json.dumps(summ), encoding="utf-8")
        top = ev.head(5)
        log(f"{name}: IV 상위 5 " + ", ".join(f"{r.variable} {r.IV:.3f}" for r in top.itertuples()) +
            f". IV>=0.02 {int((ev.IV >= IV_MIN).sum())}개, 단독 train AUC 최대 {ev.single_AUC_train.max():.3f}")
        sus = ev[ev["suspect_leak_single_AUC_gt_0.75"]]
        if len(sus):
            log(f"단독 train AUC>0.75 후보 {len(sus)}개 (누수 점검 필요): {sus.variable.tolist()[:10]}", decision=True)
        rebuild_dict()
        write_rules()
        commit([f"results/l0/candidates_eval_{name}.csv", "results/l0/candidates_eval.csv", f"results/l0/summary_{name}.json",
                "results/l0/candidates_rules.md", "docs/variable_dict_l0.csv", "results/l0/RUNLOG.md"],
               f"L0 Tier2 {st}단계({name}): 상관 중복 제거·증분 AUC·변수 선택 {int(ev['selected'].sum())}개 (test 보기 전)", do_commit)
        selected = ev.loc[ev["selected"], "variable"].tolist()
        if not selected:
            log(f"{name}: 선택된 변수가 없어 ablation 단계를 만들지 않음", decision=True)
            continue

        # ---- 3) 선택된 묶음을 누적해 학습, test 성능은 보고만
        E_new = np.hstack([E_base, Fr[:, [names.index(v) for v in selected]]])
        del Fr
        gc.collect()
        known = set(pd.read_csv(C.VAR_DICT, encoding="utf-8-sig")["variable"]) | set(pd.read_csv(C.DOCS_DIR / "variable_dict_l0.csv", encoding="utf-8-sig")["variable"])
        miss = [f for f in base_feats + t1_names + sel_names + selected if f not in known]
        assert not miss, f"변수 사전에 없는 변수 {miss[:5]}"
        t0 = time.time()
        res = None
        for attempt in (1, 2):
            try:
                res = fit_big(design, E_new, slice(0, E_new.shape[1]))
                break
            except (MemoryError, xgb.core.XGBoostError) as e:
                log(f"{st}단계 학습 오류 (시도 {attempt}/2): {type(e).__name__}")
                gc.collect()
        if res is None:
            log(f"{st}단계({name}) 학습이 2번 실패해 건너뜀", decision=True)
            continue
        mins = (time.time() - t0) / 60
        key = f"s{st}"
        preds[key] = res["p_cal"]
        pd.DataFrame({C.ID_COL: res["ids"], "target": res["y"], "pd": res["p_cal"], "pd_raw": res["p_raw"]}).to_csv(L0 / "raw" / f"test_preds_stage{st}.csv", index=False)
        n_feat = len(base_feats) + E_new.shape[1]
        row = ablation_row(st, f"+{name}", n_feat, n_feat - prev_feats, res, preds, prev_key, groups,
                           {"minutes": round(mins, 1), "note": f"선택 {len(selected)}개 (후보 {len(ev)}개 중)", "valid_AUC": res["valid_auc"]})
        d1 = paired_auc_diffs(res["y"], preds, [("vs_tier1", key, "s4r")]).iloc[0]
        row.update({"dAUC_vs_tier1r": d1["diff_a_minus_b"], "dAUC_vs_tier1r_lo": d1["ci_low"], "dAUC_vs_tier1r_hi": d1["ci_high"]})
        auc = row["AUC"]
        done_stages.append(str(st))
        if auc > LEAK_AUC:
            row["note"] = "누수 의심으로 중단"
            rows_new.append(row)
            save_rows()
            uvdf = pd.concat([df[[C.ID_COL, C.TARGET_COL]].reset_index(drop=True), pd.DataFrame(E_new[:, E_base.shape[1]:], columns=selected)], axis=1)
            uv = univariate_auc(uvdf, split, selected)
            log(f"{st}단계 test AUC {auc:.4f} > {LEAK_AUC}: **누수 의심, 중단**", decision=True)
            log("새 변수 단독 test AUC 상위:\n\n```\n" + uv.head(10).round(4).to_string(index=False) + "\n```")
            commit(["results/l0/RUNLOG.md", "results/l0/ablation.csv"], f"L0 Tier2 {st}단계({name}) 누수 의심으로 중단, 로그만 기록", do_commit)
            halted = True
            break
        rows_new.append(row)
        save_rows()
        log(f"{st}단계(+{name}) 학습 {mins:.1f}분: 입력 {n_feat}개(+{n_feat - prev_feats}), best_iter {res['best_iter']}, valid AUC {res['valid_auc']:.4f}, "
            f"**test AUC {auc:.4f}** (직전 대비 {row['dAUC_vs_prev']:+.4f} [{row['dAUC_vs_prev_lo']:+.4f}, {row['dAUC_vs_prev_hi']:+.4f}]), "
            f"KS {row['KS']:.4f}, 보정 전 ECE {row['raw_ECE']:.4f} → 후 {row['cal_ECE']:.4f}, "
            f"PD-연령 순위상관 {row['pd_age_spearman']:+.3f}, 연령구간 AUC {row['pd_age_group_auc']:.3f}")
        commit(["results/l0/RUNLOG.md", "results/l0/ablation.csv", "docs/variable_dict_l0.csv"],
               f"L0 Tier2 {st}단계({name}): 선택 {len(selected)}개 누적, test AUC {auc:.4f}", do_commit)
        E_base, sel_names = E_new, sel_names + selected
        prev_key, prev_feats = key, n_feat
        del E_new
        gc.collect()

    log(f"소요 {(time.time() - t_all) / 60:.0f}분, 테이블 {tables}, 누수 중단 {halted}")
    with open(L0 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write("\n## git log (최근 8개, 이 로그 커밋 직전 시점)\n\n```\n" + git("log", "--oneline", "-8") + "\n```\n")
    commit(["results/l0/RUNLOG.md"], "L0 Tier2: RUNLOG 갱신", do_commit)


if __name__ == "__main__":
    main()
