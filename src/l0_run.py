"""L0 Tier 1: 보조 테이블 집계 피처를 단계별로 붙이고 같은 지표로 재학습한다 (ablation).

실행: python -m src.l0_run            (전체: 집계 → 학습 → 로컬 커밋, push 안 함)
      python -m src.l0_run --build-only   (집계·EDA·키 점검만, 학습 없음)

기준선(0단계) = results/v1_noage_noweight (연령 제외, scale_pos_weight 없음). 분할은 기존 SK_ID_CURR seed-42 split.
모델 입력은 신청서 51개 + 단계별 새 변수. 성별·연령·소득분위는 모델 입력이 아니다 (results/l0/raw/group_vars.csv로 따로 보관).
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from . import l0_features as F
from .data import clean, load_raw, split_labels
from .metrics import credit_metrics, paired_auc_diffs
from .models import Calibrator, make_preprocessor
from .train import XGB_PARAMS

L0 = F.L0_DIR
BASE_DIR = C.RESULTS_DIR / "v1_noage_noweight"
LEAK_AUC = 0.81
BASE_AUC_EXPECTED = 0.7695


# ---------------------------------------------------------------- 로그
def log(msg: str, decision: bool = False):
    line = f"- `{datetime.now():%H:%M:%S}` {'**[결정]** ' if decision else ''}{msg}"
    print(line, flush=True)
    with open(L0 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, encoding="utf-8")
    return (r.stdout + r.stderr).strip()


def commit(paths: list[str], msg: str, enabled: bool):
    if not enabled:
        return
    paths = [p for p in paths if (C.ROOT / p).exists()]
    git("add", "--", *paths)
    out = git("commit", "-m", msg)  # push는 하지 않는다. Co-Authored-By도 붙이지 않는다.
    log(f"커밋: {msg.splitlines()[0]} ({git('rev-parse', '--short', 'HEAD')})")
    if "nothing to commit" in out:
        log("커밋할 변경 없음")


# ---------------------------------------------------------------- 학습·평가
def fit_eval(df_all: pd.DataFrame, split: pd.Series, feats: list[str]) -> dict:
    tr, va, te = (df_all[split == s] for s in ("train", "valid", "test"))
    y_tr, y_va, y_te = (d[C.TARGET_COL].to_numpy() for d in (tr, va, te))
    pre = make_preprocessor(tr, feats, "xgb")
    X_tr = pre.fit_transform(tr[feats])
    X_va, X_te = pre.transform(va[feats]), pre.transform(te[feats])
    model = xgb.XGBClassifier(scale_pos_weight=1.0, **XGB_PARAMS)  # 가중치 없음 (v1 기준)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    p_va_raw, p_te_raw = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
    cal = Calibrator("platt").fit(p_va_raw, y_va)
    return {
        "ids": te[C.ID_COL].to_numpy(), "y": y_te, "p_raw": p_te_raw, "p_cal": cal.predict(p_te_raw),
        "best_iter": int(model.best_iteration), "n_cols": X_tr.shape[1],
    }


def age_proxy(ids: np.ndarray, p: np.ndarray, groups: pd.DataFrame) -> dict:
    """모델 PD가 연령 정보를 담는 정도. (a) PD와 연령의 Spearman 순위상관,
    (b) PD(로짓과 제곱)만으로 연령 3구간을 5-fold CV로 맞힌 OvR macro AUC (0.5 = 정보 없음)."""
    g = groups.loc[ids]
    rho = float(spearmanr(p, g["age_years"].to_numpy()).statistic)
    x = logit(np.clip(p, 1e-7, 1 - 1e-7))
    X = np.column_stack([x, x**2])
    X = (X - X.mean(0)) / X.std(0)
    yg = g["group_age"].to_numpy()
    proba = cross_val_predict(
        LogisticRegression(max_iter=1000), X, yg, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED), method="predict_proba"
    )
    classes = np.sort(np.unique(yg))
    return {"pd_age_spearman": rho, "pd_age_group_auc": float(roc_auc_score(yg, proba, multi_class="ovr", average="macro", labels=classes))}


def ablation_row(stage, name, n_feat, n_new, res, preds, prev_key, groups, extra: dict) -> dict:
    cal, raw = credit_metrics(res["y"], res["p_cal"]), credit_metrics(res["y"], res["p_raw"])
    row = {
        "stage": stage, "name": name, "n_features": n_feat, "n_new": n_new, "best_iter": res["best_iter"],
        "AUC": cal["AUC"], "Gini": cal["Gini"], "KS": cal["KS"], "top_decile_lift": cal["top_decile_lift"],
        "raw_ECE": raw["ECE_10q"], "cal_ECE": cal["ECE_10q"], "raw_Brier": raw["Brier"], "cal_Brier": cal["Brier"],
    }
    key = f"s{stage}"
    pairs = []
    if prev_key:
        pairs.append(("vs_prev", key, prev_key))
    if stage != 0:
        pairs.append(("vs_base", key, "s0"))
    if pairs:
        d = paired_auc_diffs(res["y"], preds, pairs).set_index("comparison")
        for lab in ("vs_prev", "vs_base"):
            if lab in d.index:
                r = d.loc[lab]
                row.update({f"dAUC_{lab}": r["diff_a_minus_b"], f"dAUC_{lab}_lo": r["ci_low"], f"dAUC_{lab}_hi": r["ci_high"]})
    row.update(age_proxy(res["ids"], res["p_cal"], groups))
    row.update(extra)
    return row


def univariate_auc(df_all: pd.DataFrame, split: pd.Series, cols: list[str]) -> pd.DataFrame:
    tr, te = df_all[split == "train"], df_all[split == "test"]
    rows = []
    for c in cols:
        x = te[c].fillna(tr[c].median())
        a = roc_auc_score(te[C.TARGET_COL], x)
        rows.append({"variable": c, "test_AUC_single": max(a, 1 - a)})
    return pd.DataFrame(rows).sort_values("test_AUC_single", ascending=False)


# ---------------------------------------------------------------- IV (train만)
def iv_for(x: pd.Series, y: np.ndarray, n_bins: int = 10) -> tuple[float, int]:
    """분위수 n_bins구간(중복 경계 합침) + 결측 구간, 범주형은 범주별. 구간마다 0.5 평활."""
    if pd.api.types.is_numeric_dtype(x):
        edges = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))
        code = np.where(x.isna(), len(edges) + 1, np.searchsorted(edges, x.to_numpy(dtype=float), side="left"))
    else:
        code = x.fillna("__missing__").astype("category").cat.codes.to_numpy()
    k = int(code.max()) + 1
    bad = np.bincount(code[y == 1], minlength=k) + 0.5
    good = np.bincount(code[y == 0], minlength=k) + 0.5
    used = (np.bincount(code, minlength=k) > 0)
    bad, good = bad[used], good[used]
    b, g = bad / bad.sum(), good / good.sum()
    return float(((g - b) * np.log(g / b)).sum()), int(used.sum())


def iv_band(v: float) -> str:
    return "<0.02" if v < 0.02 else "0.02-0.1" if v < 0.1 else "0.1-0.3" if v < 0.3 else "0.3-0.5" if v < 0.5 else ">0.5 (누수 의심)"


# ---------------------------------------------------------------- 단계 실행
def run_builder(stage: int, df, base_feats, app_ids):
    """(features, eda_text) 또는 None(건너뜀). 중복 키는 예외를 그대로 올려 전체를 멈춘다."""
    attempts = [2_000_000, 1_000_000] if stage == 4 else [None, None]
    for i, chunk in enumerate(attempts, 1):
        try:
            if stage == 1:
                return F.build_stage1(df, base_feats, app_ids)
            if stage == 2:
                return F.build_stage2(app_ids)
            if stage == 3:
                return F.build_stage3(app_ids)
            return F.build_stage4(app_ids, chunk_rows=chunk)
        except MemoryError:
            log(f"{stage}단계 집계 메모리 부족 (시도 {i}/2, 청크 {chunk})")
            gc.collect()
    log(f"{stage}단계 집계가 2번 실패해 건너뜀", decision=True)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="이미 저장된 parquet이 있으면 다시 집계하지 않는다")
    ap.add_argument("--stages", default="1,2,3,4")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args(argv)
    do_commit = not args.no_commit and not args.build_only

    L0.mkdir(parents=True, exist_ok=True)
    (L0 / "raw").mkdir(exist_ok=True)
    (L0 / "RUNLOG.md").write_text(f"# L0 Tier 1 실행 로그\n\n시작 {datetime.now():%Y-%m-%d %H:%M:%S}\n\n", encoding="utf-8")
    t_all = time.time()

    log("준비 결정", decision=False)
    log("루트 PROJECT_PLAN.md는 직전 지시(루트 내용을 docs에 반영하고 루트 삭제)로 이미 삭제됨. 복원하지 않고 docs/PROJECT_PLAN.md를 유지한다. "
        "docs/PROJECT_PLAN.md의 미커밋 수정(루트 반영분)은 이 실행에서 커밋하지 않는다.", decision=True)
    log("v0-baseline 태그는 이미 d9cd7a7에 있음 (HL·PSI 결과가 d9cd7a7에서 처음 들어간 것을 git show --stat으로 확인함).", decision=True)
    log("push는 하지 않는다. 커밋 메시지에 Co-Authored-By를 넣지 않는다. v0_baseline, v1_*, comparison_2x2는 수정하지 않는다.")

    df = clean(load_raw())
    app_ids = pd.Index(df[C.ID_COL])
    assert app_ids.is_unique and len(app_ids) == 307_511
    split = split_labels(df)
    saved = pd.read_csv(BASE_DIR / "raw" / "splits.csv").set_index(C.ID_COL)["split"]
    assert (saved.reindex(app_ids).to_numpy() == split.to_numpy()).all(), "기존 분할(SK_ID_CURR seed-42)과 다르다"
    log(f"분할 확인: 기존 splits.csv와 일치 (train/valid/test = {(split=='train').sum():,}/{(split=='valid').sum():,}/{(split=='test').sum():,})")

    cfg = json.loads((BASE_DIR / "modeling" / "run_config.json").read_text(encoding="utf-8"))
    base_feats = cfg["features"]
    assert "DAYS_BIRTH" not in base_feats and "EMPLOYED_AGE_RATIO" not in base_feats and "CODE_GENDER" not in base_feats
    log(f"기준 입력 변수 {len(base_feats)}개 (연령·성별 제외 확인)")

    groups = pd.read_csv(BASE_DIR / "raw" / "group_variables.csv")
    groups.to_csv(L0 / "raw" / "group_vars.csv", index=False)
    groups = groups.set_index(C.ID_COL)

    stages = [int(s) for s in args.stages.split(",")]
    feat_store: dict[int, pd.DataFrame] = {}
    eda_paths = {}
    halted = False
    for st in stages:
        name = F.STAGE_NAMES[st]
        pq = L0 / f"features_stage{st}_{name}.parquet"
        t0 = time.time()
        try:
            if args.reuse and pq.exists():
                res = (pd.read_parquet(pq), None)
                log(f"{st}단계 집계 결과를 저장본에서 읽음")
            else:
                res = run_builder(st, df, base_feats, app_ids)
        except RuntimeError as e:  # 중복 키 / 행 수 불일치 → 전체 중단
            log(f"{st}단계 **중단**: {e}", decision=True)
            halted = True
            break
        except Exception as e:  # 예상 밖 오류는 이 단계만 건너뛰고 계속
            log(f"{st}단계 예상 밖 오류로 건너뜀: {type(e).__name__}: {e}", decision=True)
            log("```\n" + traceback.format_exc()[-1500:] + "\n```")
            continue
        if res is None:
            continue
        feat, eda = res
        feat.index.name = C.ID_COL
        assert list(feat.columns) == F.names_for(st), f"{st}단계 컬럼이 등록부와 다름"
        feat.to_parquet(pq)
        ep = L0 / f"eda_{'application' if st == 1 else name}.md"
        if eda:
            ep.write_text(eda, encoding="utf-8")
        if ep.exists():
            eda_paths[st] = ep
        feat_store[st] = feat
        log(f"{st}단계({name}) 집계 완료: {feat.shape[1]}개 변수, 키 {len(feat):,}개 (중복 없음), {time.time() - t0:.0f}초")

    if args.build_only or halted:
        log("build-only 종료" if args.build_only else "중단으로 종료")
        return

    # ---------------- ablation
    base_p = pd.read_csv(BASE_DIR / "raw" / "test_predictions.csv")
    te_ids = df.loc[split == "test", C.ID_COL].to_numpy()
    assert (base_p[C.ID_COL].to_numpy() == te_ids).all()
    y_te = base_p["target"].to_numpy()
    cm0, raw0 = credit_metrics(y_te, base_p["pd"].to_numpy()), credit_metrics(y_te, base_p["pd_raw"].to_numpy())
    assert abs(cm0["AUC"] - BASE_AUC_EXPECTED) < 5e-4, f"기준선 AUC가 {BASE_AUC_EXPECTED}와 다름: {cm0['AUC']}"
    preds = {"s0": base_p["pd"].to_numpy()}
    res0 = {"ids": te_ids, "y": y_te, "p_raw": base_p["pd_raw"].to_numpy(), "p_cal": base_p["pd"].to_numpy(),
            "best_iter": 1054}
    rows = [ablation_row(0, "baseline(v1_noage_noweight)", len(base_feats), 0, res0, preds, None, groups, {"minutes": 0.0, "note": "기존 결과 재사용"})]
    pd.DataFrame(rows).to_csv(L0 / "ablation.csv", index=False, encoding="utf-8-sig")
    log(f"0단계 기준선: AUC {cm0['AUC']:.4f}, KS {cm0['KS']:.4f}, 입력 {len(base_feats)}개 (기존 v1_noage_noweight 예측 재사용)")

    included: list[int] = []
    prev_key = "s0"
    cum_feats = list(base_feats)
    for st in stages:
        if st not in feat_store:
            continue
        name = F.STAGE_NAMES[st]
        df_all = df.join(pd.concat([feat_store[s] for s in included + [st]], axis=1), on=C.ID_COL)
        feats = base_feats + [n for s in included + [st] for n in F.names_for(s)]
        # 사전에 없는 변수는 학습에 쓰지 않는다
        known = set(pd.read_csv(C.VAR_DICT, encoding="utf-8-sig")["variable"]) | set(F.dict_rows(included + [st])["variable"])
        missing = [f for f in feats if f not in known]
        assert not missing, f"변수 사전에 없는 변수: {missing}"
        t0 = time.time()
        res = None
        for attempt in (1, 2):
            try:
                res = fit_eval(df_all, split, feats)
                break
            except (MemoryError, xgb.core.XGBoostError) as e:
                log(f"{st}단계 학습 오류 (시도 {attempt}/2): {type(e).__name__}")
                gc.collect()
        if res is None:
            log(f"{st}단계 학습이 2번 실패해 건너뜀", decision=True)
            continue
        mins = (time.time() - t0) / 60
        key = f"s{st}"
        preds[key] = res["p_cal"]
        pd.DataFrame({C.ID_COL: res["ids"], "target": res["y"], "pd": res["p_cal"], "pd_raw": res["p_raw"]}).to_csv(
            L0 / "raw" / f"test_preds_stage{st}.csv", index=False)
        row = ablation_row(st, name, len(feats), len(feats) - len(cum_feats), res, preds, prev_key, groups, {"minutes": round(mins, 1), "note": ""})
        auc = row["AUC"]
        if auc > LEAK_AUC:
            row["note"] = "누수 의심으로 중단"
            rows.append(row)
            pd.DataFrame(rows).to_csv(L0 / "ablation.csv", index=False, encoding="utf-8-sig")
            uv = univariate_auc(df_all, split, F.names_for(st))
            log(f"{st}단계 test AUC {auc:.4f} > {LEAK_AUC}: **누수 의심, 중단**", decision=True)
            log("새 변수 단독 test AUC (max(a,1-a)) 상위:\n\n```\n" + uv.head(10).round(4).to_string(index=False) + "\n```")
            commit(["results/l0/RUNLOG.md", "results/l0/ablation.csv"], f"L0 {st}단계({name}) 누수 의심으로 중단, 로그만 기록", do_commit)
            halted = True
            break
        rows.append(row)
        pd.DataFrame(rows).to_csv(L0 / "ablation.csv", index=False, encoding="utf-8-sig")
        d = F.dict_rows(included + [st])
        d.to_csv(C.DOCS_DIR / "variable_dict_l0.csv", index=False, encoding="utf-8-sig")
        dp = row.get("dAUC_vs_prev")
        log(
            f"{st}단계({name}) 학습 {mins:.1f}분: 입력 {len(feats)}개(+{len(feats) - len(cum_feats)}), best_iter {res['best_iter']}, "
            f"AUC {auc:.4f} (직전 대비 {dp:+.4f} [{row['dAUC_vs_prev_lo']:+.4f}, {row['dAUC_vs_prev_hi']:+.4f}]), KS {row['KS']:.4f}, "
            f"보정 전 ECE {row['raw_ECE']:.4f} → 후 {row['cal_ECE']:.4f}, PD-연령 순위상관 {row['pd_age_spearman']:+.3f}, 연령구간 AUC {row['pd_age_group_auc']:.3f}"
        )
        included.append(st)
        prev_key = key
        cum_feats = feats
        paths = ["results/l0/RUNLOG.md", "results/l0/ablation.csv", "docs/variable_dict_l0.csv"] + [
            str(eda_paths[s].relative_to(C.ROOT)).replace("\\", "/") for s in included if s in eda_paths
        ]
        commit(paths, f"L0 {st}단계({name}): 변수 {row['n_new']}개 추가, test AUC {auc:.4f}", do_commit)
        del df_all
        gc.collect()

    # ---------------- IV (train만, 선별 안 함)
    if included and not halted:
        df_all = df.join(pd.concat([feat_store[s] for s in included], axis=1), on=C.ID_COL)
        tr = df_all[split == "train"]
        y_tr = tr[C.TARGET_COL].to_numpy()
        src = {f: "application_train" for f in base_feats}
        for fe in F.REGISTRY:
            src[fe.name] = fe.source
        out = []
        for v in base_feats + [n for s in included for n in F.names_for(s)]:
            iv, k = iv_for(tr[v], y_tr)
            out.append({"variable": v, "source": src[v], "stage": next((f.stage for f in F.REGISTRY if f.name == v), 0),
                        "train_missing_rate": float(tr[v].isna().mean()), "n_bins": k, "IV": iv, "IV_band": iv_band(iv)})
        iv_tbl = pd.DataFrame(out).sort_values("IV", ascending=False, ignore_index=True)
        iv_tbl.to_csv(L0 / "iv_table.csv", index=False, encoding="utf-8-sig")
        hi = iv_tbl[iv_tbl["IV"] > 0.5]
        hi.to_csv(L0 / "iv_over_0.5.csv", index=False, encoding="utf-8-sig")
        log(f"IV 표 저장 (train {len(tr):,}행만 사용, 선별 안 함): 변수 {len(iv_tbl)}개, IV>0.5 {len(hi)}개 {hi['variable'].tolist()}")
        log(f"IV 상위 5: " + ", ".join(f"{r.variable} {r.IV:.3f}" for r in iv_tbl.head(5).itertuples()))
        commit(["results/l0/RUNLOG.md", "results/l0/ablation.csv", "results/l0/iv_table.csv", "results/l0/iv_over_0.5.csv"],
               f"L0: IV 표 (train만, 선별 없음), IV>0.5 변수 {len(hi)}개", do_commit)

    log(f"전체 소요 {(time.time() - t_all) / 60:.1f}분, 완료 단계 {included}, 중단 여부 {halted}")
    with open(L0 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write("\n## git log (최근 8개, 이 로그 커밋 직전 시점)\n\n```\n" + git("log", "--oneline", "-8") + "\n```\n")
    commit(["results/l0/RUNLOG.md"], "L0: RUNLOG 마무리", do_commit)


if __name__ == "__main__":
    main()
