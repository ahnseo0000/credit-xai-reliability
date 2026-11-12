"""Phase 2-2: WoE 스코어카드 (단조 구간화 + L2 로지스틱 + 부호 반복 제거 + 점수 변환). train/valid만 사용, test는 읽지 않는다.

입력: results/phase2/shortlist_vars.json (p2_shortlist --stage select 의 최종 후보).
조건: 연령·성별 제외, 가중치 없음(class_weight 없음), 구간화·WoE·계수는 train에서만 적합.
부호 규칙: WoE = ln(good/bad)이므로 계수는 음수여야 한다. 양수 계수가 나오면 가장 큰 양수 계수의 변수를 하나 제거하고 다시 적합(C도 다시 선택)하는 것을 양수가 없어질 때까지 반복한다.
비교표(compare_valid.md)는 p2_compare.py가 만든다.
실행: python -m src.p2_scorecard [--no-auto-sign] [--drop A,B] [--tag _x]
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from .l2_common import family_map
from .metrics import credit_metrics
from .p2_woe import (P2, Data, assign_categorical, assign_numeric, categorical_bins, iv_of, monotone_cuts)

# ---- 점수 변환 관행값 (가정): 600점에서 good:bad = 50:1, PDO(odds가 2배가 되는 점수) 20점
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_BINS_NUM, MIN_SHARE_NUM, MAX_BINS_CAT, MIN_SHARE_CAT = 6, 0.05, 6, 0.02


def log(msg: str, decision: bool = False):
    P2.mkdir(parents=True, exist_ok=True)
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def fmt(x: float) -> str:
    return f"{x:,.4g}"


def bin_labels_numeric(cuts: np.ndarray, has_missing: bool) -> dict:
    lab = {}
    edges = [-np.inf] + list(cuts) + [np.inf]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab[i] = f"< {fmt(hi)}" if i == 0 else (f">= {fmt(lo)}" if i == len(edges) - 2 else f"[{fmt(lo)}, {fmt(hi)})")
    if has_missing:
        lab[-1] = "결측"
    return lab


def bin_labels_categorical(mapping: dict, has_missing: bool) -> dict:
    groups: dict[int, list] = {}
    for c, b in mapping.items():
        if c == "__RARE__" or b is None:
            continue
        groups.setdefault(b, []).append(c)
    lab = {b: ", ".join(cs[:3]) + (f" 외 {len(cs) - 3}개" if len(cs) > 3 else "") for b, cs in groups.items()}
    if mapping.get("__RARE__") is not None:
        b = mapping["__RARE__"]
        lab[b] = (lab.get(b, "") + " (희소 범주 포함)").strip()
    if has_missing:
        lab[-1] = "결측"
    return lab


def fit_lr(W_tr: dict, y: np.ndarray, cols: list[str]):
    """C를 train 5-fold CV AUC로 고르고 train 전체로 적합한다."""
    X = np.column_stack([W_tr[v] for v in cols])
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    scores = {c: float(cross_val_score(LogisticRegression(C=c, max_iter=2000), X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()) for c in C_GRID}
    best_c = max(scores, key=scores.get)
    lr = LogisticRegression(C=best_c, max_iter=2000).fit(X, y)
    return lr, best_c, scores


def main(drop=(), tag="", auto_sign=True):
    """drop: 후보에서 미리 뺄 변수, tag: 산출물 파일명 접미사, auto_sign: 양수 계수 반복 제거."""
    shortlist = [v for v in json.loads((P2 / "shortlist_vars.json").read_text(encoding="utf-8"))["final"] if v not in drop]
    tr, va = Data("train"), Data("valid")
    fam = family_map(tr.variables)
    log(f"2-2 시작{tag}: 후보 {len(shortlist)}개 변수, train {len(tr.y):,}건 / valid {len(va.y):,}건 (test 미사용). 연령·성별 없음, class_weight 없음, 양수 계수 자동 제거 {'사용' if auto_sign else '미사용'}")
    log(f"2-2 구간화 규칙 (학습 전에 고정): 수치형은 분위수 20구간에서 아이소토닉(PAVA) 병합으로 단조 부도율 구간을 만든 뒤 최소 비중 {MIN_SHARE_NUM:.0%}, 최대 {MAX_BINS_NUM}구간으로 병합; "
        f"범주형은 희소(<{MIN_SHARE_CAT:.0%}) 범주를 묶고 부도율 순으로 인접 병합해 최대 {MAX_BINS_CAT}구간; 결측은 별도 구간; WoE = ln(good 비중/bad 비중), 평활화 0.5. optbinning 미설치라 분위수+병합 사용", decision=True)
    bins_rows, W_tr, W_va, meta = [], {}, {}, {}
    for v in shortlist:
        x_tr, x_va = tr.var(v), va.var(v)
        if tr.is_cat[v]:
            mp = categorical_bins(x_tr, tr.y, min_share=MIN_SHARE_CAT, max_bins=MAX_BINS_CAT)
            b_tr, b_va = assign_categorical(x_tr, mp), assign_categorical(x_va, mp)
            lab = bin_labels_categorical(mp, (b_tr == -1).any())
            direction = 0
        else:
            cuts, direction = monotone_cuts(x_tr, tr.y, max_bins=MAX_BINS_NUM, min_share=MIN_SHARE_NUM)
            b_tr, b_va = assign_numeric(x_tr, cuts), assign_numeric(x_va, cuts)
            lab = bin_labels_numeric(cuts, (b_tr == -1).any())
        iv_tr, t = iv_of(b_tr, tr.y)
        woe = dict(zip(t["bin"], t["woe"]))
        W_tr[v] = np.array([woe.get(b, 0.0) for b in b_tr])
        W_va[v] = np.array([woe.get(b, 0.0) for b in b_va])
        rate = (t["bad"] / t["n"]).to_numpy()
        r_nm = rate[t["bin"].to_numpy() >= 0]
        mono = bool(np.all(np.diff(r_nm) >= -1e-12) or np.all(np.diff(r_nm) <= 1e-12)) if (not tr.is_cat[v] and len(r_nm) > 1) else None
        meta[v] = {"n_bins": int(len(t)), "iv_train": iv_tr, "monotone": mono}
        _, tv = iv_of(b_va, va.y)
        vcount = dict(zip(tv["bin"], tv["n"]))
        for r in t.itertuples():
            bins_rows.append({"variable": v, "family_final": fam[v], "bin_id": int(r.bin), "bin": lab.get(int(r.bin), str(r.bin)), "n_train": int(r.n), "share_train": r.n / len(tr.y),
                              "bad_rate_train": r.bad / r.n, "woe": r.woe, "iv_contrib": r.iv, "n_valid": int(vcount.get(r.bin, 0))})
    bt_all = pd.DataFrame(bins_rows)
    nonmono = [v for v, m in meta.items() if m["monotone"] is False]
    log(f"2-2.6 단조 구간화 완료: 변수당 구간 수 {pd.Series({v: m['n_bins'] for v, m in meta.items()}).describe()[['min', '50%', 'max']].to_dict()}, 단조가 아닌 수치형 변수 {nonmono or '없음'}")

    # ---- 로지스틱 + 양수 계수 반복 제거
    cols, removed = list(shortlist), []
    while True:
        lr, best_c, scores = fit_lr(W_tr, tr.y, cols)
        coef = pd.Series(lr.coef_[0], index=cols)
        pos = coef[coef > 0]
        log(f"2-2.7 L2 로지스틱({len(cols)}개 변수): C={best_c} (train 5-fold CV AUC {scores[best_c]:.4f}); 양수 계수 {len(pos)}개 {list(pos.round(3).to_dict().items()) or ''}")
        if pos.empty or not auto_sign:
            break
        worst = pos.idxmax()
        removed.append({"variable": worst, "coef": float(pos[worst]), "iteration": len(removed) + 1, "n_vars_before": len(cols)})
        log(f"부호 규칙: WoE=ln(good/bad)라 계수는 음수여야 하는데 {worst}의 계수가 {pos[worst]:+.3f}. 가장 큰 양수 계수 변수를 제거하고 다시 적합한다", decision=True)
        cols.remove(worst)
    if removed:
        log(f"양수 계수 반복 제거 결과: {len(removed)}회 제거 {[r['variable'] for r in removed]} -> 최종 {len(cols)}개 변수 (WoE는 전체 train에서 계산한 뒤 CV해 CV AUC가 약간 낙관적일 수 있음)", decision=True)
    bt = bt_all[bt_all["variable"].isin(cols)].copy()
    Xtr = np.column_stack([W_tr[v] for v in cols])
    Xva = np.column_stack([W_va[v] for v in cols])
    p_tr, p_va = lr.predict_proba(Xtr)[:, 1], lr.predict_proba(Xva)[:, 1]
    m_tr, m_va = credit_metrics(tr.y, p_tr), credit_metrics(va.y, p_va)
    (P2 / "raw").mkdir(exist_ok=True)
    np.save(P2 / "raw" / f"valid_p_scorecard{tag}.npy", p_va)
    np.save(P2 / "raw" / f"train_p_scorecard{tag}.npy", p_tr)

    # ---- 점수 변환 (PD는 로지스틱 출력, 점수는 로짓의 선형 변환)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    n = len(cols)
    b0 = float(lr.intercept_[0])
    bt["coef"] = bt["variable"].map(coef)
    bt["points"] = offset / n - factor * (b0 / n + bt["coef"] * bt["woe"])
    bt["points_rounded"] = np.rint(bt["points"]).astype(int)
    bt.to_csv(P2 / f"woe_bins{tag}.csv", index=False, encoding="utf-8-sig")
    bt.to_csv(P2 / f"scorecard_points{tag}.csv", index=False, encoding="utf-8-sig")
    score_va = offset - factor * (b0 + Xva @ coef.to_numpy())
    woe_to_pts = {v: dict(zip(np.round(g["woe"], 10), g["points_rounded"])) for v, g in bt.groupby("variable")}   # 같은 WoE의 구간은 점수도 같다
    table_score = sum(np.array([woe_to_pts[v][round(w, 10)] for w in W_va[v]]) for v in cols)
    diff = np.abs(table_score - score_va)
    np.save(P2 / "raw" / f"valid_score{tag}.npy", score_va)
    np.save(P2 / "raw" / f"valid_table_score{tag}.npy", table_score)
    log(f"2-2.8 점수 변환(가정): 기준 {BASE_SCORE}점에서 good:bad = {BASE_ODDS}:1, PDO {PDO}점 -> Factor {factor:.3f}, Offset {offset:.2f}. 점수 = Offset − Factor×logit(PD), PD는 로지스틱 predict_proba 출력. "
        f"valid 점수 평균 {score_va.mean():.1f} (최소 {score_va.min():.0f}, 최대 {score_va.max():.0f}); 점수표 정수 합과 정확한 점수의 차이: 평균 {diff.mean():.2f}, 최대 {diff.max():.1f}점", decision=True)
    met = pd.DataFrame([{"model": f"WoE 스코어카드 ({n}개)", "split": s, **{k: m[k] for k in ("AUC", "Gini", "KS", "top_decile_lift", "top30_capture", "Brier", "ECE_10q", "mean_pred", "observed_rate")}}
                        for s, m in (("valid", m_va), ("train", m_tr))])
    met.to_csv(P2 / f"metrics{tag}.csv", index=False, encoding="utf-8-sig")
    model = {"variables": cols, "shortlist": shortlist, "removed_for_sign": removed, "C": best_c, "intercept": b0, "coef": coef.to_dict(), "factor": factor, "offset": offset,
             "base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO}
    (P2 / f"scorecard_model{tag}.json").write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"2-2 완료{tag}: 스코어카드 {n}개 변수, valid AUC {m_va['AUC']:.4f} (train {m_tr['AUC']:.4f}), KS {m_va['KS']:.4f}, ECE {m_va['ECE_10q']:.4f}, Brier {m_va['Brier']:.4f}. "
        f"저장: woe_bins{tag}.csv, scorecard_points{tag}.csv, scorecard_model{tag}.json, metrics{tag}.csv, raw/valid_p_scorecard{tag}.npy")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="", help="쉼표로 구분한 미리 제외할 변수")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-auto-sign", action="store_true")
    a = ap.parse_args()
    main(drop=tuple(x for x in a.drop.split(",") if x), tag=a.tag, auto_sign=not a.no_auto_sign)
