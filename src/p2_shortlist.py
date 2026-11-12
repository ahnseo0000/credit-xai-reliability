"""Phase 2-1: v2 입력 100개 -> WoE/IV -> IV 필터 -> family 안 상관 정리 -> 후보 15~25개 (train/valid만 사용, test는 읽지 않는다).

실행: python -m src.p2_shortlist --stage iv        (IV·상관표만 만든다)
      python -m src.p2_shortlist --stage select    (선택 규칙 적용, shortlist.csv)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .l2_common import family_map
from .p2_woe import (P2, Data, MISSING, assign_categorical, assign_numeric, categorical_bins, fine_cuts, iv_of)

FORBIDDEN = {"DAYS_BIRTH", "CODE_GENDER", "EMPLOYED_AGE_RATIO", "AGE", "AGE_YEARS", "GENDER", "SEX"}
IV_MIN = 0.02            # 사전 고정
CORR_MAX = 0.7           # family 안 WoE 변환값 상관 상한
EXT_KEEP = "EXT_SOURCE_MEAN"
B_LIST = ["NAME_FAMILY_STATUS", "INCOME_PER_PERSON", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT_W_CITY", "EXT_SOURCE_1", "DAYS_EMPLOYED",
          "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "BUREAU_OLDEST_DAYS", "NAME_INCOME_TYPE", "DEF_60_CNT_SOCIAL_CIRCLE", "NAME_EDUCATION_TYPE"]   # L3_SPEC 7절 B 목록


def log(msg: str, decision: bool = False):
    P2.mkdir(parents=True, exist_ok=True)
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def stage_iv():
    tr, va = Data("train"), Data("valid")
    # 2-1.1 연령·성별 확인
    hit = sorted(set(tr.variables) & FORBIDDEN)
    col_hit = [c for c in tr.col_names if any(c.upper().startswith(f) for f in FORBIDDEN)]
    feat = json.loads((C.RESULTS_DIR / "v2_frozen" / "features.json").read_text(encoding="utf-8"))
    assert not hit and not col_hit and not (set(feat["application_features"] + feat["l0_features"]) & FORBIDDEN)
    gv = pd.read_csv(C.RESULTS_DIR / "l0" / "raw" / "group_vars.csv", usecols=["SK_ID_CURR", "split", "age_years"]).set_index("SK_ID_CURR")
    age = gv.loc[tr.ids, "age_years"].to_numpy()          # train 행만 (분할은 gv의 split과 일치 확인)
    assert (gv.loc[tr.ids, "split"] == "train").all() and (gv.loc[va.ids, "split"] == "valid").all()
    log(f"2-1.1 연령·성별 확인: v2 입력 {len(tr.variables)}개 변수명과 {len(tr.col_names)}개 입력 열에 DAYS_BIRTH/CODE_GENDER/EMPLOYED_AGE_RATIO/AGE/GENDER 없음 (통과). "
        f"train {len(tr.y):,}건(부도 {tr.y.mean():.4f}), valid {len(va.y):,}건 사용, test 미사용", decision=True)
    fam = family_map(tr.variables)
    rows, woe_cols = [], {}
    for v in tr.variables:
        x_tr, x_va = tr.var(v), va.var(v)
        if tr.is_cat[v]:
            mp = categorical_bins(x_tr, tr.y, min_share=0.02, max_bins=10)
            b_tr, b_va = assign_categorical(x_tr, mp), assign_categorical(x_va, mp)
            miss_rate = float((x_tr == MISSING).mean())
            nuniq = len(set(x_tr))
        else:
            cuts = fine_cuts(x_tr, n_q=10)
            b_tr, b_va = assign_numeric(x_tr, cuts), assign_numeric(x_va, cuts)
            miss_rate = float(np.isnan(x_tr).mean())
            nuniq = int(len(np.unique(x_tr[~np.isnan(x_tr)])))
        iv_tr, t = iv_of(b_tr, tr.y)
        iv_va, _ = iv_of(b_va, va.y)
        woe_map = dict(zip(t["bin"], t["woe"]))
        w = np.array([woe_map.get(b, 0.0) for b in b_tr], dtype=np.float32)
        woe_cols[v] = w
        rows.append({"variable": v, "family_final": fam[v], "type": "범주" if tr.is_cat[v] else "수치", "missing_rate": miss_rate, "n_unique": nuniq,
                     "n_bins_fine": int(len(t)), "iv_train": iv_tr, "iv_valid": iv_va,
                     "abs_corr_age_woe": float(abs(np.corrcoef(w, age)[0, 1])) if w.std() > 0 else 0.0,
                     "min_bin_share": float(t["n"].min() / len(b_tr))})
    iv = pd.DataFrame(rows).sort_values("iv_train", ascending=False)
    P2.mkdir(parents=True, exist_ok=True)
    iv.to_csv(P2 / "iv_all.csv", index=False, encoding="utf-8-sig")
    # family 안 상관 (WoE 변환값 피어슨)
    prs = []
    for g, vs in pd.Series(fam).groupby(pd.Series(fam)):
        names = list(vs.index)
        for a, b in itertools.combinations(names, 2):
            r = float(np.corrcoef(woe_cols[a], woe_cols[b])[0, 1]) if woe_cols[a].std() > 0 and woe_cols[b].std() > 0 else 0.0
            prs.append({"family_final": g, "a": a, "b": b, "corr": r})
    pd.DataFrame(prs).to_csv(P2 / "corr_within_family.csv", index=False, encoding="utf-8-sig")
    log(f"2-1.2 WoE/IV 계산(train 기준, 분위수 10구간·결측 별도 구간·범주형은 희소 병합, 평활화 {0.5}): IV≥{IV_MIN} {int((iv.iv_train >= IV_MIN).sum())}개 / 100, "
        f"IV≥0.1 {int((iv.iv_train >= 0.1).sum())}개. family 안 상관쌍 {len(prs)}개 저장 (corr_within_family.csv, iv_all.csv)")
    return iv


CAP_BIG, CAP_SMALL, BIG_FAMILY_N = 3, 2, 10   # family 상한: v2 입력 변수가 10개 이상인 family는 3개, 그 외 2개 (모델 학습 전에 고정)


def stage_select(priority: str = "gain"):
    """priority='gain': 우선순위에 train gain 순위를 쓴다 (test 정보 없음, 기본). 'boot': 처음 버전(L2 부트스트랩 안정성, test SHAP 기반) 재현용."""
    from .l2_llm_input import korean_names

    iv = pd.read_csv(P2 / "iv_all.csv", encoding="utf-8-sig").set_index("variable")
    cr = pd.read_csv(P2 / "corr_within_family.csv", encoding="utf-8-sig")
    cm = {}
    for r in cr.itertuples():
        cm[(r.a, r.b)] = cm[(r.b, r.a)] = r.corr
    vs = list(iv.index)
    gain = pd.read_csv(C.RESULTS_DIR / "l0_v2" / "gain_importance_190.csv", encoding="utf-8-sig").set_index("variable")["total_gain"]
    gain_rank = gain.reindex(vs).rank(ascending=False, method="min")        # 100개 변수 안에서의 train gain 순위 (190개 모형, train 기준)
    if priority == "boot":
        bvars = [str(v) for v in np.load(C.RESULTS_DIR / "l2" / "raw" / "shap_variables.npy")]
        vm = np.load(C.RESULTS_DIR / "l2" / "raw" / "boot_var_meanabs.npy")
        rk = (-vm).argsort(axis=1).argsort(axis=1) + 1
        boot = pd.DataFrame({"boot_top4": (rk <= 4).mean(0), "boot_top6": (rk <= 6).mean(0), "boot_top10": (rk <= 10).mean(0)}, index=bvars)
        stab = boot["boot_top10"]
        stab_name, stab_txt = "부트스트랩 상위10위 안 비율", (lambda v: f"안정성 상위10위 안 {stab[v]:.0%}")
    else:
        stab = 1 - (gain_rank - 1) / (len(vs) - 1)                           # 1(최고 gain) ~ 0(최저)
        stab_name, stab_txt = "train gain 순위 백분위(1=최고)", (lambda v: f"train gain 순위 {int(gain_rank[v])}/100")
    kn = korean_names(vs)
    n_in_family = iv["family_final"].value_counts()
    log(f"2-1.3 선택 규칙 (학습 전에 고정, 우선순위 기준 = {'부트스트랩(L2, test SHAP 기반; 첫 버전)' if priority == 'boot' else 'train gain (test 정보 배제)'}): ① train IV < {IV_MIN} 제거 ② EXT_SOURCE 계열은 {EXT_KEEP}만 유지 "
        f"③ 같은 family_final 안에서 IV 큰 순으로 보며 WoE 변환값 |상관| > {CORR_MAX}인 이미 유지된 변수가 있으면 탈락 ④ 남은 변수에 priority = IV × (0.5 + 0.5 × {stab_name})를 매겨 "
        f"family당 상한(v2 입력 {BIG_FAMILY_N}개 이상인 family {CAP_BIG}개, 그 외 {CAP_SMALL}개)만 유지 → 최대 25개. 대리 후보(L3_SPEC B 목록)는 표시만 하고 자동 제외하지 않는다.", decision=True)
    status, reason, prio = {}, {}, {}
    cand = []
    for v in vs:
        if iv.loc[v, "iv_train"] < IV_MIN:
            status[v], reason[v] = "탈락", f"IV {iv.loc[v, 'iv_train']:.4f} < {IV_MIN} (train)"
        elif v.startswith("EXT_SOURCE") and v != EXT_KEEP:
            status[v] = "탈락"
            r = cm.get((v, EXT_KEEP))
            reason[v] = (f"EXT_SOURCE 계열은 {EXT_KEEP} 하나만 유지: 개별 점수·MIN/MAX/PROD는 같은 세 점수에서 나온 중복 정보 (WoE 상관 {EXT_KEEP}와 {abs(r):.2f})라 한 모형에 함께 넣으면 계수가 불안정하고 "
                         f"해석이 어렵다. {EXT_KEEP}는 IV 최고({iv.loc[EXT_KEEP, 'iv_train']:.3f})이고 결측이 거의 없다 ({iv.loc[EXT_KEEP, 'missing_rate']:.2%}). 이 변수 자체 IV {iv.loc[v, 'iv_train']:.3f}")
        else:
            cand.append(v)
    kept = []
    for v in sorted(cand, key=lambda x: -iv.loc[x, "iv_train"]):
        clash = [(k, abs(cm.get((v, k), 0.0))) for k in kept if iv.loc[k, "family_final"] == iv.loc[v, "family_final"] and abs(cm.get((v, k), 0.0)) > CORR_MAX]
        if clash:
            k, r = max(clash, key=lambda t: t[1])
            status[v], reason[v] = "탈락", f"같은 family({iv.loc[v, 'family_final']})의 {k}와 WoE 상관 {r:.2f} > {CORR_MAX}, IV가 더 낮음 ({iv.loc[v, 'iv_train']:.3f} < {iv.loc[k, 'iv_train']:.3f})"
        else:
            kept.append(v)
    for v in kept:
        prio[v] = iv.loc[v, "iv_train"] * (0.5 + 0.5 * stab[v])
    final = []
    for g, vsg in pd.Series({v: iv.loc[v, "family_final"] for v in kept}).groupby(lambda v: iv.loc[v, "family_final"]):
        cap = CAP_BIG if n_in_family[g] >= BIG_FAMILY_N else CAP_SMALL
        order = sorted(vsg.index, key=lambda x: -prio[x])
        for i, v in enumerate(order):
            if i < cap:
                final.append(v)
                status[v], reason[v] = "유지", f"IV {iv.loc[v, 'iv_train']:.3f}, family({g}) 상한 {cap}개 중 {i + 1}번째 (priority {prio[v]:.4f}, {stab_txt(v)}); 상관 정리 통과"
            else:
                status[v], reason[v] = "탈락", f"family({g}) 상한 {cap}개 초과: priority {prio[v]:.4f}로 유지 변수({', '.join(order[:cap])})보다 낮다 (IV {iv.loc[v, 'iv_train']:.3f}, {stab_txt(v)})"
    assert 15 <= len(final) <= 25, len(final)
    feat = json.loads((C.RESULTS_DIR / "v2_frozen" / "features.json").read_text(encoding="utf-8"))
    app = set(feat["application_features"])
    out = iv.copy()
    out.insert(1, "korean_name", [kn[v]["korean_name"] for v in out.index])
    out.insert(3, "source", ["신청서/Tier1 파생" if v in app else "L0" for v in out.index])
    if priority == "boot":
        for c in ("boot_top4", "boot_top6", "boot_top10"):
            out[c] = boot.loc[out.index, c]
    out["gain_rank_train_190"] = gain_rank.reindex(out.index)
    out["proxy_B"] = ["Y" if v in B_LIST else "" for v in out.index]
    out["priority"] = pd.Series(prio).reindex(out.index)
    out["status"] = [status[v] for v in out.index]
    out["reason"] = [reason[v] for v in out.index]
    out = out.reset_index().sort_values(["status", "family_final", "iv_train"], ascending=[False, True, False])
    out.to_csv(P2 / "shortlist.csv", index=False, encoding="utf-8-sig")
    (P2 / "shortlist_vars.json").write_text(json.dumps({"final": final, "priority": priority}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"2-1.3 결과({priority}): ①IV 미달 {sum(1 for v in vs if iv.loc[v, 'iv_train'] < IV_MIN)}개 ②EXT 개별/파생 탈락 {sum(1 for v in vs if v.startswith('EXT_SOURCE') and v != EXT_KEEP and iv.loc[v, 'iv_train'] >= IV_MIN)}개 "
        f"③상관 정리 후 {len(kept)}개 ④family 상한 후 최종 {len(final)}개. 대리 후보(B 목록) 포함 {sum(1 for v in final if v in B_LIST)}개: {[v for v in final if v in B_LIST]}")
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["iv", "select"])
    ap.add_argument("--priority", default="gain", choices=["gain", "boot"])
    a = ap.parse_args()
    if a.stage == "iv":
        stage_iv()
    else:
        stage_select(a.priority)
