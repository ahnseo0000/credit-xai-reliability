"""L2 -> L3: 300건 각각의 LLM 입력 자료.

산출 (results/l2/):
- llm_input_300.jsonl  한 줄 = 한 고객. 설명에 쓰는 정보(pd, 상위 SHAP 기여, 변수값)와 RQ6용 신청자 정보(applicant_info)를 필드로 분리한다.
- llm_input_300.csv    interface.md 형식(wide): sample_id, target, pd, base_value, group_*, feat__<변수>, shap__<변수>
- llm_input_300_long.csv  고객 x 변수 (값, SHAP, 순위, family)
성별·연령·소득 집단은 모형 입력이 아니며, 설명 생성 프롬프트에는 넣지 않고 RQ6 분석(applicant_info)에서만 쓴다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .l2_common import L2, L2RAW, FROZEN, family_map, load_cache, log

TOP_N = 10
TOP_FAMILY = 5


def korean_names(variables):
    a = pd.read_csv(C.VAR_DICT, encoding="utf-8-sig").set_index("variable")
    l0 = pd.read_csv(FROZEN / "variable_dict_v2_l0.csv", encoding="utf-8-sig").set_index("variable")
    out = {}
    for v in variables:
        r = a.loc[v] if v in a.index else l0.loc[v]
        out[v] = {"korean_name": r["korean_name"], "unit": r["unit"] if isinstance(r["unit"], str) else ""}
    return out


def display(v, val, unit):
    if val is None:
        return "결측"
    if v.startswith("DAYS_") and val < 0:
        return f"{-val / 365.25:.1f}년 전"
    if isinstance(val, float):
        return f"{val:,.4g}" if abs(val) < 1000 else f"{val:,.0f}"
    return str(val)


def main():
    d, col_names, variables, f2v = load_cache()
    fam = family_map(variables)
    names = korean_names(variables)
    ids = d["test"]["ids"]
    S = np.load(L2RAW / "shap_test.npy")
    base = np.load(L2RAW / "shap_base_test.npy")
    pos = {i: k for k, i in enumerate(ids)}
    smp = pd.read_csv(L2 / "sample_300.csv", encoding="utf-8-sig")
    vals = pd.read_pickle(L2RAW / "test_values.pkl").set_index(C.ID_COL)
    gv = pd.read_csv(C.RESULTS_DIR / "l0" / "raw" / "group_vars.csv").set_index(C.ID_COL)
    pr = pd.read_csv(FROZEN / "raw" / "test_predictions.csv").set_index(C.ID_COL)

    recs, wide, long = [], [], []
    for r in smp.itertuples():
        sid = int(r.sample_id)
        k = pos[sid]
        s = S[k]
        pd_ = float(pr.loc[sid, "pd_xgb_v2"])
        assert abs(1 / (1 + np.exp(-(base[k] + s.sum()))) - pd_) < 1e-4, "SHAP 합 + 기준값이 pd와 다르다"
        order = np.argsort(-np.abs(s))
        row_vals = vals.loc[sid]
        contribs = []
        for rank, j in enumerate(order, 1):
            v = variables[j]
            x = row_vals[v]
            x = None if pd.isna(x) else (x.item() if hasattr(x, "item") else x)
            long.append({"sample_id": sid, "variable": v, "korean_name": names[v]["korean_name"], "family": fam[v], "value": x, "shap": float(s[j]), "abs_rank": rank})
            if rank <= TOP_N:
                contribs.append({"rank": rank, "variable": v, "korean_name": names[v]["korean_name"], "family": fam[v], "value": x, "value_display": display(v, x, names[v]["unit"]),
                                 "unit": names[v]["unit"], "shap": round(float(s[j]), 5), "direction": "부도 위험을 높임" if s[j] > 0 else "부도 위험을 낮춤"})
        rest = float(s[order[TOP_N:]].sum())
        fsum = pd.Series(s, index=variables).groupby(fam).sum()
        fam_top = [{"family": f, "shap": round(float(fsum[f]), 5), "direction": "높임" if fsum[f] > 0 else "낮춤"} for f in fsum.abs().sort_values(ascending=False).index[:TOP_FAMILY]]
        g = gv.loc[sid]
        recs.append({
            "sample_id": sid, "cell_id": r.cell_id,
            "explanation_input": {"pd": round(pd_, 6), "base_value_logodds": round(float(base[k]), 5), "top_contributions": contribs,
                                  "other_variables_shap_sum": round(rest, 5), "top_family_contributions": fam_top},
            "applicant_info_for_RQ6": {"group_gender": g["group_gender"], "group_age": g["group_age"], "age_years": round(float(g["age_years"]), 1), "group_income": g["group_income"]},
            "target": int(r.target), "pd_bin": r.pd_bin,
        })
        w = {"sample_id": sid, "cell_id": r.cell_id, "target": int(r.target), "pd": pd_, "base_value": float(base[k]), "group_gender": g["group_gender"],
             "group_age": g["group_age"], "group_income": g["group_income"]}
        w.update({f"feat__{v}": row_vals[v] for v in variables})
        w.update({f"shap__{v}": float(s[j]) for j, v in enumerate(variables)})
        wide.append(w)
    with open(L2 / "llm_input_300.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    pd.DataFrame(wide).to_csv(L2 / "llm_input_300.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long).to_csv(L2 / "llm_input_300_long.csv", index=False, encoding="utf-8-sig")
    log(f"L3용 입력 자료 300건 생성: 상위 {TOP_N}개 기여(방향·변수값 포함) + 기타 합계 + family 상위 {TOP_FAMILY}, RQ6용 신청자 정보는 별도 필드. 모두 SHAP 합 + 기준값 = pd 확인 통과")


if __name__ == "__main__":
    main()
