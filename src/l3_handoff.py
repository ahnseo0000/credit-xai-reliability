"""L3 인수 패키지: 300명 각각의 LLM 입력 전체본 (results/l3_handoff/llm_input_300_full.jsonl) + zip.

행 단위 신청자 데이터라 git에 올리지 않는다 (results/l3_handoff/ 는 .gitignore). 팀원에게는 zip 하나로 전달한다.
- features: 변수 100개 전부 (한국어 이름, 값, 읽기용 문자열, SHAP, 방향)를 |SHAP| 큰 순서로
- 임시 거절선: test 전체 PD(동결 모형, 보정 없음)의 q80. pd >= q80 이면 거절
- 성별·연령·소득 집단은 applicant_info에 따로 둔다 (모형 입력이 아니며, 프롬프트 기본 입력에도 포함하지 않는다)
실행: python -m src.l3_handoff
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from . import config as C
from .l2_common import FROZEN, L2, L2RAW, family_map, load_cache, log
from .l2_llm_input import korean_names

OUT = C.RESULTS_DIR / "l3_handoff"
# 날짜 기준 변수: 신청일로부터의 시점 (음수 = 과거). 값이 양수인 '경과일' 변수는 이미 지난 일수라 '전'으로 읽는다.
TIME_POINT_NEG = {"DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"}
TIME_POINT_POS = {"BUREAU_OLDEST_DAYS", "PREV_DAYS_SINCE_LAST", "PRV_RJ_LIMIT_RECENCY_MIN"}
EPS = 1e-4   # |SHAP| 이 이 값 이하이면 방향 없음


def span_text(days: float) -> str:
    d = abs(days)
    if d < 30:
        return f"{d:.0f}일"
    if d < 365:
        return f"약 {d / 30.4:.0f}개월"
    y = d / 365.25
    return f"약 {y:.0f}년" if y >= 10 else f"약 {y:.1f}년"


def value_text(v: str, x, unit: str) -> str:
    if x is None:
        return "재직 정보 없음" if v == "DAYS_EMPLOYED" else "정보 없음"
    if v in TIME_POINT_NEG:
        return "신청일 당일" if x == 0 else f"{span_text(x)} {'전' if x < 0 else '후'}"
    if v in TIME_POINT_POS:
        return f"{span_text(x)} 전"
    if isinstance(x, str):
        return x
    if unit == "일":   # 기간·지연일수 (시점이 아님)
        if "LATE_DAYS" in v:
            return f"{abs(x):.0f}일 {'지연' if x > 0 else ('일찍 납부' if x < 0 else '정시')}"
        return f"{x:,.0f}일"
    if unit == "0/1":
        return "예" if x else "아니오"
    if unit == "통화 단위":
        return f"{x:,.0f}"
    if unit in ("건", "회"):
        return f"{x:,.0f}{unit}"
    if unit == "년":
        return f"{x:.0f}년"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.4g}"


def main():
    d, col_names, variables, f2v = load_cache()
    fam = family_map(variables)
    names = korean_names(variables)
    desc = {}
    for p in (C.VAR_DICT, FROZEN / "variable_dict_v2_l0.csv"):
        t = pd.read_csv(p, encoding="utf-8-sig").set_index("variable")["description"]
        desc.update({v: t[v] for v in variables if v in t.index and v not in desc})
    ids = d["test"]["ids"]
    S, base = np.load(L2RAW / "shap_test.npy"), np.load(L2RAW / "shap_base_test.npy")
    assert [str(v) for v in np.load(L2RAW / "shap_variables.npy")] == variables
    pos = {i: k for k, i in enumerate(ids)}
    pr = pd.read_csv(FROZEN / "raw" / "test_predictions.csv").set_index(C.ID_COL)
    q80 = float(np.quantile(pr["pd_xgb_v2"].to_numpy(), 0.8))
    smp = pd.read_csv(L2 / "sample_300.csv", encoding="utf-8-sig")
    vals = pd.read_pickle(L2RAW / "test_values.pkl").set_index(C.ID_COL)
    gv = pd.read_csv(C.RESULTS_DIR / "l0" / "raw" / "group_vars.csv").set_index(C.ID_COL)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "llm_input_300_full.jsonl"
    n_rej = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in smp.itertuples():
            sid = int(r.sample_id)
            k = pos[sid]
            s = S[k]
            p = float(pr.loc[sid, "pd_xgb_v2"])
            assert abs(1 / (1 + np.exp(-(base[k] + s.sum()))) - p) < 1e-4
            order = np.argsort(-np.abs(s))
            row = vals.loc[sid]
            feats = []
            for rank, j in enumerate(order, 1):
                v = variables[j]
                x = row[v]
                x = None if pd.isna(x) else (x.item() if hasattr(x, "item") else x)
                if isinstance(x, float):
                    x = float(f"{x:.7g}")   # float32 잔여 자릿수 제거
                feats.append({"abs_rank": rank, "variable": v, "korean_name": names[v]["korean_name"], "family": fam[v], "description": desc.get(v, ""),
                              "unit": names[v]["unit"], "value": x, "value_text": value_text(v, x, names[v]["unit"]),
                              "shap": round(float(s[j]), 5),
                              "direction": "부도 위험 증가" if s[j] > EPS else ("부도 위험 감소" if s[j] < -EPS else "영향 거의 없음")})
            reject = p >= q80
            n_rej += reject
            g = gv.loc[sid]
            rec = {
                "sample_id": sid,
                "cell": {"cell_id": r.cell_id, "pd_bin": r.pd_bin, "weight": round(float(r.weight), 4)},
                "model_output": {"pd": round(p, 6), "base_value_logodds": round(float(base[k]), 5), "shap_sum_logodds": round(float(s.sum()), 5),
                                 "cutoff_rule": "임시 거절선: test 전체 PD의 80번째 백분위수 (pd >= 거절선이면 거절)", "cutoff_pd": round(q80, 6),
                                 "decision": "거절" if reject else "승인"},
                "features": feats,
                "applicant_info": {"group_gender": g["group_gender"], "group_age": g["group_age"], "age_years": round(float(g["age_years"]), 1), "group_income": g["group_income"]},
                "analysis_only": {"target": int(r.target), "note": "실제 부도 여부. 프롬프트에 넣지 않는다."},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # zip: 데이터 + 사양서 사본
    zp = OUT / "l3_handoff_package.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "llm_input_300_full.jsonl")
        z.write(C.DOCS_DIR / "L3_SPEC.md", "L3_SPEC.md")
        z.write(C.DOCS_DIR / "variable_dict.csv", "variable_dict.csv")
        z.write(FROZEN / "variable_dict_v2_l0.csv", "variable_dict_v2_l0.csv")
    log(f"L3 인수 패키지 생성: 300명 x 변수 100개, 임시 거절선 q80={q80:.6f}, 300명 중 거절 {n_rej}명. zip={zp.name} ({zp.stat().st_size / 1e6:.1f}MB, git 제외)")


if __name__ == "__main__":
    main()
