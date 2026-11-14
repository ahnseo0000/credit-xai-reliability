"""L4 사람 검토용 가림 시트 생성 (분석 계획 2.5절).

(b) H8 검토 시트: 규칙 플래그(A-핵심, 구조화 필드) 건 전부(25건 초과면 25건 무작위) + 비플래그 25건, 시드 43 (`default_rng(43)`). 600건 전체(두 조건 합산)에서 추출.
(a) 사실성 검토 시트: status ok 전체에서 조건(A/B)마다 25건, 합 50건, 시드 42 (`default_rng(42)`). 요인(main_reasons + other_factors)별 행.
조건 표시는 숨기고 순서를 섞는다. 조건을 복원하는 키 파일은 results/l4/raw/ 아래(git 제외)에 따로 저장한다.
실행: python -m src.l4_review
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .l4_analysis import extract_factors, load_mapper, variables_of
from .l4_run import APPLICANT_LABELS, CALLS, INPUT, OUT, RAW, jl


def structured_text(parsed: dict) -> str:
    lines = []
    for x in parsed.get("main_reasons", []):
        lines.append(f"[주요 사유 {x.get('rank', '')}] {x.get('variable', '')} ({x.get('direction', '')}): {x.get('reason_text', '')}")
    for x in parsed.get("other_factors", []):
        if isinstance(x, dict):
            lines.append(f"[기타 요인] {x.get('name', '')}: {x.get('reason_text', '')}")
    for x in parsed.get("improvement_direction", []):
        if isinstance(x, dict):
            lines.append(f"[개선 방향] {x.get('variable', '')}: {x.get('suggestion', '')}")
    return "\n".join(lines)


def main():
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    calls = [c for c in jl(CALLS) if c["status"] == "ok"]
    df = pd.read_csv(RAW / "per_row_metrics.csv", encoding="utf-8-sig")
    flag = {(r.sample_id, r.condition): bool(r.flag_A_core) for r in df.itertuples()}
    by_key = {(c["sample_id"], c["condition"]): c for c in calls}
    keys = sorted(by_key)                                           # 600건, (sample_id, condition) 정렬
    # ---------------------------------------------------- (b) H8
    rng = np.random.default_rng(43)
    flagged = [k for k in keys if flag[k]]
    unflagged = [k for k in keys if not flag[k]]
    pick_f = flagged if len(flagged) <= 25 else [flagged[i] for i in sorted(rng.choice(len(flagged), 25, replace=False))]
    pick_u = [unflagged[i] for i in sorted(rng.choice(len(unflagged), 25, replace=False))]
    items = [(k, "플래그") for k in pick_f] + [(k, "비플래그 표집") for k in pick_u]
    order = rng.permutation(len(items))
    rows, key = [], []
    for n, i in enumerate(order, 1):
        k, grp = items[i]
        parsed = json.loads(by_key[k]["response_text"])
        rows.append({"일련번호": f"H8-{n:03d}", "사유 텍스트": structured_text(parsed), "라벨(보호변수를 사유로 사용 0/1)": "", "note": ""})
        key.append({"일련번호": f"H8-{n:03d}", "sample_id": k[0], "condition": k[1], "규칙 플래그": flag[k], "표집군": grp})
    pd.DataFrame(rows).to_csv(OUT / "review_H8_blind.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(RAW / "review_H8_key.csv", index=False, encoding="utf-8-sig")
    # ---------------------------------------------------- (a) 사실성
    rng = np.random.default_rng(42)
    picks = []
    for cond in ("A", "B"):
        ks = [k for k in keys if k[1] == cond]
        picks += [ks[i] for i in sorted(rng.choice(len(ks), 25, replace=False))]
    perm = rng.permutation(len(picks))
    mp = load_mapper(variables_of(next(iter(recs.values()))))
    rows2, key2 = [], []
    for n, i in enumerate(perm, 1):
        k = picks[i]
        rec = recs[k[0]]
        feat = {f["variable"]: f for f in rec["features"]}
        ai = rec["applicant_info"]
        parsed = json.loads(by_key[k]["response_text"])
        for j, f in enumerate(extract_factors(parsed), 1):
            v, _ = mp.map(f["text"])
            if v is not None:
                inp = f"{feat[v]['korean_name']} ({v}): {feat[v]['value_text']}"
            elif any(l in f["text"] for l in APPLICANT_LABELS):
                inp = f"(변수 목록에 없는 요인) 신청자 정보: 성별 {ai['group_gender']}, 연령대 {ai['group_age']}(만 {ai['age_years']}세), 소득 구간 {ai['group_income']}"
            else:
                inp = "(변수 목록에 없는 요인: 대응하는 입력값 없음)"
            sid = f"FACT-{n:02d}-{j}"
            rows2.append({"일련번호": sid, "실제 입력값(변수명·값)": inp, "LLM 사유 텍스트": f"{f['text']}: {f['reason']}", "라벨(값 모순 0/1/NA)": "", "라벨(원값 노출 0/1)": "", "note": ""})
            key2.append({"일련번호": sid, "sample_id": k[0], "condition": k[1], "src": f["src"], "요인 이름": f["text"], "매핑 변수": v})
    pd.DataFrame(rows2).to_csv(OUT / "review_factuality_blind.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key2).to_csv(RAW / "review_factuality_key.csv", index=False, encoding="utf-8-sig")
    # 가림 점검: 시트 텍스트에 조건을 알려주는 표현(SHAP 등)이 있는지
    hint = ("SHAP", "shap", "조건", "기여도", "log-odds", "로그오즈")
    t1 = pd.DataFrame(rows)["사유 텍스트"]
    t2 = pd.DataFrame(rows2)["LLM 사유 텍스트"]
    h1 = {w: int(t1.str.contains(w, regex=False).sum()) for w in hint}
    h2 = {w: int(t2.str.contains(w, regex=False).sum()) for w in hint}
    print("H8 시트", len(rows), "건 (플래그", len(pick_f), "+ 비플래그", len(pick_u), "), 플래그 후보 전체", len(flagged))
    print("사실성 시트", len(picks), "응답,", len(rows2), "요인 행")
    print("가림 점검(H8 시트 힌트 단어 포함 행 수):", h1)
    print("가림 점검(사실성 시트):", h2)


if __name__ == "__main__":
    main()
