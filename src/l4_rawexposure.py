"""원값 노출 규칙 기반 전수 계산 (docs/L4_labeling_guide.md의 정의를 규칙으로 구현). 탐색적, 확증 판정에 사용하지 않음.

대상: 600건 응답의 main_reasons / other_factors / improvement_direction의 고객용 문장(reason_text, suggestion).
규칙: R1 영문 대문자+밑줄 변수 코드명, R2 소수점 4자리 이상 수치, R3 음수 일수 표현. 원문 텍스트는 출력하지 않는다 (건수만).
실행: python -m src.l4_rawexposure
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .l4_run import B_BOOT, CALLS, OUT, SEED, jl

R1 = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")          # 영문 대문자+밑줄 변수 코드명 (예: AAA_BBB_CC)
R2 = re.compile(r"\d+\.\d{4,}")                                # 소수점 4자리 이상 수치
R3 = re.compile(r"[-−–]\s?\d[\d,]*\s*일")                      # 음수 일수 표현 (예: -1234일, −1,234 일)
RULES = [("R1", "영문 대문자+밑줄 변수 코드명", R1), ("R2", "소수점 4자리 이상 수치", R2), ("R3", "음수 일수 표현", R3)]
FIELDS = [("main_reasons", "reason_text", "variable"), ("other_factors", "reason_text", "name"), ("improvement_direction", "suggestion", "variable")]


def main():
    rows = []        # 요인 행(항목) 단위
    names_hit = {"A": 0, "B": 0}
    for c in jl(CALLS):
        if c["status"] != "ok":
            continue
        p = json.loads(c["response_text"])
        for fld, textk, namek in FIELDS:
            for x in p.get(fld, []) or []:
                if not isinstance(x, dict):
                    continue
                t = str(x.get(textk, ""))
                hit = {rid: bool(rx.search(t)) for rid, _, rx in RULES}
                rows.append({"sample_id": c["sample_id"], "condition": c["condition"], "field": fld, **hit, "any": any(hit.values())})
                if R1.search(str(x.get(namek, ""))):
                    names_hit[c["condition"]] += 1
    df = pd.DataFrame(rows)
    ids = sorted(df.sample_id.unique())
    n = len(ids)
    # 신청자 × 조건 집계
    g = df.groupby(["sample_id", "condition"]).agg(rows=("any", "size"), hits=("any", "sum"), **{r: (r, "sum") for r, _, _ in RULES}).reset_index()
    resp = g.assign(resp_any=lambda d: d.hits > 0, **{f"resp_{r}": (lambda d, r=r: d[r] > 0) for r, _, _ in RULES})
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (B_BOOT, n))
    out = {}
    for cond in "AB":
        d = resp[resp.condition == cond].set_index("sample_id").loc[ids]
        out[cond] = d
    def ci(x):
        return np.percentile(x, [2.5, 97.5])
    md = ["# L4 탐색: 원값 노출 (규칙 기반 전수)", "",
          "> **탐색적, 규칙 기반, 확증 판정에 사용하지 않음.** 사람 검토가 아니며 정의는 `docs/L4_labeling_guide.md`의 원값 노출 정의를 규칙으로 구현한 것이다. 원문 텍스트는 싣지 않았고 건수와 비율만 적는다.", "",
          "- 대상: 600건 응답 전체 (두 조건 모두 ok), 검토 필드 `main_reasons` / `other_factors` / `improvement_direction`의 고객용 문장(`reason_text`, `improvement_direction`은 `suggestion`). 변수 이름 칸(`variable`, `name`)은 규칙 적용 대상이 아니다.",
          "- **정규식** (파이썬 `re`):", "",
          "  | 규칙 | 의미 | 정규식 |", "  | --- | --- | --- |"]
    for rid, label, rx in RULES:
        md.append(f"  | {rid} | {label} | `{rx.pattern}` |")
    md += ["", "  규칙 어느 하나라도 걸리면 그 문장을 '노출'로 센다. 문장 단위는 한 항목(요인 행)이고, 응답 단위는 한 응답에 노출 항목이 하나라도 있는 경우이다.",
           f"- 단위: 신청자 {n}명 × 조건. 신뢰구간은 신청자 단위 부트스트랩(B={B_BOOT:,}, 시드 {SEED}, percentile)이다. 요인 행 비율은 신청자 묶음 재표본에서 (노출 항목 합 / 전체 항목 합)이다.", "",
           "## 1. 조건별 노출 비율", "", "| 지표 | 조건 A | 조건 B | 차이 B−A [95% CI] |", "| --- | --- | --- | --- |"]
    csv = []
    # 응답 비율
    pa, pb = out["A"].resp_any.to_numpy().astype(float), out["B"].resp_any.to_numpy().astype(float)
    la, ha = ci(pa[idx].mean(1)); lb, hb = ci(pb[idx].mean(1)); ld, hd = ci((pb - pa)[idx].mean(1))
    md.append(f"| 응답 비율 (노출 항목 ≥ 1) | {int(pa.sum())}/{n} ({pa.mean():.1%}) [{la:.1%}, {ha:.1%}] | {int(pb.sum())}/{n} ({pb.mean():.1%}) [{lb:.1%}, {hb:.1%}] | {(pb - pa).mean():+.1%} [{ld:+.1%}, {hd:+.1%}] |")
    csv.append({"metric": "response_rate", "A": pa.mean(), "A_lo": la, "A_hi": ha, "B": pb.mean(), "B_lo": lb, "B_hi": hb, "diff": (pb - pa).mean(), "diff_lo": ld, "diff_hi": hd})
    # 요인 행 비율
    ra, rb = out["A"][["hits", "rows"]].to_numpy().astype(float), out["B"][["hits", "rows"]].to_numpy().astype(float)
    fr = lambda m: m[idx][:, :, 0].sum(1) / m[idx][:, :, 1].sum(1)
    fa, fb = fr(ra), fr(rb)
    la, ha = ci(fa); lb, hb = ci(fb); ld, hd = ci(fb - fa)
    obsA, obsB = ra[:, 0].sum() / ra[:, 1].sum(), rb[:, 0].sum() / rb[:, 1].sum()
    md.append(f"| 요인 행 비율 (노출 항목 / 전체 항목) | {int(ra[:, 0].sum())}/{int(ra[:, 1].sum())} ({obsA:.1%}) [{la:.1%}, {ha:.1%}] | {int(rb[:, 0].sum())}/{int(rb[:, 1].sum())} ({obsB:.1%}) [{lb:.1%}, {hb:.1%}] | {obsB - obsA:+.1%} [{ld:+.1%}, {hd:+.1%}] |")
    csv.append({"metric": "row_rate", "A": obsA, "A_lo": la, "A_hi": ha, "B": obsB, "B_lo": lb, "B_hi": hb, "diff": obsB - obsA, "diff_lo": ld, "diff_hi": hd})
    md += ["", "## 2. 규칙별 건수 (한 문장이 여러 규칙에 걸릴 수 있다)", "", "| 규칙 | 응답 수 A | 응답 수 B | 노출 항목 수 A | 노출 항목 수 B |", "| --- | --- | --- | --- | --- |"]
    for rid, label, _ in RULES:
        ra_ = int(out["A"][f"resp_{rid}"].sum()); rb_ = int(out["B"][f"resp_{rid}"].sum())
        ia, ib = int(df[(df.condition == "A")][rid].sum()), int(df[(df.condition == "B")][rid].sum())
        md.append(f"| {rid} {label} | {ra_} | {rb_} | {ia} | {ib} |")
        csv.append({"metric": f"count_{rid}", "A": ia, "B": ib, "A_resp": ra_, "B_resp": rb_})
    md += ["", "## 3. 필드별 항목 수와 노출 항목 수", "", "| 필드 | 항목 수 A | 노출 A | 항목 수 B | 노출 B |", "| --- | --- | --- | --- | --- |"]
    for fld, _, _ in FIELDS:
        a_ = df[(df.condition == "A") & (df.field == fld)]; b_ = df[(df.condition == "B") & (df.field == fld)]
        md.append(f"| {fld} | {len(a_)} | {int(a_["any"].sum())} | {len(b_)} | {int(b_["any"].sum())} |")
    md += ["", f"- 참고: 변수 이름 칸(`variable`, `name`)에 R1 형태의 코드명이 적힌 항목은 A {names_hit['A']}건, B {names_hit['B']}건이다 (이름 칸은 위 비율에 포함하지 않았다).",
           "- 해석 문장은 쓰지 않았다. 규칙은 단순 패턴이라 \"가공 전 수치\"를 정확히 가려내지 못한다 (예: 소수점 4자리 이상 수치는 가공 전 값일 수도, 사람이 다듬은 값일 수도 있다). 사람 검토(사실성 시트의 원값 노출 열)는 실행하지 않았다.", ""]
    (OUT / "exploratory_raw_exposure.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(csv).to_csv(OUT / "exploratory_raw_exposure.csv", index=False, encoding="utf-8-sig")
    print("\n".join(md[12:40]))


if __name__ == "__main__":
    main()
