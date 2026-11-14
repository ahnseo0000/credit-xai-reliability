"""L4 완전성 기술 보고: 출력 스키마 필수 키·항목이 있는 비율, 주요 기준 요인의 매핑률 (조건별). 응답 원문은 싣지 않는다.

주의: 내용이 4요소를 실제로 담았는지는 평가하지 않는다. 필드의 존재·형식과 변수 매핑만 센다.
실행: python -m src.l4_completeness
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .l4_analysis import DIRS, extract_factors, load_mapper, variables_of
from .l4_run import CALLS, INPUT, OUT, cp_ci, jl


def main():
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    mp = load_mapper(variables_of(next(iter(recs.values()))))
    rows = []
    for c in jl(CALLS):
        ok = c["status"] == "ok"
        p = json.loads(c["response_text"]) if ok else {}
        res = p.get("result") if isinstance(p.get("result"), dict) else {}
        mr = p.get("main_reasons") if isinstance(p.get("main_reasons"), list) else []
        imp = p.get("improvement_direction") if isinstance(p.get("improvement_direction"), list) else []
        other = p.get("other_factors") if isinstance(p.get("other_factors"), list) else []
        nonempty = lambda x: isinstance(x, str) and len(x.strip()) > 0
        keys5 = all(k in p for k in ("result", "main_reasons", "basis_info_overview", "improvement_direction", "other_factors"))
        mr_ok = len(mr) == 4 and all(isinstance(x, dict) and all(k in x for k in ("rank", "variable", "direction", "reason_text")) and x.get("direction") in DIRS and nonempty(x.get("reason_text")) for x in mr)
        maps = [mp.map(x.get("variable", ""))[0] for x in mr if isinstance(x, dict)]
        fac = extract_factors(p) if ok else []
        fmaps = [mp.map(f["text"])[0] for f in fac]
        imp_maps = [mp.map(x.get("variable", ""))[0] for x in imp if isinstance(x, dict)]
        rows.append({"condition": c["condition"], "ok": ok, "keys5": keys5,
                     "decision_summary": nonempty(res.get("decision")) and nonempty(res.get("summary")), "main4": mr_ok, "basis_nonempty": nonempty(p.get("basis_info_overview")),
                     "improve_1to3": 1 <= len(imp) <= 3 and all(isinstance(x, dict) and nonempty(x.get("variable")) and nonempty(x.get("suggestion")) for x in imp), "other_is_list": isinstance(p.get("other_factors"), list),
                     "all_elements": keys5 and mr_ok and nonempty(res.get("decision")) and nonempty(res.get("summary")) and nonempty(p.get("basis_info_overview")) and 1 <= len(imp) <= 3,
                     "n_main": len(maps), "n_main_mapped": sum(v is not None for v in maps), "n_fac": len(fmaps), "n_fac_mapped": sum(v is not None for v in fmaps),
                     "n_imp": len(imp_maps), "n_imp_mapped": sum(v is not None for v in imp_maps), "n_other": len(other)})
    df = pd.DataFrame(rows)
    md = ["# L4 완전성 (기술 보고)", "",
          "> **내용이 4요소를 실제로 담았는지는 평가하지 않았다.** 아래는 출력 JSON의 필드가 있는지(존재·형식)와 변수 매핑률만 센 기술 통계이다. 완전성 축은 분석 계획에 항목이 없는 탐색 보고이며 확증 판정에 쓰지 않는다. 응답 원문은 싣지 않았다.", "",
          "- 4요소와 스키마 필드의 대응(L3_SPEC 3절, `docs/PROJECT_PLAN.md` L3-3): 평가 결과 = `result.decision`·`result.summary`, 주요 판단 기준 상위 4개(방향 포함) = `main_reasons`, 사용된 기초정보 개요 = `basis_info_overview`, 개선 방향 = `improvement_direction`. 목록 밖 요인 `other_factors`는 환각 측정용 부가 필드이다.",
          f"- 대상: 600건 (조건 A {int((df.condition == 'A').sum())}, B {int((df.condition == 'B').sum())}), status ok A {int(df[df.condition == 'A'].ok.sum())}, B {int(df[df.condition == 'B'].ok.sum())}.", "",
          "## 1. 필드가 있는 비율 (조건별, Clopper-Pearson 95% 구간)", "", "| 항목 | 조건 A | 조건 B |", "| --- | --- | --- |"]
    items = [("keys5", "최상위 필수 키 5개(result, main_reasons, basis_info_overview, improvement_direction, other_factors)가 모두 있음"),
             ("decision_summary", "result.decision과 result.summary가 비어 있지 않음 (평가 결과)"), ("main4", "main_reasons가 정확히 4개이고 각각 rank·variable·direction(허용 값)·reason_text(비어 있지 않음)를 가짐 (주요 판단 기준)"),
             ("basis_nonempty", "basis_info_overview가 비어 있지 않음 (기초정보 개요)"), ("improve_1to3", "improvement_direction이 1~3개이고 각각 variable·suggestion을 가짐 (개선 방향)"),
             ("other_is_list", "other_factors가 리스트로 존재함 (비어 있어도 됨)"), ("all_elements", "위 4요소 필드가 모두 갖춰진 응답")]
    csv_rows = []
    for col, label in items:
        cells = []
        for c in "AB":
            d = df[df.condition == c]
            k, n = int(d[col].sum()), len(d)
            lo, hi = cp_ci(k, n)
            cells.append(f"{k}/{n} ({k / n:.1%}) [{lo:.1%}, {hi:.1%}]")
            csv_rows.append({"item": col, "condition": c, "count": k, "n": n, "rate": k / n})
        md.append(f"| {label} | {cells[0]} | {cells[1]} |")
    md += ["", "## 2. 변수 매핑률 (규칙 기반 매핑, 동결된 별칭 파일)", "", "| 항목 | 조건 A | 조건 B |", "| --- | --- | --- |"]
    for label, num, den in (("주요 기준 요인(main_reasons) 매핑률 = 매핑된 항목 / 전체 항목", "n_main_mapped", "n_main"), ("main_reasons + other_factors 매핑률 (환각률의 보수)", "n_fac_mapped", "n_fac"),
                            ("개선 방향(improvement_direction)의 variable 매핑률", "n_imp_mapped", "n_imp")):
        cells = []
        for c in "AB":
            d = df[df.condition == c]
            k, n = int(d[num].sum()), int(d[den].sum())
            lo, hi = cp_ci(k, n)
            cells.append(f"{k}/{n} ({k / n:.2%}) [{lo:.2%}, {hi:.2%}]")
            csv_rows.append({"item": num, "condition": c, "count": k, "n": n, "rate": k / n})
        md.append(f"| {label} | {cells[0]} | {cells[1]} |")
    allmap = {c: int((df[df.condition == c].n_main_mapped == 4).sum()) for c in "AB"}
    md += ["", f"- main_reasons 4개가 전부 매핑된 응답: A {allmap['A']}/300, B {allmap['B']}/300.",
           f"- other_factors 항목 수 합계: A {int(df[df.condition == 'A'].n_other.sum())}, B {int(df[df.condition == 'B'].n_other.sum())} (환각률 분모에 포함, 항목이 하나도 없는 응답 A {int((df[df.condition == 'A'].n_other == 0).sum())}, B {int((df[df.condition == 'B'].n_other == 0).sum())}).",
           "- 제36조의2 제1항 4요소의 법적 정의에 비추어 내용이 충분한지, 설명이 사실과 맞는지는 이 보고에서 판단하지 않았다.", ""]
    (OUT / "completeness_descriptive.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(OUT / "completeness_descriptive.csv", index=False, encoding="utf-8-sig")
    print("\n".join(md[8:30]))


if __name__ == "__main__":
    main()
