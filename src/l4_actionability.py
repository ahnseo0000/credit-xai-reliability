"""L4 탐색: 실행가능성 지표 (분석 계획 3절, 계산 전에 커밋). 탐색적, 확증 판정에 사용하지 않음. 응답 원문은 싣지 않는다.

(1) 신청자별 SHAP 상위 4 변수의 class 비율 평균, (2) 같은 비율을 LLM main_reasons 매핑 변수에 대해 조건별로,
민감도 (가) NAME_EDUCATION_TYPE을 장기변경으로, (나) FLAG_DOCUMENT_3 제외.
실행: python -m src.l4_actionability
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .l4_analysis import load_mapper, variables_of
from .l4_run import B_BOOT, CALLS, C, INPUT, OUT, SEED, jl

CLASSES = ["단기변경", "장기변경", "변경불가", "해당없음·간접"]


def class_shares(vars_: list[str], cmap: dict, drop=()):
    vs = [v for v in vars_ if v not in drop]
    if not vs:
        return None
    return np.array([sum(cmap[v] == c for v in vs) / len(vs) for c in CLASSES])


def boot_ci(mat: np.ndarray, rng):
    n = len(mat)
    idx = rng.integers(0, n, (B_BOOT, n))
    m = mat[idx].mean(1)
    return np.percentile(m, [2.5, 97.5], axis=0)


def main():
    act = pd.read_csv(C.DOCS_DIR / "variable_actionability.csv", encoding="utf-8-sig")
    assert len(act) == 100 and set(act["class"]) <= set(CLASSES)
    base_map = dict(zip(act.variable, act["class"]))
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    mp = load_mapper(variables_of(next(iter(recs.values()))))
    calls = {(c["sample_id"], c["condition"]): c for c in jl(CALLS) if c["status"] == "ok"}
    ids = sorted(recs)
    shap_top = {sid: [f["variable"] for f in sorted(recs[sid]["features"], key=lambda f: f["shap_rank"]) if f["shap_rank"] <= 4] for sid in ids}
    llm = {}
    for cond in "AB":
        for sid in ids:
            parsed = json.loads(calls[(sid, cond)]["response_text"])
            seen = []
            for x in parsed["main_reasons"][:4]:
                v, _ = mp.map(x.get("variable", ""))
                if v is not None and v not in seen:
                    seen.append(v)
            llm[(sid, cond)] = seen
    edu_long = dict(base_map)
    edu_long["NAME_EDUCATION_TYPE"] = "장기변경"
    scenarios = [("기본 (분류표 그대로)", base_map, ()), ("민감도 (가): NAME_EDUCATION_TYPE을 장기변경으로", edu_long, ()), ("민감도 (나): FLAG_DOCUMENT_3 제외", base_map, ("FLAG_DOCUMENT_3",))]
    rng = np.random.default_rng(SEED)
    md = ["# L4 탐색: 실행가능성 지표", "",
          "> **탐색적, 확증 판정에 사용하지 않음.** 보정 없음, p값 없음. 비율과 부트스트랩 95% 구간(B=10,000, 시드 42, 신청자 단위)만 적는다. 응답 원문은 싣지 않았다.", "",
          "- 정의: `docs/analysis_plan.md` 3절 (계산 전에 커밋 `52ea1b3`). 분류표: `docs/variable_actionability.csv` (커밋 `c428f70`, class = 단기변경 / 장기변경 / 변경불가 / 해당없음·간접).",
          "- **SHAP 상위 4**: 신청자별 `shap_rank` 1~4 변수 중 각 class의 비율 → 신청자 평균 (입력 기준이라 두 조건에 같다). **LLM**: `main_reasons` 4개를 규칙 기반 매핑한 변수(중복·매핑 실패 제외) 중 각 class의 비율 → 신청자 평균, 조건별. "
          "**변경 가능 = 단기변경 + 장기변경.** 분모는 신청자별로 남은 변수 수이다 (민감도 (나)에서만 4보다 작아질 수 있다).",
          f"- 단위: 신청자 300명 (두 조건 모두 ok). 분류표 분포: " + ", ".join(f"{c} {int((act['class'] == c).sum())}" for c in CLASSES) + ".", ""]
    csv_rows = []
    for title, cmap, drop in scenarios:
        mats = {}
        mats["SHAP 상위 4"] = [class_shares(shap_top[sid], cmap, drop) for sid in ids]
        for cond in "AB":
            mats[f"LLM 조건 {cond}"] = [class_shares(llm[(sid, cond)], cmap, drop) for sid in ids]
        # 쌍 완전: 세 종류 모두 분모가 있는 신청자만
        keep = [i for i in range(len(ids)) if all(mats[k][i] is not None for k in mats)]
        M = {k: np.vstack([mats[k][i] for i in keep]) for k in mats}
        n = len(keep)
        md += [f"## {title}", "", f"신청자 {n}명 (분모 0인 신청자 {len(ids) - n}명 제외).", "",
               "| 대상 | 단기변경 | 장기변경 | 변경불가 | 해당없음·간접 | 변경 가능 (단기+장기) [95% CI] |", "| --- | --- | --- | --- | --- | --- |"]
        for k, m in M.items():
            mean = m.mean(0)
            can = m[:, :2].sum(1)
            lo, hi = np.percentile(can[rng.integers(0, n, (B_BOOT, n))].mean(1), [2.5, 97.5])
            md.append(f"| {k} | {mean[0]:.4f} | {mean[1]:.4f} | {mean[2]:.4f} | {mean[3]:.4f} | {can.mean():.4f} [{lo:.4f}, {hi:.4f}] |")
            csv_rows.append({"scenario": title, "target": k, "n": n, **{c: mean[i] for i, c in enumerate(CLASSES)}, "changeable": can.mean(), "changeable_lo": lo, "changeable_hi": hi})
        d = M["LLM 조건 B"] - M["LLM 조건 A"]
        dl, dh = boot_ci(d, rng)
        dc = d[:, :2].sum(1)
        cl, ch = np.percentile(dc[rng.integers(0, n, (B_BOOT, n))].mean(1), [2.5, 97.5])
        md += ["", "조건 간 차이 (B − A, 신청자 쌍 부트스트랩 95% CI):", "", "| class | 차이 [95% CI] |", "| --- | --- |"]
        for i, c in enumerate(CLASSES):
            md.append(f"| {c} | {d[:, i].mean():+.4f} [{dl[i]:+.4f}, {dh[i]:+.4f}] |")
        md.append(f"| 변경 가능 (단기+장기) | {dc.mean():+.4f} [{cl:+.4f}, {ch:+.4f}] |")
        md += ["", "SHAP 상위 4와의 차이 (LLM − SHAP, 변경 가능 비율):", ""]
        for cond in "AB":
            dd = (M[f"LLM 조건 {cond}"][:, :2].sum(1) - M["SHAP 상위 4"][:, :2].sum(1))
            l_, h_ = np.percentile(dd[rng.integers(0, n, (B_BOOT, n))].mean(1), [2.5, 97.5])
            md.append(f"- 조건 {cond}: {dd.mean():+.4f} [{l_:+.4f}, {h_:+.4f}]")
        md.append("")
    (OUT / "exploratory_actionability.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(OUT / "exploratory_actionability.csv", index=False, encoding="utf-8-sig")
    print("\n".join(md))


if __name__ == "__main__":
    main()
