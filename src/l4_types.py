"""L4 탐색: 불일치 유형 4종(누락형, 순위 역전형, 환각형, 편향 전이형)과 PD 5분위별 F(A). 정의는 docs/analysis_plan.md 3절 (계산 전에 커밋).

탐색적, 확증 판정에 사용하지 않음. 규칙 기반 분류이며 사람 검토가 아니다.
실행: python -m src.l4_types
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from .l4_analysis import extract_factors, load_keywords, load_mapper, mention_flags, variables_of
from .l4_run import B_BOOT, B_LIST, CALLS, INPUT, OUT, RAW, SEED, STRUCT, boot_ci_mean, cp_ci, jl


def classify(parsed: dict, rec: dict, mp, kw) -> dict:
    feats = {f["variable"]: f["shap_rank"] for f in rec["features"]}
    S = [v for v, r in sorted(feats.items(), key=lambda t: t[1]) if r <= 4]
    llm = []                                       # (변수, LLM rank) 중복·매핑 실패 제외
    for i, x in enumerate(parsed["main_reasons"][:4]):
        v, _ = mp.map(x.get("variable", ""))
        if v is not None and v not in [a for a, _ in llm]:
            llm.append((v, x.get("rank", i + 1)))
    L = [v for v, _ in llm]
    missing = [v for v in S if v not in L]
    both = [(v, r) for v, r in llm if v in S]
    inv = 0
    for (v1, r1), (v2, r2) in combinations(both, 2):
        if (feats[v1] - feats[v2]) * (r1 - r2) < 0:
            inv += 1
    fac = extract_factors(parsed)
    n_unmapped = sum(1 for f in fac if mp.map(f["text"])[0] is None)
    b_outside = [v for v in L if v in B_LIST and v not in S]
    mf = mention_flags(parsed, kw)
    kw_struct = any(h[0].startswith(STRUCT) for h in mf["hits_core"]) or any(h[0].startswith(STRUCT) for h in mf["hits_ext"])
    t1, t2, t3, t4 = len(missing) >= 1, inv >= 1, n_unmapped >= 1, bool(b_outside) or kw_struct
    return {"omission": t1, "n_missing": len(missing), "missing_vars": ";".join(missing), "inversion": t2, "n_inv_pairs": inv, "n_overlap": len(both), "hallucination": t3, "n_unmapped": n_unmapped,
            "bias_transfer": t4, "bias_b_outside": bool(b_outside), "bias_keyword": bool(kw_struct), "none": not (t1 or t2 or t3 or t4)}


def main():
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    calls = [c for c in jl(CALLS) if c["status"] == "ok"]
    mp, kw = load_mapper(variables_of(next(iter(recs.values())))), load_keywords()
    rows = []
    for c in calls:
        rec = recs[c["sample_id"]]
        rows.append({"sample_id": c["sample_id"], "condition": c["condition"], "pd": rec["model_output"]["pd"], "pd_bin": c["pd_bin"],
                     **classify(json.loads(c["response_text"]), rec, mp, kw)})
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "mismatch_types_per_row.csv", index=False, encoding="utf-8-sig")
    # ---------------- 유형별 빈도
    names = [("omission", "누락형 (SHAP 상위 4개 중 LLM이 안 든 변수 ≥ 1)"), ("inversion", "순위 역전형 (겹친 변수 ≥ 2이고 순서가 반대인 쌍 ≥ 1)"), ("hallucination", "환각형 (매핑 실패 요인 ≥ 1)"),
             ("bias_transfer", "편향 전이형 (SHAP 상위 4 밖의 B 목록 변수 사유, 또는 보호변수 키워드 플래그)"), ("none", "해당 없음 (4유형 모두 아님)")]
    md = ["# L4 탐색: 규칙 기반 불일치 유형 (조건별 빈도) 와 PD 5분위별 F(A)", "",
          "> **탐색적, 확증 판정에 사용하지 않음.** 유형은 규칙 기반 분류이며 사람 검토가 아니다. 보정·가설 검정 없음, 비율과 Clopper-Pearson 95% 구간만 적는다.", "",
          "- 정의는 `docs/analysis_plan.md` 3절 (계산 전에 커밋 `e5761ab`). S = SHAP 상위 4개 변수, L = LLM `main_reasons` 4개를 매핑한 변수. 한 응답이 여러 유형에 속할 수 있다 (배타적이지 않음).",
          f"- 단위: 신청자 × 조건 응답. 둘 다 ok라 A {int((df.condition == 'A').sum())}건, B {int((df.condition == 'B').sum())}건.", "",
          "## 1. 유형별 빈도 (조건별)", "", "| 유형 | A 건수 (비율) [95% CI] | B 건수 (비율) [95% CI] |", "| --- | --- | --- |"]
    out_rows = []
    for col, label in names:
        cells = []
        for c in "AB":
            d = df[df.condition == c]
            k, n = int(d[col].sum()), len(d)
            lo, hi = cp_ci(k, n)
            cells.append(f"{k} ({k / n:.1%}) [{lo:.1%}, {hi:.1%}]")
            out_rows.append({"type": col, "condition": c, "count": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
        md.append(f"| {label} | {cells[0]} | {cells[1]} |")
    pd.DataFrame(out_rows).to_csv(OUT / "exploratory_types.csv", index=False, encoding="utf-8-sig")
    # 유형 수 분포
    df["n_types"] = df[["omission", "inversion", "hallucination", "bias_transfer"]].sum(axis=1)
    md += ["", "### 한 응답에 해당하는 유형의 수", "", "| 유형 수 | A | B |", "| --- | --- | --- |"]
    for k in range(0, 5):
        md.append(f"| {k} | {int(((df.condition == 'A') & (df.n_types == k)).sum())} | {int(((df.condition == 'B') & (df.n_types == k)).sum())} |")
    # 세부
    md += ["", "## 2. 세부", ""]
    for c in "AB":
        d = df[df.condition == c]
        elig = int((d.n_overlap >= 2).sum())
        md.append(f"- **조건 {c}**: 누락 변수 수 평균 {d.n_missing.mean():.3f} (분포 " + ", ".join(f"{k}개 {int((d.n_missing == k).sum())}" for k in range(5)) +
                  f"); 순위 역전 판정 대상(겹친 변수 ≥ 2) {elig}건 중 역전 있음 {int(d.inversion.sum())}건 (역전 쌍 합계 {int(d.n_inv_pairs.sum())}); "
                  f"편향 전이형 {int(d.bias_transfer.sum())}건 = B 목록 변수가 S 밖 {int(d.bias_b_outside.sum())}건 + 키워드 플래그 {int(d.bias_keyword.sum())}건 (겹침 포함).")
    md += ["", "가장 자주 누락된 SHAP 상위 4 변수 (조건별 상위 5개, 응답 건수):", ""]
    for c in "AB":
        s = pd.Series(";".join(df[df.condition == c].missing_vars.fillna("")).split(";")).replace("", np.nan).dropna().value_counts().head(5)
        md.append(f"- {c}: " + (", ".join(f"`{k}` {v}" for k, v in s.items()) or "없음"))
    # ---------------- PD 5분위별 F(A)
    base = pd.read_csv(RAW / "per_row_metrics.csv", encoding="utf-8-sig")
    a = base[(base.condition == "A") & (base.status == "ok")][["sample_id", "F", "pd_bin"]].merge(df[df.condition == "A"][["sample_id", "pd"]], on="sample_id")
    a["quintile"] = pd.qcut(a.pd, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rng = np.random.default_rng(SEED)
    q_rows = []
    for q, d in a.groupby("quintile"):
        lo, hi = boot_ci_mean(d.F.to_numpy(), rng)
        q_rows.append({"pd_quintile": q, "n": len(d), "pd_min": d.pd.min(), "pd_max": d.pd.max(), "n_경계": int((d.pd_bin == "경계").sum()), "F_A_mean": d.F.mean(), "F_A_ci_lo": lo, "F_A_ci_hi": hi})
    qd = pd.DataFrame(q_rows)
    qd.to_csv(OUT / "exploratory_pd_bands.csv", index=False, encoding="utf-8-sig")
    md += ["", "## 3. PD 5분위별 F(A)", "",
           "표본 300명의 모형 PD를 5분위로 나눈 것이다. 표본이 경계 구간을 과표집(경계 150, 저 75, 고 75)해서 모집단의 5분위가 아니다. 부트스트랩 B=10,000, 시드 42, percentile 95% 구간. 조건 A의 확증 F(변수 단위, 분모 4).", "",
           "| 5분위 (1=PD 낮음) | n | PD 범위 | 경계 구간 인원 | F(A) 평균 | 95% CI |", "| --- | --- | --- | --- | --- | --- |"]
    for r in qd.itertuples():
        md.append(f"| {r.pd_quintile} | {r.n} | {r.pd_min:.4f} ~ {r.pd_max:.4f} | {r.n_경계} | {r.F_A_mean:.4f} | [{r.F_A_ci_lo:.4f}, {r.F_A_ci_hi:.4f}] |")
    (OUT / "exploratory_types.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
