"""L4-2 순열 기준선(분석 계획 2.4절)과 완전성 기술 보고. 원문 응답은 집계에만 쓰고 출력 파일에 사유 원문을 싣지 않는다.

기준선 두 가지 (둘 다 기술통계, BH 묶음 밖):
 (1) 계획 2.4절의 순열: 같은 조건 안에서 LLM 응답(변수·방향)을 신청자 사이에서 무작위로 섞어 SHAP과 짝을 바꾼다 (B=1,000, 시드 42).
 (2) 사용자 지정 영가설: LLM이 변수 목록 100개에서 무작위로 4개를 고른다 (방향은 부도 위험 증가/감소 중 무작위). B=1,000, 시드 42.
실행: python -m src.l4_baseline
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .l4_analysis import DIRS, UP, DOWN, load_mapper, variables_of
from .l4_run import CALLS, INPUT, OUT, RAW, SEED, jl

B_PERM = 1000


def load_matrices():
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    vs = [f["variable"] for f in next(iter(recs.values()))["features"]]
    vi = {v: i for i, v in enumerate(vs)}
    mp = load_mapper(variables_of(next(iter(recs.values()))))
    calls = {(c["sample_id"], c["condition"]): c for c in jl(CALLS) if c["status"] == "ok"}
    ids = sorted(recs)
    S_mask = np.zeros((len(ids), len(vs)), bool)
    S_dir = np.zeros((len(ids), len(vs)), int)
    for n, sid in enumerate(ids):
        for f in recs[sid]["features"]:
            if f["shap_rank"] <= 4:
                S_mask[n, vi[f["variable"]]] = True
                S_dir[n, vi[f["variable"]]] = 1 if f["direction"] == UP else (-1 if f["direction"] == DOWN else 0)
    L = {}
    for cond in "AB":
        m = np.zeros((len(ids), len(vs)), bool)
        d = np.zeros((len(ids), len(vs)), int)
        for n, sid in enumerate(ids):
            parsed = json.loads(calls[(sid, cond)]["response_text"])
            seen = []
            for x in parsed["main_reasons"][:4]:
                v, _ = mp.map(x.get("variable", ""))
                if v is None or v in seen:
                    continue
                seen.append(v)
                m[n, vi[v]] = True
                d[n, vi[v]] = 1 if x.get("direction") == UP else (-1 if x.get("direction") == DOWN else 0)
        L[cond] = (m, d)
    return ids, vs, S_mask, S_dir, L


def f_values(Lm, Ld, S_mask, S_dir):
    hit = Lm & S_mask
    F = hit.sum(1) / 4
    Fd = (hit & (Ld == S_dir) & (S_dir != 0)).sum(1) / 4
    return F, Fd


def main():
    ids, vs, S_mask, S_dir, L = load_matrices()
    n, nv = S_mask.shape
    rng = np.random.default_rng(SEED)
    res = {}
    for cond in "AB":
        Lm, Ld = L[cond]
        Fo, Fdo = f_values(Lm, Ld, S_mask, S_dir)
        nullF, nullFd = np.empty(B_PERM), np.empty(B_PERM)
        for b in range(B_PERM):
            p = rng.permutation(n)
            F, Fd = f_values(Lm[p], Ld[p], S_mask, S_dir)
            nullF[b], nullFd[b] = F.mean(), Fd.mean()
        res[(cond, "perm")] = (Fo.mean(), Fdo.mean(), nullF, nullFd)
    # (2) 목록에서 무작위 4개
    rF, rFd = np.empty(B_PERM), np.empty(B_PERM)
    for b in range(B_PERM):
        m = np.zeros((n, nv), bool)
        d = np.zeros((n, nv), int)
        for i in range(n):
            pick = rng.choice(nv, 4, replace=False)
            m[i, pick] = True
            d[i, pick] = rng.choice([1, -1], 4)
        F, Fd = f_values(m, d, S_mask, S_dir)
        rF[b], rFd[b] = F.mean(), Fd.mean()
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    pv = lambda obs, null: (1 + int((null >= obs).sum())) / (B_PERM + 1)
    md = ["# L4-2 우연 기준선 (순열)", "",
          "> **기술통계, BH 묶음 밖.** 확증 가설이 아니며 p값은 참고용이다 (분석 계획 2.4절). 효과 크기와 구간 중심으로 읽는다.", "",
          f"- 분석 단위: 신청자 {n}명 × 조건, 입력은 두 조건 모두 ok인 300쌍. 시드 {SEED}, 반복 B={B_PERM}. 출력에는 응답 원문을 싣지 않았다 (변수 이름·방향을 집계에만 사용).",
          "- F = |LLM이 든 4개 변수 ∩ SHAP 상위 4개| / 4, F_dir = 그중 방향이 SHAP 방향과 같은 변수 수 / 4 (입력의 \"영향 거의 없음\"은 일치로 세지 않음). 둘 다 신청자 평균이다.", "",
          "## 1. 계획 2.4절의 기준선: LLM 응답을 신청자 사이에서 섞음 (같은 조건 안)", "",
          "응답(변수·방향 묶음)을 다른 신청자의 SHAP과 짝지어 F의 귀무분포를 만든다. 모든 신청자에게 흔한 변수를 말하는 경향이 우연 수준에 들어간다.", "",
          "| 지표 | 조건 | 관측 평균 | 우연 평균 | 관측 − 우연 | 우연 분포 95% 구간 | 순열 p (참고) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for name, ix in (("F", 0), ("F_dir", 1)):
        for cond in "AB":
            Fo, Fdo, nF, nFd = res[(cond, "perm")]
            obs = (Fo, Fdo)[ix]
            null = (nF, nFd)[ix]
            lo, hi = ci(null)
            md.append(f"| {name} | {cond} | {obs:.4f} | {null.mean():.4f} | {obs - null.mean():+.4f} | [{lo:.4f}, {hi:.4f}] | {pv(obs, null):.4f} |")
            rows.append({"baseline": "perm_shuffle", "metric": name, "condition": cond, "observed": obs, "null_mean": null.mean(), "diff": obs - null.mean(), "null_lo": lo, "null_hi": hi, "p_ref": pv(obs, null)})
    md += ["", "## 2. 영가설: LLM이 변수 목록 100개에서 무작위로 4개를 고른다", "",
           "신청자마다 100개 변수에서 무작위로 4개를 비복원으로 고르고, 방향은 \"부도 위험 증가\"/\"감소\" 중 무작위(같은 확률)로 붙인다. 조건과 무관한 한 가지 기준선이다.", "",
           "| 지표 | 조건 | 관측 평균 | 무작위 4개 기대 평균 | 관측 − 기대 | 무작위 분포 95% 구간 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, ix, null in (("F", 0, rF), ("F_dir", 1, rFd)):
        for cond in "AB":
            obs = res[(cond, "perm")][ix]
            lo, hi = ci(null)
            md.append(f"| {name} | {cond} | {obs:.4f} | {null.mean():.4f} | {obs - null.mean():+.4f} | [{lo:.4f}, {hi:.4f}] |")
            rows.append({"baseline": "random4_of_100", "metric": name, "condition": cond, "observed": obs, "null_mean": null.mean(), "diff": obs - null.mean(), "null_lo": lo, "null_hi": hi, "p_ref": np.nan})
    md += ["", "- 참고: 100개에서 무작위 4개를 고를 때 F의 이론 기대값은 4/100 = 0.04이다 (SHAP 상위 4개와 겹칠 확률). 방향 확률 0.5를 곱한 F_dir의 기대값은 약 0.02이다 (방향이 \"영향 거의 없음\"인 변수는 일치로 세지 않아 더 작을 수 있다).",
           "- 해석 문장은 쓰지 않았다. 위 표는 관측 평균과 두 가지 우연 기준선의 숫자이다."]
    (OUT / "baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "baseline.csv", index=False, encoding="utf-8-sig")
    print("\n".join(md[8:30]))


if __name__ == "__main__":
    main()
