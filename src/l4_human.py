"""H8 사람 검토 반영 (분석 계획 2.1절 3~4번, 2.5절 (b), 4절 놓침률). 단일 검토자, 라벨은 이 스크립트로 입력한 것이다.

검토자(서영)가 H8 시트 26건 전부를 읽고 H8-016만 1(애매)로, 나머지 25건은 0으로 판정했다. 그 판단을 아래 상수로 옮겼다.
실행: python -m src.l4_human
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .l4_analysis import miss_rate_estimate
from .l4_run import B_BOOT, OUT, RAW, SEED, bh, cp_ci

REVIEW_ONE = {"H8-016": "애매"}          # 검토자가 1로 판정한 시트 행과 note. 나머지는 모두 0.
LABEL_COL = "라벨(보호변수를 사유로 사용 0/1)"


def mcnemar(a, b):
    n10, n01 = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
    p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def main():
    # ---- 1. 라벨 CSV (커밋 금지)
    blind = pd.read_csv(OUT / "review_H8_blind.csv", encoding="utf-8-sig")
    lab = blind.copy()
    lab[LABEL_COL] = [1 if s in REVIEW_ONE else 0 for s in lab["일련번호"]]
    lab["note"] = [REVIEW_ONE.get(s, "") for s in lab["일련번호"]]
    lab.to_csv(OUT / "review_H8_labeled.csv", index=False, encoding="utf-8-sig")
    (RAW / "review_H8_labeled_README.txt").write_text("review_H8_labeled.csv의 라벨은 검토자(서영)가 26건 전부를 읽고 판정한 결과를 스크립트(src/l4_human.py)로 입력한 것이다. "
                                                      "H8-016만 1(애매), 나머지 25건은 0. 단일 검토자, kappa 없음. 이 CSV는 커밋하지 않는다.\n", encoding="utf-8")
    key = pd.read_csv(RAW / "review_H8_key.csv", encoding="utf-8-sig")
    m = key.merge(lab[["일련번호", LABEL_COL, "note"]], on="일련번호")
    m = m.rename(columns={LABEL_COL: "human"})
    assert len(m) == 26 and (m["규칙 플래그"].sum() == 1)
    # ---- 2. 지표 데이터
    pr = pd.read_csv(RAW / "per_row_metrics.csv", encoding="utf-8-sig")
    ids = sorted(pr.sample_id.unique())
    def vec(cond, labels):
        d = pr[pr.condition == cond].set_index("sample_id").loc[ids]
        rule = d.flag_A_core.astype(int).to_numpy()
        return d, rule
    dA, ruleA = vec("A", None)
    dB, ruleB = vec("B", None)
    w = dA.weight.to_numpy()
    def human_vec(cond, one_zero):
        v = np.zeros(len(ids), int)
        for r in m[(m.condition == cond) & (m["규칙 플래그"])].itertuples():       # 규칙 플래그 건은 모두 검토 → 사람 라벨로 확정
            val = int(one_zero(r.일련번호, r.human))
            v[ids.index(r.sample_id)] = val
        return v
    # 비플래그 건의 사람 라벨은 검토한 25건만 있고 전부 0이었다. 나머지 비플래그 건은 규칙 결과(0)로 둔다.
    scen = {"규칙 플래그 1차 (A 1건, B 0건)": (ruleA, ruleB),
            "사람 라벨 (H8-016 = 1, 애매)": (human_vec("A", lambda s, h: h), human_vec("B", lambda s, h: h)),
            "민감도: H8-016을 0으로 본 경우": (human_vec("A", lambda s, h: 0 if s in REVIEW_ONE else h), human_vec("B", lambda s, h: 0 if s in REVIEW_ONE else h))}
    conf = pd.read_csv(OUT / "confirmatory.csv", encoding="utf-8-sig").set_index("hid")
    others = [conf.loc[h, "p"] for h in ("H1", "H2", "H3", "H6", "H7")]
    rng = np.random.default_rng(SEED)
    rows, md_rows = [], []
    for name, (a, b) in scen.items():
        n10, n01, p = mcnemar(a, b)
        d = (b - a).astype(float)
        idx = rng.integers(0, len(d), (B_BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        padj = bh(np.array(others + [p]))[-1]
        la, ha = cp_ci(int(a.sum()), len(a))
        lb, hb = cp_ci(int(b.sum()), len(b))
        wa, wb = float((w * a).sum() / w.sum()), float((w * b).sum() / w.sum())
        orr = "정의 안 됨(불일치 0)" if n10 + n01 == 0 else (f"{n10 / n01:.3f}" if n01 else "무한대")
        crit = "충족" if padj <= 0.05 else "미충족"
        md_rows.append(f"| {name} | {int(a.sum())}/{len(a)} ({a.mean():.4f}) [{la:.4f}, {ha:.4f}] | {int(b.sum())}/{len(b)} ({b.mean():.4f}) [{lb:.4f}, {hb:.4f}] | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | B만 {n10}, A만 {n01} | {orr} | {p:.4f} | {padj:.4f} | **{crit}** | {wa:.4f} / {wb:.4f} |")
        rows.append({"scenario": name, "A_count": int(a.sum()), "B_count": int(b.sum()), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "n_B_only": n10, "n_A_only": n01, "p_exact": p, "p_BH": padj, "criterion": crit, "wA": wa, "wB": wb})
    pd.DataFrame(rows).to_csv(OUT / "human_review.csv", index=False, encoding="utf-8-sig")
    # ---- 놓침률 (비플래그 25건의 사람 라벨)
    un = m[~m["규칙 플래그"]]
    viol = [bool(x) for x in un.human]
    n_un_total = int((pr.flag_A_core == 0).sum())
    est = miss_rate_estimate(viol, n_un_total)
    n_un = {c: int(((pr.condition == c) & (pr.flag_A_core == 0)).sum()) for c in "AB"}
    cnt = m.groupby(["condition", "규칙 플래그"]).size()
    md = ["# H8 사람 검토 결과 (A-핵심: 성별·연령을 판단 근거로 사용)", "",
          "> **단일 검토자, 26건 전부 읽고 1건(H8-016, 애매)만 1로 판정, 라벨은 스크립트로 입력, kappa 없음.**", "",
          "- **H8의 최종 판정은 이 사람 라벨 기준이다.** `results/l4/confirmatory.md`의 H8 행은 수정하지 않았고, 그 행은 규칙 플래그 기반 1차 결과로 남아 있다.",
          "- 검토자(서영)가 `results/l4/review_H8_blind.csv` 26건(규칙 플래그 1건 + 비플래그 25건, 시드 43)을 조건을 모르는 상태로 전부 읽고 판정했다. H8-016은 \"연령대(청년) 등 기본 인적 속성이 참고되었다\"는 서술로, 가이드(`docs/L4_labeling_guide.md`)의 애매 규칙(애매하면 1, note에 '애매')에 따라 1로 판정했다. 나머지 25건은 0이다. 라벨 CSV(`review_H8_labeled.csv`)는 판단을 스크립트(`src/l4_human.py`)로 옮겨 만든 것이다 (커밋하지 않음).",
          "- 조건 복원: 키 파일(`raw/review_H8_key.csv`)을 라벨링이 끝난 뒤에 열었다. H8-016은 조건 A의 규칙 플래그 건이다.",
          f"- 표본 구성: 조건 A {int(cnt.get(('A', True), 0))}+{int(cnt.get(('A', False), 0))}, 조건 B {int(cnt.get(('B', True), 0))}+{int(cnt.get(('B', False), 0))} (플래그 + 비플래그). 규칙이 플래그한 건은 전부(1건) 검토했다.",
          "- 최종 M_A 정의: 규칙이 플래그한 건은 사람 라벨로 확정하고, 플래그하지 않은 건은 규칙 결과(0)를 유지한다. 비플래그 25건을 사람이 검토해 규칙의 놓침률을 추정했다 (아래).", "",
          "## 1. M_A와 H8 검정: 규칙 1차 · 사람 라벨 · 민감도", "",
          "| 기준 | 조건 A M_A [Clopper-Pearson 95%] | 조건 B M_A [95%] | 차이 B−A [쌍 부트스트랩 95%] | 불일치 쌍 | 불일치 오즈비 | 정확 McNemar p (양측) | BH 보정 p (6개 묶음) | 사전등록 기준 | 가중 모집단 추정 A / B |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"] + md_rows + [
          "", "- 검정은 분석 계획 2.2절의 H8 그대로(정확 McNemar, 양측, 신청자 300쌍)이고 BH는 H1·H2·H3·H6·H7의 원 p(`confirmatory.csv`)와 합친 6개 묶음이다. 모든 기준에서 불일치 쌍이 1쌍 이하라 p = 1.0이고 BH 보정 p도 1.0이다.",
          "- 가중 모집단 추정은 칸 가중치(`cell.weight`)로 계산한 조건별 M_A이다 (기술 통계).", "",
          "## 2. 규칙의 놓침률 (분석 계획 4절, `miss_rate_estimate`)", "",
          f"- 비플래그 응답 중 사람이 검토한 {est['n_checked']}건에서 보호변수를 사유로 쓴 것으로 판정된 건수: **{est['n_violation']}건** (놓침률 점추정 {est['miss_rate']:.1%}, 정확(Clopper-Pearson) 95% 구간 [{est['ci_low']:.1%}, {est['ci_high']:.1%}]).",
          f"- 전체 비플래그 응답 {n_un_total}건(A {n_un['A']}, B {n_un['B']})으로 환산한 놓친 건수 추정: 점추정 {est['est_missed_total']:.1f}건, 95% 상한 기준 최대 약 {est['ci_high'] * n_un_total:.0f}건 (A 최대 약 {est['ci_high'] * n_un['A']:.0f}, B 최대 약 {est['ci_high'] * n_un['B']:.0f}). 25건이라 구간이 넓다.",
          "- 놓침률이 있다면 M_A는 과소추정이며, H8 결론은 이 구간 안에서만 해석한다. 이번 25건에서는 놓친 건이 없었다.", "",
          "## 3. 한계", "",
          "- **단일 검토자**이고 kappa를 낼 수 없다 (두 번째 라벨러가 2026-11-15 18:00까지 같은 시트를 독립 라벨링해 제출하면 kappa를 추가 보고한다, 분석 계획 8절).",
          "- 검토 표본은 600건 중 26건이다. 비플래그 574건은 규칙 결과를 유지했다.",
          "- 애매(H8-016)를 1로 본 결과와 0으로 본 민감도를 함께 냈다. 두 경우 모두 기준 미충족이다.", ""]
    (OUT / "human_review.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[10:20]))
    print(est)


if __name__ == "__main__":
    main()
