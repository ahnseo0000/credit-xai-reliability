"""L2-13: test에서 300건 층화 추출 (docs/interface.md '300건 추출' 설계 v2 그대로).

PD 구간(저/경계/고) x 성별(M/F) x 연령(청년/중년/장년) = 18칸. 경계 6칸 25건, 저·고 12칸 12~13건.
q70/q90은 v2 동결 모형의 test pd 분위수 (interface.md 고정값과 일치해야 한다). 재추출하지 않는다.
산출: results/l2/sample_300.csv, sample_300_check.md, cell_weights.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from . import config as C
from .l2_common import FROZEN, L2RAW, L2, log

Q70_DOC, Q90_DOC = 0.082282, 0.194800


def main():
    pr = pd.read_csv(FROZEN / "raw" / "test_predictions.csv")
    gv = pd.read_csv(C.RESULTS_DIR / "l0" / "raw" / "group_vars.csv").set_index(C.ID_COL)
    t = pr.merge(gv[["group_gender", "group_age", "group_income"]], left_on=C.ID_COL, right_index=True, how="left")
    assert t["group_gender"].notna().all() and len(t) == 46127
    t["pd"] = t["pd_xgb_v2"]
    q70, q90 = np.quantile(t["pd"], [0.7, 0.9])
    assert abs(q70 - Q70_DOC) < 5e-7 and abs(q90 - Q90_DOC) < 5e-7, (q70, q90)
    t["pd_bin"] = np.where(t["pd"] >= q90, "고", np.where(t["pd"] >= q70, "경계", "저"))
    ok = t[t["group_gender"].isin(["M", "F"])].copy()
    ok["cell_id"] = ok["pd_bin"] + "_" + ok["group_gender"] + "_" + ok["group_age"]
    sizes = ok.groupby("cell_id").size()
    ref = pd.read_csv(FROZEN / "cell_sizes_18.csv", encoding="utf-8-sig").set_index("cell_id")["n"]
    assert (sizes.reindex(ref.index) == ref).all() and len(sizes) == 18, "18칸 크기가 interface.md와 다르다"

    cells_lo_hi = sorted(c for c in sizes.index if not c.startswith("경계"))
    assert len(cells_lo_hi) == 12
    big = set(np.array(cells_lo_hi)[np.random.default_rng(42).choice(12, size=6, replace=False)])
    alloc = {c: 25 if c.startswith("경계") else (13 if c in big else 12) for c in sizes.index}
    assert sum(alloc.values()) == 300
    parts, rows = [], []
    for c in sorted(sizes.index):
        g = ok[ok["cell_id"] == c].sort_values(C.ID_COL)
        n = min(alloc[c], len(g))
        parts.append(g.sample(n=n, random_state=42))
        rows.append({"cell_id": c, "칸 크기": len(g), "할당": alloc[c], "추출": n, "가중치": len(g) / n})
    cw = pd.DataFrame(rows)
    smp = pd.concat(parts).merge(cw[["cell_id", "가중치"]], on="cell_id")
    assert len(smp) == 300 and smp[C.ID_COL].is_unique
    vals = pd.read_pickle(L2RAW / "test_values.pkl").set_index(C.ID_COL)
    feat = vals.loc[smp[C.ID_COL]].add_prefix("feat__").reset_index(drop=True)
    out = pd.concat([smp.reset_index(drop=True).rename(columns={C.ID_COL: "sample_id", "pd": "pd", "가중치": "weight"})[
        ["sample_id", "cell_id", "pd_bin", "group_gender", "group_age", "group_income", "target", "pd", "weight"]], feat], axis=1)
    L2.mkdir(parents=True, exist_ok=True)
    out.to_csv(L2 / "sample_300.csv", index=False, encoding="utf-8-sig")
    cw.to_csv(L2 / "cell_weights.csv", index=False, encoding="utf-8-sig")

    # ---- 점검 (interface.md '추출 후 점검')
    md = ["# 300건 추출 점검", "", f"- 모형: v2 동결 (K=100). q70={q70:.6f}, q90={q90:.6f} (interface.md 고정값과 일치 확인). 시드 42, 재추출 안 함.",
          f"- 13건 칸 6개: {', '.join(sorted(big))}", "", "## 1. 칸별 건수·가중치 (가중치 = 칸 크기 / 추출 건수)", "",
          "| cell_id | 칸 크기 | 할당 | 추출 | 가중치 |", "| --- | --- | --- | --- | --- |"]
    md += [f"| `{r.cell_id}` | {r._2:,} | {r.할당} | {r.추출} | {r.가중치:.1f} |" for r in cw.itertuples()]
    md += ["", f"합계 추출 {len(out)}건 (할당표와 {'일치' if (cw.추출 == cw.할당).all() else '불일치'}). 가중치는 전체 모집단 추정에만 쓴다.", ""]
    inc_all = ok["group_income"].value_counts(normalize=True)
    md += ["## 2. 소득 분위 균형 (참고용)", "", f"test 전체(M/F) 소득 비율: 하 {inc_all['하']:.1%} / 중 {inc_all['중']:.1%} / 상 {inc_all['상']:.1%}", "",
           "| PD 구간 | n | 하 | 중 | 상 | 하 % | 중 % | 상 % |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in ["저", "경계", "고", "전체"]:
        g = out if b == "전체" else out[out["pd_bin"] == b]
        c = g["group_income"].value_counts().reindex(["하", "중", "상"]).fillna(0).astype(int)
        md.append(f"| {b} | {len(g)} | {c['하']} | {c['중']} | {c['상']} | {c['하']/len(g):.1%} | {c['중']/len(g):.1%} | {c['상']/len(g):.1%} |")
    cnt = out["group_income"].value_counts().reindex(["하", "중", "상"]).fillna(0)
    chi = chisquare(cnt.to_numpy(), f_exp=inc_all.reindex(["하", "중", "상"]).to_numpy() * 300)
    low = [k for k, v in cnt.items() if v < 60]
    md += ["", f"카이제곱 적합도(300건 vs test 전체 비율, 참고용): χ²={chi.statistic:.2f}, p={chi.pvalue:.3g}. (층화가 PD·성별·연령 기준이라 소득이 전체 비율과 다른 것은 예상된 일이다.)",
           f"쏠림 판정(어느 분위라도 60건 미만): **{'소득 쏠림: ' + ','.join(low) + ' 분위가 60건 미만' if low else '해당 없음'}**. 재추출하지 않는다.",
           "쏠림이 있으면 L4-9 교란 통제 로지스틱에 소득 분위를 공변량으로 넣고 한계에 적는다.", "",
           "## 3. 성별·연령 주변 분포와 실제 부도율 (PD 구간별)", "", "| PD 구간 | n | M % | F % | 청년 % | 중년 % | 장년 % | 실제 부도율(추출) | 실제 부도율(test 전체 구간) |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in ["저", "경계", "고"]:
        g = out[out["pd_bin"] == b]
        tb = ok[ok["pd_bin"] == b]
        md.append(f"| {b} | {len(g)} | {(g.group_gender=='M').mean():.1%} | {(g.group_gender=='F').mean():.1%} | {(g.group_age=='청년').mean():.1%} | "
                  f"{(g.group_age=='중년').mean():.1%} | {(g.group_age=='장년').mean():.1%} | {g.target.mean():.1%} | {tb.target.mean():.1%} |")
    (L2 / "sample_300_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"L2-13 300건 추출 완료: 칸별 건수 = 할당표, 소득 쏠림 {'있음 ' + ','.join(low) if low else '없음'} (χ² p={chi.pvalue:.3g}), 소득 하/중/상 = {int(cnt['하'])}/{int(cnt['중'])}/{int(cnt['상'])}")


if __name__ == "__main__":
    main()
