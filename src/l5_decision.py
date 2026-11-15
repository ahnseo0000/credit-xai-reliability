"""L5 최소판: 확률을 승인·거절로 바꿀 때의 기대손익과, 설명이 약한 경계 구간(q70~q90)이 의사결정에서 차지하는 비중.

탐색적이며 실제 수익이 아니다. 아래 파라미터는 모두 외부 가정이다. 입력은 v2 동결 모형의 valid 예측 확률(test는 쓰지 않는다).
실행: python -m src.l5_decision
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as C

# =============================== 외부 가정 (데이터에서 구한 값이 아님, 실제 수익이 아님) ===============================
RATES = (0.10, 0.15, 0.20)     # 외부 가정: 금리 r
LGDS = (0.45, 0.60, 0.75)      # 외부 가정: 부도시손실률 LGD
COSTS = (0.02, 0.04)           # 외부 가정: 비용률 c
LOAN = 1.0                     # 외부 가정: 대출액 A = 1로 정규화
# ==================================================================================================================
TEMP_CUTOFF = 0.119124         # 사전등록 임시 거절선 (test PD q80, 고정값, test 데이터를 다시 읽지 않음). pd >= 거절선이면 거절
Q70, Q90 = 0.082282, 0.194800  # 경계 구간 q70~q90 (표본 설계의 고정값, results/l2/sample_300_check.md). 경계 = Q70 <= pd < Q90
OUT = C.RESULTS_DIR / "l5"


def expected_profit(pd_, r, lgd, c, a=LOAN):
    """승인했을 때 대출 1건의 기대손익 = (1−PD)·r·A − PD·LGD·A − c·A."""
    return (1 - pd_) * r * a - pd_ * lgd * a - c * a


def realized_profit(y, r, lgd, c, a=LOAN):
    """실제 부도 여부(y)로 계산한 승인 시 손익 = (1−y)·r·A − y·LGD·A − c·A."""
    return (1 - y) * r * a - y * lgd * a - c * a


def breakeven_cutoff(r, lgd, c):
    """기대손익 > 0 ⇔ PD < (r − c) / (r + LGD). 손익분기 금리 (PD·LGD + c)/(1 − PD) < r 와 같은 조건."""
    return (r - c) / (r + lgd)


def main():
    pdv = np.load(C.RESULTS_DIR / "l0_v2" / "raw" / "valid_p_K100.npy").astype(float)     # v2 동결 모형의 valid 예측 (재현 확인: results/phase2/ext_excluded)
    y = np.load(C.RESULTS_DIR / "l2" / "raw" / "y_valid.npy").astype(float)
    n = len(pdv)
    assert len(y) == n
    in_band = (pdv >= Q70) & (pdv < Q90)
    order = np.argsort(pdv)
    ps, ys = pdv[order], y[order]
    rows = []
    for r, lgd, c in itertools.product(RATES, LGDS, COSTS):
        e = expected_profit(pdv, r, lgd, c)
        rl = realized_profit(y, r, lgd, c)
        cut = breakeven_cutoff(r, lgd, c)
        row = {"r": r, "LGD": lgd, "c": c, "optimal_cutoff_pd": cut, "temp_cutoff_pd": TEMP_CUTOFF}
        for tag, k in (("opt", cut), ("temp", TEMP_CUTOFF)):
            ap = pdv < k
            row[f"approve_rate_{tag}"] = ap.mean()
            row[f"exp_profit_total_{tag}"] = e[ap].sum()
            row[f"exp_profit_per_applicant_{tag}"] = e[ap].sum() / n
            row[f"real_profit_total_{tag}"] = rl[ap].sum()
            row[f"band_exp_profit_{tag}"] = e[ap & in_band].sum()
            row[f"band_share_of_exp_profit_{tag}"] = e[ap & in_band].sum() / e[ap].sum() if e[ap].sum() != 0 else np.nan
            row[f"band_share_of_real_profit_{tag}"] = rl[ap & in_band].sum() / rl[ap].sum() if rl[ap].sum() != 0 else np.nan
        row["opt_minus_temp_exp_total"] = row["exp_profit_total_opt"] - row["exp_profit_total_temp"]
        row["opt_minus_temp_real_total"] = row["real_profit_total_opt"] - row["real_profit_total_temp"]
        # 실제 부도 라벨로 본 손익 최대 컷오프 (valid 안에서, 참고)
        cum = np.concatenate([[0.0], np.cumsum(realized_profit(ys, r, lgd, c))])
        kbest = int(cum.argmax())
        row["real_opt_cutoff_pd"] = ps[kbest - 1] if kbest > 0 else 0.0
        row["real_opt_approve_rate"] = kbest / n
        row["real_opt_profit_total"] = cum[kbest]
        row["band_share_applicants"] = in_band.mean()
        row["band_approve_rate_opt"] = (pdv[in_band] < cut).mean()
        row["band_abs_stake_share"] = np.abs(e[in_band]).sum() / np.abs(e).sum()
        row["optimal_cutoff_in_band"] = bool(Q70 <= cut < Q90)
        row["n_valid"] = n
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "decision_minimal.csv", index=False, encoding="utf-8-sig")
    # ---------------- md
    md = ["# L5 최소판: 승인·거절 기대손익과 경계 구간 비중", "",
          "> **외부 가정, 실제 수익이 아님, 탐색적.** 금리·부도시손실률·비용률은 모두 외부 가정이고 데이터에서 구한 값이 아니다. 대출액은 1로 정규화했다. 해석 문장 없이 숫자와 표만 적는다.", "",
          "## 입력과 정의", "",
          f"- 확률: v2 동결 모형(K=100)의 **valid** 예측 PD, {n:,}건 (test 미사용). 실제 부도율 {y.mean():.4f}, 평균 PD {pdv.mean():.4f}. 보정 없음.",
          "- 기대손익(승인 시) = (1−PD)·r·A − PD·LGD·A − c·A, A=1. 거절하면 0. 승인 ⇔ 기대손익 > 0 ⇔ PD < (r − c)/(r + LGD) (손익분기 금리 (PD·LGD + c)/(1−PD) < r 과 같다). 이 PD를 '최적 컷오프'라 부른다.",
          "- 실현 손익 = (1−y)·r·A − y·LGD·A − c·A (y = valid의 실제 부도 여부). 같은 외부 가정으로 계산한 참고값이다.",
          f"- 임시 컷오프: {TEMP_CUTOFF} (사전등록, test PD q80의 고정값). pd ≥ 컷오프면 거절. valid에서 승인율 {(pdv < TEMP_CUTOFF).mean():.2%}.",
          f"- 경계 구간: {Q70} ≤ PD < {Q90} (q70~q90, 표본 설계의 고정값). valid에서 이 구간의 신청자 비율 **{in_band.mean():.2%}** ({int(in_band.sum()):,}명). 참고: valid 자체 분위수는 q70 {np.quantile(pdv, .7):.4f}, q80 {np.quantile(pdv, .8):.4f}, q90 {np.quantile(pdv, .9):.4f}.",
          f"- 파라미터: r ∈ {RATES}, LGD ∈ {LGDS}, c ∈ {COSTS} (18개 조합, 모두 외부 가정).", "",
          "## 1. 파라미터 조합별 최적 컷오프와 임시 컷오프의 비교 (valid 전체, 대출액 1 기준 총합)", "",
          "| r | LGD | c | 최적 컷오프 PD | 승인율(최적) | 기대 총손익(최적) | 기대 총손익(임시) | 차이(최적−임시) | 승인율(임시) | 실현 총손익(최적) | 실현 총손익(임시) | 실현 손익 최대 컷오프 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r_ in df.itertuples():
        md.append(f"| {r_.r:.2f} | {r_.LGD:.2f} | {r_.c:.2f} | {r_.optimal_cutoff_pd:.4f} | {r_.approve_rate_opt:.2%} | {r_.exp_profit_total_opt:,.1f} | {r_.exp_profit_total_temp:,.1f} | {r_.opt_minus_temp_exp_total:+,.1f} | {r_.approve_rate_temp:.2%} | {r_.real_profit_total_opt:,.1f} | {r_.real_profit_total_temp:,.1f} | {r_.real_opt_cutoff_pd:.4f} |")
    md += ["", "## 2. 경계 구간(q70~q90)의 비중 (valid)", "",
           f"경계 구간 신청자 비율 {in_band.mean():.2%} (모든 조합에 같다).", "",
           "| r | LGD | c | 최적 컷오프가 경계 안 | 경계 승인율(최적) | 경계 기대손익 / 총 기대손익 (최적 정책) | 경계 기대손익 / 총 기대손익 (임시 정책) | 경계 실현손익 / 총 실현손익 (최적 정책) | 경계의 기대손익 절댓값 비중 (전 신청자) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r_ in df.itertuples():
        md.append(f"| {r_.r:.2f} | {r_.LGD:.2f} | {r_.c:.2f} | {'예' if r_.optimal_cutoff_in_band else '아니오'} | {r_.band_approve_rate_opt:.2%} | {r_.band_share_of_exp_profit_opt:.2%} | {r_.band_share_of_exp_profit_temp:.2%} | {r_.band_share_of_real_profit_opt:.2%} | {r_.band_abs_stake_share:.2%} |")
    md += ["", "- 비중의 분모는 해당 정책으로 승인한 신청자의 기대(또는 실현) 손익 합이다. 마지막 열은 정책과 무관하게 모든 신청자의 승인 시 기대손익 절댓값 합 대비 경계 구간 몫이다.",
           "- 원자료 행(18개 조합 × 전체 지표)은 `results/l5/decision_minimal.csv`에 있다.", ""]
    (OUT / "decision_minimal.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[5:16]))
    print(df[["r", "LGD", "c", "optimal_cutoff_pd", "approve_rate_opt", "band_share_of_exp_profit_opt", "band_abs_stake_share"]].round(4).to_string())


if __name__ == "__main__":
    main()
