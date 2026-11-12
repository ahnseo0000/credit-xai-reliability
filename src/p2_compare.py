"""Phase 2-3/4: GBM(100변수) vs GBM(shortlist) vs 스코어카드(shortlist) 비교표 -> results/phase2/compare_valid.md (valid만, test 미사용).

변수 축소 비용과 모형 형태 비용을 나누기 위해 보조 모형 GBM(스코어카드 최종 변수와 같은 변수)을 다리로 쓴다.
실행: python -m src.p2_compare   (p2_scorecard, p2_gbm, p2_gbm --from-scorecard 를 먼저 실행)
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .metrics import bootstrap_auc, credit_metrics, decile_table
from .p2_woe import P2, Data


def log(msg: str):
    line = f"- `{datetime.now():%H:%M:%S}` {msg}"
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def main():
    y = Data("valid").y
    raw = P2 / "raw"
    sc = json.loads((P2 / "scorecard_model.json").read_text(encoding="utf-8"))
    sl = json.loads((P2 / "shortlist_vars.json").read_text(encoding="utf-8"))["final"]
    n_sc, n_sl = len(sc["variables"]), len(sl)
    P = {"GBM(100변수)": np.load(C.RESULTS_DIR / "l0_v2" / "raw" / "valid_p_K100.npy"),
         f"GBM(shortlist {n_sl}변수)": np.load(raw / "valid_p_shortlist.npy"),
         f"GBM(스코어카드와 같은 {n_sc}변수, 보조)": np.load(raw / "valid_p_scorecard_vars.npy"),
         f"스코어카드({n_sc}변수)": np.load(raw / "valid_p_scorecard.npy")}
    names = list(P)
    g100, g25, g22, scn = names
    M = {k: credit_metrics(y, p) for k, p in P.items()}
    score = np.load(raw / "valid_score.npy")
    tscore = np.load(raw / "valid_table_score.npy")
    ptr = np.load(raw / "train_p_scorecard.npy")
    m_tr = credit_metrics(Data("train").y, ptr)
    factor, offset = sc["factor"], sc["offset"]
    pd_of = lambda s: 1 / (1 + np.exp((np.asarray(s, float) - offset) / factor))
    assert np.allclose(pd_of(score), P[scn], atol=1e-8), "점수와 로지스틱 PD가 일치해야 한다"

    # ---- 쌍 부트스트랩 차이 (같은 재표본, B=1000, 시드 42)
    d_a = bootstrap_auc(y, {k: P[k] for k in (g100, g25, g22, scn)}, baseline=g100).set_index("model")      # 각 모형 − GBM(100)
    d_b = bootstrap_auc(y, {k: P[k] for k in (g25, g22)}, baseline=g25).set_index("model")                  # GBM(22) − GBM(25)
    d_c = bootstrap_auc(y, {k: P[k] for k in (g22, scn)}, baseline=scn).set_index("model")                  # GBM(22) − 스코어카드
    def ci_neg(tab, row):   # (GBM − 스코어카드) 구간을 (스코어카드 − GBM) 부호로 뒤집어 다른 행과 부호를 맞춘다
        r = tab.loc[row]
        return f"[{-r['diff_ci_high']:+.4f}, {-r['diff_ci_low']:+.4f}]"

    auc = {k: M[k]["AUC"] for k in names}
    red1, red2, form, total = auc[g100] - auc[g25], auc[g25] - auc[g22], auc[g22] - auc[scn], auc[g100] - auc[scn]

    def ci(tab, row, col):
        r = tab.loc[row]
        return f"[{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]"

    md = ["# Phase 2 비교: GBM(100변수) vs GBM(shortlist) vs WoE 스코어카드 (valid)", "",
          f"- valid {len(y):,}건, 부도율 {y.mean():.4f}. **test는 읽지 않았다.** 변수 선택(IV, family 상관, train gain 우선순위)과 구간화·WoE·계수는 train에서만 적합했고 valid는 평가에만 썼다 (스코어카드의 C는 train 5-fold CV).",
          "- 이전 버전(우선순위에 L2 부트스트랩 안정성, 즉 test SHAP 기반 값을 썼던 25개)은 `*_v1` 파일로 보존했다. 이 문서는 test 정보를 배제한 새 버전이다.",
          f"- GBM은 모두 v2와 같은 설정이다 (XGBoost, lr 0.03, depth 5, min_child_weight 20, subsample 0.8, colsample 0.7, reg_lambda 5, 조기 종료 150, 시드 42, scale_pos_weight 없음). 조기 종료에 valid를 썼으므로 GBM은 valid에 약간 유리하다. 연령·성별 제외, 가중치 없음.",
          f"- shortlist {n_sl}개 중 부호가 거꾸로인 계수를 반복 제거해(BUREAU_DEBT_SUM, INS_LATE_L12M_PAY_RATIO_MEAN, BUR_ALL_DEBT_RATIO_MEAN) 스코어카드는 {n_sc}개 변수가 되었다. 보조 모형 GBM({n_sc}변수)은 이 {n_sc}개로 학습한 GBM이다.", "",
          "## 1. 지표 비교 (valid)", "",
          "| 모형 | 변수 | AUC | Gini | KS | 1분위 lift | 상위30% 포착 | Brier | ECE(10분위) | 평균 예측 PD |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    nv = {g100: 100, g25: n_sl, g22: n_sc, scn: n_sc}
    for k in names:
        m = M[k]
        md.append(f"| {k} | {nv[k]} | {m['AUC']:.4f} | {m['Gini']:.4f} | {m['KS']:.4f} | {m['top_decile_lift']:.2f} | {m['top30_capture']:.1%} | {m['Brier']:.4f} | {m['ECE_10q']:.4f} | {m['mean_pred']:.4f} |")
    md.append(f"| (참고) 스코어카드, train | {n_sc} | {m_tr['AUC']:.4f} | {m_tr['Gini']:.4f} | {m_tr['KS']:.4f} | {m_tr['top_decile_lift']:.2f} | {m_tr['top30_capture']:.1%} | {m_tr['Brier']:.4f} | {m_tr['ECE_10q']:.4f} | {m_tr['mean_pred']:.4f} |")
    md += ["", "## 2. 변수 축소 비용과 모형 형태 비용 (AUC, valid, 95% 구간은 같은 재표본 쌍 부트스트랩 B=1000)", "",
           "| 구간 | 무엇이 바뀌나 | AUC 차이 (뒤 − 앞) | 95% 구간 |", "| --- | --- | --- | --- |",
           f"| ① 변수 축소 100 → {n_sl} | 같은 GBM, 입력 변수만 줄임 | −{red1:.4f} | {ci(d_a, g25, 'a')} |",
           f"| ② 부호 규칙 제거 {n_sl} → {n_sc} | 같은 GBM, 변수 {n_sl - n_sc}개 더 제거 | −{red2:.4f} | {ci(d_b, g22, 'b')} |",
           f"| ③ 모형 형태 GBM → 스코어카드 | 같은 {n_sc}개 변수, 모형만 바꿈 (GBM → WoE 구간 + 로지스틱) | −{form:.4f} | {ci_neg(d_c, g22)} |",
           f"| 합계 GBM(100) → 스코어카드 | ① + ② + ③ | −{total:.4f} | {ci(d_a, scn, 'a')} |", "",
           f"- **변수 축소 비용** (① + ②, 100 → {n_sc}개): AUC {red1 + red2:.4f} ({(red1 + red2) / total:.0%}). **모형 형태 비용** (③): AUC {form:.4f} ({form / total:.0%}). 합계 {total:.4f}.",
           f"- 읽는 법: ①②는 정보를 줄인 대가이고 같은 GBM으로 측정했다. ③은 정보를 같게 둔 채 비선형·상호작용을 구간별 가산 점수로 단순화한 대가이다. 보조 모형 GBM({n_sc}변수)이 두 비용을 가르는 다리이다.",
           f"- KS 기준으로는 GBM(100) {M[g100]['KS']:.4f} → GBM({n_sl}) {M[g25]['KS']:.4f} → GBM({n_sc}) {M[g22]['KS']:.4f} → 스코어카드 {M[scn]['KS']:.4f}. 교정(ECE)은 스코어카드 {M[scn]['ECE_10q']:.4f}, GBM(100) {M[g100]['ECE_10q']:.4f}로 모두 작다.", ""]
    dec = {k: decile_table(y, P[k]) for k in (g100, g25, scn)}
    md += ["## 3. 십분위 (valid, 1 = 최고 위험, 각 모형의 자기 예측 순위 기준)", "", f"| 십분위 | GBM(100) 부도율 | lift | GBM({n_sl}) 부도율 | lift | 스코어카드 부도율 | lift |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for i in range(10):
        md.append(f"| {i + 1} | {dec[g100].loc[i, 'bad_rate']:.4f} | {dec[g100].loc[i, 'lift']:.2f} | {dec[g25].loc[i, 'bad_rate']:.4f} | {dec[g25].loc[i, 'lift']:.2f} | {dec[scn].loc[i, 'bad_rate']:.4f} | {dec[scn].loc[i, 'lift']:.2f} |")

    # ---- 점수 척도 주의 + 점수 구간표
    mean_pd, base_rate = P[scn].mean(), y.mean()
    s_base = offset - factor * np.log(base_rate / (1 - base_rate))
    s_pd600 = float(pd_of(600))
    err = np.abs(pd_of(tscore) - P[scn])
    rel = err / P[scn]
    md += ["", "## 4. 점수 척도에 관한 주의", "",
           "- **점수↔PD 변환은 정확하다.** 점수 = Offset − Factor × logit(PD), 즉 PD = 1 / (1 + exp((점수 − Offset) / Factor)). 여기서 **PD는 로지스틱 출력(predict_proba)**이고 점수는 그 로짓의 선형 변환일 뿐이라, 점수와 PD는 서로 정확히 오갈 수 있다 (valid 46,127건에서 두 값이 일치함을 확인했다).",
           f"- **600점, odds 50:1, PDO 20점은 업계 관행값을 쓴 가정이다.** 데이터에서 구한 값이 아니다. 이 척도에서 600점은 모형 PD {s_pd600:.2%}에 해당할 뿐, 600점이 '기준 고객'이거나 모집단 부도율 2%라는 뜻이 아니다. 모집단 부도율 {base_rate:.2%}는 약 {s_base:.0f}점에 해당한다.",
           f"- **valid 평균 점수는 {score.mean():.1f}점(약 568점)**이고 5/50/95 백분위는 {np.percentile(score, 5):.0f} / {np.percentile(score, 50):.0f} / {np.percentile(score, 95):.0f}점이다. 점수는 높을수록 안전하다.",
           f"- 정수 점수 반올림 오차: 점수표 정수 합으로 PD를 환산하면 정확한 PD와의 차이가 평균 {err.mean():.5f}, 최대 {err.max():.5f} (상대 오차 평균 {rel.mean():.1%}, 최대 {rel.max():.1%}). 정수 점수의 합과 정확한 점수는 평균 {np.abs(tscore - score).mean():.2f}점, 최대 {np.abs(tscore - score).max():.1f}점 차이이고 1점은 PD를 약 {100 * (1 - np.exp(-1 / factor)):.1f}% (상대) 바꾼다.", "",
           "PD ↔ 점수 대응 (이 척도 가정에서):", "", "| 모형 PD | 점수 |", "| --- | --- |"]
    for q in (0.01, 0.02, 0.05, base_rate, 0.15, 0.30):
        md.append(f"| {q:.2%} | {offset - factor * np.log(q / (1 - q)):.0f} |")
    d = pd.DataFrame({"score": score, "pd": P[scn], "y": y})
    d["dec"] = pd.qcut(d.score.rank(method="first"), 10, labels=False)
    t = d.groupby("dec").agg(n=("y", "size"), s_lo=("score", "min"), s_hi=("score", "max"), mean_pd=("pd", "mean"), bad=("y", "mean"))
    md += ["", "### valid 점수 10분위 (1 = 점수 낮음 = 최고 위험)", "", "| 점수 10분위 | 인원 | 점수 범위 | 모델 평균 PD | 실제 부도율 | 차이(실제−모델) |", "| --- | --- | --- | --- | --- | --- |"]
    for i, r in t.iterrows():
        md.append(f"| {i + 1} | {int(r.n):,} | {r.s_lo:.0f} ~ {r.s_hi:.0f} | {r.mean_pd:.4f} | {r.bad:.4f} | {r.bad - r.mean_pd:+.4f} |")
    edges = [-np.inf, 500, 520, 540, 560, 580, 600, 620, np.inf]
    d["band"] = pd.cut(d.score, edges, right=False)
    tb = d.groupby("band", observed=True).agg(n=("y", "size"), mean_pd=("pd", "mean"), bad=("y", "mean"))
    md += ["", "### valid 20점 간격 구간 (점수가 높을수록 안전)", "", "| 점수 구간 | 인원 | 모델 평균 PD | 실제 부도율 | 차이(실제−모델) |", "| --- | --- | --- | --- | --- |"]
    for b, r in tb.iterrows():
        lo = "" if np.isinf(b.left) else f"{b.left:.0f}"
        hi = "" if np.isinf(b.right) else f"{b.right:.0f}"
        label = f"< {hi}" if lo == "" else (f"≥ {lo}" if hi == "" else f"{lo} ~ {hi}미만")
        md.append(f"| {label} | {int(r.n):,} | {r.mean_pd:.4f} | {r.bad:.4f} | {r.bad - r.mean_pd:+.4f} |" + (" (인원 적음, 해석 주의)" if r.n < 200 else ""))
    md += ["", "## 5. 한계", "",
           "- **DAYS_EMPLOYED 연령 정규화 버전은 다음 단계라 이번에는 하지 않았다.** DAYS_EMPLOYED는 연령과 WoE 상관이 0.53으로 가장 커 연령 대리 가능성이 있는 변수(L3_SPEC B 목록)이고, 스코어카드에도 그대로 들어 있다.",
           "- 단일 분할(valid) 한 번의 비교이고 반복이나 교차검증은 없다. test는 열지 않았다.",
           "- 변수 선택 우선순위의 train gain은 190개 변수 모형(train 적합, valid로 조기 종료)에서 나온 값이다.",
           "- family 안 상관만 정리했고 family 간 상관은 보지 않았다 (양수 계수 3개가 나온 이유로 추정). 구간화·WoE는 전체 train에서 계산한 뒤 CV를 했다.",
           "- 모형 형태 비용 ③은 이 변수 집합과 구간화 규칙에 대한 값이다. 구간 수나 상호작용 항을 더하면 달라질 수 있다.", ""]
    (P2 / "compare_valid.md").write_text("\n".join(md), encoding="utf-8")
    log(f"2-3 비교표 저장(compare_valid.md): AUC GBM(100) {auc[g100]:.4f} → GBM({n_sl}) {auc[g25]:.4f} → GBM({n_sc}) {auc[g22]:.4f} → 스코어카드 {auc[scn]:.4f}. "
        f"변수 축소 비용 {red1 + red2:.4f}, 모형 형태 비용 {form:.4f}, 합계 {total:.4f}. 정수 점수 PD 오차 평균 {err.mean():.5f}, 최대 {err.max():.5f}")


if __name__ == "__main__":
    main()
