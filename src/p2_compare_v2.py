"""Phase 2-4c: 네 모형 비교 (GBM 100변수, GBM 22변수, 단조 GBM 22변수, 스코어카드 22변수) -> results/phase2/compare_valid_v2.md. valid만 사용, test 미사용.

AUC 차이의 95% 구간은 같은 재표본(B=1000, 시드 42, 단순 복원추출)에서 모형별 AUC를 계산해 구한 쌍 부트스트랩 percentile 구간이다.
'격차 메움 비율' = (단조 GBM − 스코어카드) / (GBM22 − 스코어카드), 같은 재표본에서 비율을 계산해 구간을 만든다.
실행: python -m src.p2_compare_v2
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .metrics import credit_metrics, decile_table, fast_auc
from .p2_woe import P2, Data

B = 1000


def log(msg: str, decision: bool = False):
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def main():
    y = Data("valid").y
    raw = P2 / "raw"
    g = json.loads((P2 / "gbm_mono.json").read_text(encoding="utf-8"))
    P = {"GBM(100변수)": np.load(C.RESULTS_DIR / "l0_v2" / "raw" / "valid_p_K100.npy"), "GBM(22변수, v2 설정)": np.load(raw / "valid_p_scorecard_vars.npy"),
         "GBM(22변수, 튜닝)": np.load(raw / "valid_p_free22.npy"), "단조 GBM(22변수, 튜닝)": np.load(raw / "valid_p_mono.npy"), "스코어카드(22변수)": np.load(raw / "valid_p_scorecard.npy")}
    g100, g22v2, g22, mono, sc = list(P)      # g22 = 단조 GBM과 같은 방식으로 튜닝한 제약 없는 GBM (비교의 기준)
    gf = json.loads((P2 / "gbm_free22.json").read_text(encoding="utf-8"))
    M = {k: credit_metrics(y, p) for k, p in P.items()}
    rng = np.random.default_rng(42)
    n = len(y)
    A = {k: np.empty(B) for k in P}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for k, p in P.items():
            A[k][b] = fast_auc(y[idx], p[idx])
    auc = {k: M[k]["AUC"] for k in P}

    def d(a, b_):
        diff = A[a] - A[b_]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return auc[a] - auc[b_], lo, hi

    def fmt(a, b_):
        v, lo, hi = d(a, b_)
        return f"{v:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"

    gap = auc[g22] - auc[sc]
    closed = auc[mono] - auc[sc]
    ratio_b = (A[mono] - A[sc]) / (A[g22] - A[sc])
    r_lo, r_hi = np.percentile(ratio_b, [2.5, 97.5])
    ratio = closed / gap
    ks_gap = M[g22]["KS"] - M[sc]["KS"]
    ks_ratio = (M[mono]["KS"] - M[sc]["KS"]) / ks_gap
    tm = pd.read_csv(P2 / "mono_optuna_trials.csv", encoding="utf-8-sig")
    tf = pd.read_csv(P2 / "free22_optuna_trials.csv", encoding="utf-8-sig")
    n_trial = len(tm)
    ct = pd.read_csv(P2 / "mono_constraints.csv", encoding="utf-8-sig")

    md = ["# Phase 2 비교 v2: GBM(100) · GBM(22) · 단조 GBM(22) · 스코어카드(22) (valid)", "",
          f"- valid {len(y):,}건, 부도율 {y.mean():.4f}. **test는 읽지 않았다.** 22개 변수는 스코어카드 최종 변수와 같다 (train에서만 선택·적합). 연령·성별 제외, scale_pos_weight 없음.",
          "- GBM(100), GBM(22, v2 설정)은 v2 설정(lr 0.03, depth 5, min_child_weight 20, subsample 0.8, colsample 0.7, reg_lambda 5, 조기 종료 150, 튜닝 없음). "
          f"**GBM(22, 튜닝)과 단조 GBM(22, 튜닝)은 완전히 같은 절차**: 같은 입력 22열(범주형은 스코어카드 bin 서수), 같은 train 5-fold, 같은 탐색 범위, Optuna TPE 시드 42, {n_trial} trial, 탐색은 lr 0.1 + 조기 종료 50(train CV AUC 최대화, valid 미사용), "
          "최종은 찾은 파라미터로 lr 0.03에서 재학습(valid AUC 조기 종료 150). 두 모형의 유일한 차이는 단조 제약이다. 조기 종료에 valid를 쓰므로 GBM 계열은 valid에 약간 유리하다.", "",
          "## 1. 지표 비교 (valid)", "",
          "| 모형 | 변수 | AUC | AUC 95% 구간 | Gini | KS | 1분위 lift | 상위30% 포착 | Brier | ECE(10분위) | 평균 예측 PD |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    nv = {g100: 100, g22v2: 22, g22: 22, mono: 22, sc: 22}
    for k in P:
        m = M[k]
        lo, hi = np.percentile(A[k], [2.5, 97.5])
        md.append(f"| {k} | {nv[k]} | {m['AUC']:.4f} | [{lo:.4f}, {hi:.4f}] | {m['Gini']:.4f} | {m['KS']:.4f} | {m['top_decile_lift']:.2f} | {m['top30_capture']:.1%} | {m['Brier']:.4f} | {m['ECE_10q']:.4f} | {m['mean_pred']:.4f} |")
    md += ["", f"## 2. AUC 차이 (앞 − 뒤, 같은 재표본 쌍 부트스트랩 B={B}, 95% 구간)", "", "| 비교 | 의미 | AUC 차이 | 95% 구간 | 0 포함 |", "| --- | --- | --- | --- | --- |"]
    rows = [("GBM(100) − GBM(22, 튜닝)", g100, g22, "변수 축소 비용 (같은 GBM 계열)"), ("GBM(22, 튜닝) − GBM(22, v2 설정)", g22, g22v2, "튜닝 효과 (제약 없음)"),
            ("GBM(22, 튜닝) − 단조 GBM(22, 튜닝)", g22, mono, "단조 제약의 비용 (같은 절차)"),
            ("단조 GBM(22) − 스코어카드(22)", mono, sc, "단조 제약 후에도 남는 모형 형태 격차"), ("GBM(22, 튜닝) − 스코어카드(22)", g22, sc, "원래 모형 형태 격차 (제약 없는 튜닝 GBM 기준)"),
            ("GBM(22, v2 설정) − 스코어카드(22)", g22v2, sc, "참고: 튜닝 전 GBM 기준 격차"),
            ("GBM(100) − 스코어카드(22)", g100, sc, "전체 격차"), ("GBM(100) − 단조 GBM(22)", g100, mono, "변수 축소 + 단조 제약 합계")]
    for lab, a, b_, mean in rows:
        v, ci = fmt(a, b_)
        lo, hi = d(a, b_)[1:]
        md.append(f"| {lab} | {mean} | {v} | {ci} | {'예' if lo <= 0 <= hi else '아니오'} |")
    md += ["", "## 3. 단조 제약이 스코어카드–GBM 격차를 얼마나 메우나", "",
           f"- 기준은 같은 절차로 튜닝한 제약 없는 GBM(22, 튜닝). 격차 = GBM(22, 튜닝) − 스코어카드(22) = AUC {gap:.4f}. 단조 GBM(22)이 스코어카드보다 높은 만큼 = {closed:.4f}. **메움 비율 = {ratio:.0%}** (95% 구간 [{r_lo:.0%}, {r_hi:.0%}], 같은 재표본에서 비율 계산).",
           f"- 남는 격차(단조 GBM − 스코어카드)는 {1 - ratio:.0%}이고, 단조 제약이 깎은 것(GBM 튜닝 − 단조 GBM)은 AUC {auc[g22] - auc[mono]:.4f}이다. KS 기준 메움 비율은 {ks_ratio:.0%}.",
           f"- 참고: 튜닝 전(v2 설정) GBM(22) AUC {auc[g22v2]:.4f} 기준으로 같은 비율을 내면 {(auc[mono] - auc[sc]) / (auc[g22v2] - auc[sc]):.0%} (격차 {auc[g22v2] - auc[sc]:.4f}).",
           f"- 읽는 법: 단조 GBM은 변수마다 단조 관계만 허용하되 변수 사이 상호작용(트리 깊이 {g['best_params']['max_depth']})과 연속적인 모양은 유지한다. 스코어카드와의 남은 차이는 상호작용·세밀한 비선형과 구간화 손실, 계수 추정 방식(로지스틱) 차이를 합친 것이고 여기서 더 쪼개지는 않았다.", "",
           "## 4. 튜닝과 단조 제약 GBM 세부", "",
           f"- 단조 GBM: Optuna TPE(시드 42) {n_trial} trial, 최고 train 5-fold CV AUC {g['cv_auc']:.5f} (trial 0 = v2 설정 {tm.loc[0, 'cv_auc']:.5f}). 기록 `mono_optuna_trials.csv`. 최적: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in g["best_params"].items()) + f". 최종 best_iteration {g['best_iteration']}, lr {g['final_lr']}.",
           f"- 제약 없는 GBM(튜닝): 같은 절차, 최고 CV AUC {gf['cv_auc']:.5f} (trial 0 = v2 설정 {tf.loc[0, 'cv_auc']:.5f}). 기록 `free22_optuna_trials.csv`. 최적: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in gf["best_params"].items()) + f". 최종 best_iteration {gf['best_iteration']}, lr {gf['final_lr']}.",
           "- 탐색 범위: max_depth 2~6, min_child_weight 5~100(log), subsample 0.6~1, colsample_bytree 0.5~1, reg_lambda 0.1~30(log), reg_alpha 1e-3~5(log), gamma 0~2. Optuna 시드 42, fold StratifiedKFold(5, shuffle, 시드 42).",
           f"- 단조 위반 점검(변수 22개 × 분위수 격자 × valid 500행 부분집합): 위반 {g['monotone_violations_in_check']}건.", "",
           "| 변수 | 형태 | 제약 | 뜻 | 첫 구간 train 부도율 | 마지막 구간 부도율 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ct.itertuples():
        md.append(f"| {r.variable} | {r.type} | {r.constraint:+d} | {r.meaning} | {r.bad_rate_first_bin:.3f} | {r.bad_rate_last_bin:.3f} |")
    dec = {k: decile_table(y, P[k]) for k in (g22, mono, sc)}
    md += ["", "## 5. 십분위 (valid, 1 = 최고 위험, 모형 자기 예측 순위)", "", "| 십분위 | GBM(22, 튜닝) 부도율 | lift | 단조 GBM 부도율 | lift | 스코어카드 부도율 | lift |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for i in range(10):
        md.append(f"| {i + 1} | {dec[g22].loc[i, 'bad_rate']:.4f} | {dec[g22].loc[i, 'lift']:.2f} | {dec[mono].loc[i, 'bad_rate']:.4f} | {dec[mono].loc[i, 'lift']:.2f} | {dec[sc].loc[i, 'bad_rate']:.4f} | {dec[sc].loc[i, 'lift']:.2f} |")
    summary = (f"단조 제약 GBM(22)은 제약 없는 튜닝 GBM(22)과 스코어카드(22)의 AUC 격차 {gap:.4f} 중 {closed:.4f}({ratio:.0%}, 95% 구간 {r_lo:.0%}~{r_hi:.0%})를 메웠고, "
               f"AUC {auc[mono]:.4f}로 스코어카드({auc[sc]:.4f})보다 높고 같은 절차로 튜닝한 제약 없는 GBM({auc[g22]:.4f})보다 {auc[g22] - auc[mono]:.4f} 낮다.")
    md += ["", "## 6. 한 문장 요약", "", summary, "", "## 7. 한계", "",
           "- 단일 valid 분할의 비교이고, GBM 네 개는 valid로 조기 종료해 valid에 약간 유리하다 (스코어카드는 valid를 적합에 쓰지 않았다).",
           f"- 튜닝은 {n_trial} trial의 제한된 탐색이다. CV와 valid 순위가 달라질 수 있으며 단조 GBM의 하이퍼파라미터 최적성은 보장하지 않는다.",
           "- 범주형 변수(NAME_EDUCATION_TYPE, OCCUPATION_TYPE)는 스코어카드의 bin 번호(부도율 순)를 서수로 넣어 단조를 걸었다. 이 순서는 train 부도율에서 나온 것이라 스코어카드와 같은 정보를 쓴다.",
           "- 제약 방향은 스코어카드 구간화(train)에서 정했다. 변수별 방향이 train에서 단조로 보이는 것이지 인과적으로 옳다는 뜻은 아니다 (예: DAYS_EMPLOYED는 연령 대리 가능성).",
           "- 이 변수 집합·구간화 규칙에 대한 결과이며 test는 열지 않았다.", ""]
    (P2 / "compare_valid_v2.md").write_text("\n".join(md), encoding="utf-8")
    log(f"4c 비교표 저장(compare_valid_v2.md): AUC GBM(100) {auc[g100]:.4f}, GBM(22) {auc[g22]:.4f}, 단조 GBM(22) {auc[mono]:.4f}, 스코어카드 {auc[sc]:.4f}. 격차 메움 비율 {ratio:.1%} (95% [{r_lo:.1%}, {r_hi:.1%}], KS 기준 {ks_ratio:.1%})")
    return summary


if __name__ == "__main__":
    print(main())
