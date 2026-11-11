"""연령 포함/제외 × scale_pos_weight 있음/없음 2×2 비교.

실행: python -m src.compare_runs   (각 조합을 먼저 src.train으로 학습해 둔다)
입력: results/<버전>/raw/test_predictions.csv (같은 테스트셋, 같은 행 순서)
출력: results/comparison_2x2/

비교는 모두 같은 테스트셋에서 쌍으로 한다. AUC 차이의 신뢰구간은 같은 부트스트랩 재표본으로 계산한다.
F1·정확도는 쓰지 않는다 (컷오프 기준이 달라 조합 간 비교가 불공평하다)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .metrics import credit_metrics, paired_auc_diffs

CELLS = {
    "연령 포함·가중치 있음": "v0_baseline",
    "연령 포함·가중치 없음": "v1_age_noweight",
    "연령 제외·가중치 있음": "v1_noage_weight",
    "연령 제외·가중치 없음": "v1_noage_noweight",
}
OUT = C.RESULTS_DIR / "comparison_2x2"


def load_cells() -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    data = {name: pd.read_csv(C.RESULTS_DIR / v / "raw" / "test_predictions.csv") for name, v in CELLS.items()}
    ref = next(iter(data.values()))
    for name, d in data.items():
        if not (d[C.ID_COL].to_numpy() == ref[C.ID_COL].to_numpy()).all() or not (d["target"].to_numpy() == ref["target"].to_numpy()).all():
            raise SystemExit(f"{name}: 테스트셋 행/라벨이 기준과 다르다. 쌍 비교 불가")
    return ref["target"].to_numpy(), data


def best_iteration(version: str) -> int | None:
    m = re.search(r"best iteration = (\d+)", (C.RESULTS_DIR / version / "modeling" / "modeling_summary.md").read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, data = load_cells()

    rows = []
    for name, d in data.items():
        for label, col in [("XGBoost 보정 전", "pd_raw"), ("XGBoost Platt 보정 후", "pd")]:
            rows.append({"cell": name, "model": label, "best_iteration": best_iteration(CELLS[name]), **credit_metrics(y, d[col].to_numpy())})
        rows.append({"cell": name, "model": "로지스틱", "best_iteration": None, **credit_metrics(y, d["pd_lr"].to_numpy())})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "comparison_table.csv", index=False, encoding="utf-8-sig")

    A_W, A_N, N_W, N_N = list(CELLS)
    preds = {f"{c}|xgb": data[c]["pd"].to_numpy() for c in CELLS}
    preds.update({f"{c}|lr": data[c]["pd_lr"].to_numpy() for c in (A_W, N_W)})  # 로지스틱은 가중치와 무관
    pairs = [
        ("연령 효과 (가중치 있음): 제외 − 포함", f"{N_W}|xgb", f"{A_W}|xgb"),
        ("연령 효과 (가중치 없음): 제외 − 포함", f"{N_N}|xgb", f"{A_N}|xgb"),
        ("가중치 효과 (연령 포함): 없음 − 있음", f"{A_N}|xgb", f"{A_W}|xgb"),
        ("가중치 효과 (연령 제외): 없음 − 있음", f"{N_N}|xgb", f"{N_W}|xgb"),
        ("둘 다 변경: (제외·없음) − (포함·있음)", f"{N_N}|xgb", f"{A_W}|xgb"),
        ("연령 효과, 로지스틱: 제외 − 포함", f"{N_W}|lr", f"{A_W}|lr"),
    ]
    diffs = paired_auc_diffs(y, preds, pairs)
    diffs.to_csv(OUT / "paired_auc_diff.csv", index=False, encoding="utf-8-sig")
    write_md(tbl, diffs)
    print(tbl.round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))


def write_md(tbl: pd.DataFrame, diffs: pd.DataFrame):
    L = [
        "# 연령 × 가중치 2×2 비교",
        "",
        "`python -m src.compare_runs`로 생성. 같은 테스트셋(46,127건), 같은 분할·시드·하이퍼파라미터. "
        "달라지는 것은 모형 입력에서 `DAYS_BIRTH`·`EMPLOYED_AGE_RATIO`를 뺄지, `scale_pos_weight`를 쓸지뿐이다.",
        "",
        "- 판단 지표는 컷오프와 무관한 AUC·KS·Gini·lift와 보정 지표(Brier·ECE)다. F1·정확도는 컷오프 기준이 조합마다 달라 비교에 쓰지 않았다.",
        "- 순위 지표(AUC·KS·Gini·lift)는 Platt 보정 전후가 같다 (보정은 단조변환). 보정 전후는 Brier·ECE만 다르다.",
        "- 조합: " + ", ".join(f"{k} = `{v}`" for k, v in CELLS.items()),
        "",
        "## 결론: v1 기준 = 연령 제외·가중치 없음 (`v1_noage_noweight`)",
        "",
        "- **채택 이유는 두 가지다.** (1) 연령을 모형이 직접 쓰지 않아야 RQ6(보호변수를 사유로 언급하는지)과 연령 집단 공정성 분석의 전제가 선다. "
        "(2) 가중치를 빼면 보정 전 확률이 이미 맞아서(보정 전 ECE 0.0033) Platt 보정기가 필요 없다. 두 칸에서 보정 전후 Brier는 같고 ECE 차이는 0.0001 이하다.",
        "- **성능이 최고여서 고른 것이 아니다.** AUC는 연령 포함·가중치 없음(0.7727)이 가장 높다. "
        "연령 제외·가중치 없음(0.7695)은 v0(0.7704)과 구분되지 않는다 (AUC 차이 −0.0009, 95% 구간 [−0.0027, +0.0009]).",
        "- 연령 제외의 비용은 같은 가중치 조건에서 AUC −0.003이다 (구간이 0을 제외). 로지스틱에서는 차이가 없다.",
        "- 이후 모든 비교(L0 피처 추가 등)는 이 칸의 숫자를 새 기준선으로 쓴다. v0은 재현·비교용으로 보존한다.",
        "",
        "## 조합별 지표 (XGBoost)",
        "",
        "| 조합 | best iter | AUC | Gini | KS | 1분위 lift | 상위 30% 포착 | 보정 전 Brier | 보정 전 ECE | 보정 후 Brier | 보정 후 ECE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in CELLS:
        raw = tbl[(tbl.cell == cell) & (tbl.model == "XGBoost 보정 전")].iloc[0]
        cal = tbl[(tbl.cell == cell) & (tbl.model == "XGBoost Platt 보정 후")].iloc[0]
        L.append(
            f"| {cell} | {int(cal.best_iteration)} | {cal.AUC:.4f} | {cal.Gini:.4f} | {cal.KS:.4f} | {cal.top_decile_lift:.2f} | {cal.top30_capture:.1%} "
            f"| {raw.Brier:.4f} | {raw.ECE_10q:.4f} | {cal.Brier:.4f} | {cal.ECE_10q:.4f} |"
        )
    L += ["", "## 로지스틱 (가중치와 무관)", "", "| 조합 | AUC | Gini | KS | 1분위 lift | Brier | ECE |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell in CELLS:
        if "가중치 없음" in cell:
            continue
        r = tbl[(tbl.cell == cell) & (tbl.model == "로지스틱")].iloc[0]
        L.append(f"| {cell.split('·')[0]} | {r.AUC:.4f} | {r.Gini:.4f} | {r.KS:.4f} | {r.top_decile_lift:.2f} | {r.Brier:.4f} | {r.ECE_10q:.4f} |")
    L += [
        "",
        "## AUC 차이 (쌍 부트스트랩, B=1,000, 시드 42, 95% percentile)",
        "",
        "같은 재표본에서 두 모형의 AUC를 모두 다시 계산해 차이의 분포를 만든다. 구간이 0을 포함하면 이 표본에서는 차이를 구분할 수 없다.",
        "",
        "| 비교 | AUC a | AUC b | 차이 (a−b) | 95% 구간 | 0 제외 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in diffs.itertuples():
        L.append(f"| {r.comparison} | {r.AUC_a:.4f} | {r.AUC_b:.4f} | {r.diff_a_minus_b:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {'예' if r.ci_excludes_0 else '아니오'} |")
    L += [
        "",
        "## 한계",
        "",
        "1. **시드와 분할이 하나뿐이다.** `random_state=42`, 분할 1회. best iteration이 686~1077로 크게 달라, 일부 차이는 early stopping과 분할의 잡음일 수 있다. "
        "AUC 차이의 신뢰구간은 *테스트셋 표본 변동*만 반영하고 학습·분할 변동은 반영하지 않는다.",
        "2. **ECE 차이에는 신뢰구간이 없다.** 보정 전후·조합 간 ECE 차이(0.002 대 0.003 등)가 유의한지 검정하지 않았다. 10구간 ECE는 표본 잡음이 있어 소수 셋째 자리 차이는 해석하지 않는다.",
        "3. **연령의 대리변수 가능성.** `DAYS_BIRTH`와 `EMPLOYED_AGE_RATIO`를 빼도 `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, "
        "`OWN_CAR_AGE`, `DAYS_LAST_PHONE_CHANGE`, 혼인·학력·직업 등이 나이와 상관되어 연령 정보를 간접적으로 담을 수 있다. 이 실험은 연령의 *직접 사용*만 제거한 것이다. "
        "대리 정도는 측정하지 않았다 (예: 나머지 변수로 연령을 예측하는 모형의 성능).",
        "4. 검증셋을 early stopping과 보정에 함께 썼다. 분리해 확인했을 때 테스트 Brier·ECE 차이는 없었다 (ECE 차이 0.0004 이하, 임시 실험).",
        "5. 하이퍼파라미터는 고정값이고 조합별로 다시 튜닝하지 않았다. 가중치 유무에 따라 최적값이 다를 수 있다.",
        "6. 시간 변수가 없어 OOT 검증이 불가능하다. 모든 비교는 무작위 분할 테스트셋 기준이다.",
        "7. **test 셋을 여러 번 봤다.** 이 문서 시점까지 test(46,127건)는 보고·비교 목적으로 v0 재현·지표 확장, 보정·HL·PSI 점검, 이 2×2의 4칸, L0 Tier1 4단계(기준선 재사용 제외), L0 Tier2 6회(재학습 점검 1 + 테이블 5단계)에서 확인했다 (일부 행으로 한 스모크 점검 2회는 폐기). 이 2×2의 기준 선택(연령 제외·가중치 없음)은 test 비교표를 본 뒤 확정했다. Tier2의 변수 선택은 train/valid만 쓰고 test는 선택이 끝난 묶음의 보고에만 썼다. 별도 hold-out이 없어, 최종 성능을 보고할 때는 test 점수가 모형 선택에 약하게 오염되었을 수 있다는 단서를 붙인다.",
        "",
    ]
    (OUT / "comparison.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
