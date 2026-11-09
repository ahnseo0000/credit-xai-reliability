"""T1-1 EDA: 부도율, 결측 비율, 주요 변수 분포.

실행: python -m src.eda
출력: results/<MODEL_VERSION>/eda/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import add_groups, clean, income_cuts, load_raw, load_var_dict, model_features, split_labels

OUT = C.EDA_DIR
C.set_korean_font()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df = clean(raw)
    vd = load_var_dict()
    feats = model_features(vd)
    kor = dict(zip(vd["variable"], vd["korean_name"]))
    split = split_labels(df)
    cuts = income_cuts(df[split == "train"])
    groups = add_groups(df, cuts)
    y = df[C.TARGET_COL]

    lines = ["# EDA 요약 (T1-1)", "", "`python -m src.eda`로 생성. 원본 `application_train.csv` 전체 기준.", ""]

    # 1. 규모와 부도율
    lines += [
        "## 1. 규모와 부도율",
        "",
        f"- 행 {len(df):,}개, 원본 컬럼 {raw.shape[1]}개, 모형 입력 변수 {len(feats)}개 (파생 4개 포함, 성별 제외)",
        f"- 부도(TARGET=1) {int(y.sum()):,}건, 부도율 **{y.mean():.2%}**",
        "",
    ]

    # 2. 결측
    miss = df[feats].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    miss_tbl = pd.DataFrame(
        {"variable": miss.index, "korean_name": [kor.get(v, "") for v in miss.index], "missing_rate": miss.values}
    )
    miss_tbl.to_csv(OUT / "missing_rates.csv", index=False, encoding="utf-8-sig")
    lines += ["## 2. 결측 비율 (모형 입력 변수, 결측 있는 것만)", "", "| 변수 | 한국어 | 결측 비율 |", "| --- | --- | --- |"]
    lines += [f"| `{r.variable}` | {r.korean_name} | {r.missing_rate:.1%} |" for r in miss_tbl.itertuples()]
    lines += [
        "",
        f"- `DAYS_EMPLOYED` 365243(무직·연금 수령 표시값) {int((raw['DAYS_EMPLOYED'] == C.DAYS_EMPLOYED_ANOMALY).sum()):,}건은 결측으로 바꿨다. 그래서 `DAYS_EMPLOYED`, `EMPLOYED_AGE_RATIO`의 결측 비율에 포함된다.",
        "",
    ]

    # 3. 이상치 메모
    inc = df["AMT_INCOME_TOTAL"]
    lines += [
        "## 3. 이상치 메모",
        "",
        f"- `AMT_INCOME_TOTAL`: 중앙값 {inc.median():,.0f}, 99.5% 분위 {inc.quantile(0.995):,.0f}, 최댓값 {inc.max():,.0f}. 로지스틱 baseline에서는 0.5%/99.5% 윈저화, XGBoost는 그대로.",
        f"- `CODE_GENDER = XNA` {int((df['CODE_GENDER'] == 'XNA').sum())}건 (추출·성별 분석에서 제외)",
        f"- `NAME_FAMILY_STATUS = Unknown` {int((df['NAME_FAMILY_STATUS'] == 'Unknown').sum())}건 (원-핫 시 infrequent로 묶임)",
        "",
    ]

    # 4. 집단별 부도율
    lines += ["## 4. 집단별 부도율", "", f"소득 경계값(학습셋 기준): q1 = {cuts[0]:,.1f}, q2 = {cuts[1]:,.1f}", ""]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    orders = {"group_gender": ["F", "M"], "group_age": ["청년", "중년", "장년"], "group_income": ["하", "중", "상"]}
    titles = {"group_gender": "성별", "group_age": "연령", "group_income": "소득"}
    for ax, (g, order) in zip(axes, orders.items()):
        t = y.groupby(groups[g]).agg(["mean", "size"]).reindex(order)
        lines += [f"**{titles[g]}**", "", "| 집단 | 인원 | 부도율 |", "| --- | --- | --- |"]
        lines += [f"| {k} | {int(r['size']):,} | {r['mean']:.2%} |" for k, r in t.iterrows()]
        lines += [""]
        ax.bar(order, t["mean"] * 100, color="#4C72B0")
        ax.axhline(y.mean() * 100, color="gray", ls="--", lw=1)
        ax.set_title(f"{titles[g]}별 부도율")
        ax.set_ylabel("부도율 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "default_rate_by_group.png", dpi=150)
    plt.close(fig)

    # 5. 주요 변수 분포 (부도/정상)
    key = {
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": None,
        "EXT_SOURCE_3": None,
        "DAYS_BIRTH": lambda s: -s / 365.25,
        "AMT_INCOME_TOTAL": lambda s: np.log10(s),
        "CREDIT_INCOME_RATIO": None,
        "ANNUITY_INCOME_RATIO": None,
        "DAYS_EMPLOYED": lambda s: -s / 365.25,
    }
    labels = {"DAYS_BIRTH": "나이 (세)", "AMT_INCOME_TOTAL": "log10 연소득", "DAYS_EMPLOYED": "근속 (년)"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    lines += ["## 5. 주요 변수: 부도/정상 중앙값", "", "| 변수 | 한국어 | 정상 중앙값 | 부도 중앙값 |", "| --- | --- | --- | --- |"]
    for ax, (v, f) in zip(axes.ravel(), key.items()):
        s = df[v] if f is None else f(df[v])
        s = s.clip(upper=s.quantile(0.995))
        for t, color, name in [(0, "#4C72B0", "정상"), (1, "#C44E52", "부도")]:
            ax.hist(s[y == t].dropna(), bins=50, density=True, alpha=0.5, color=color, label=name)
        ax.set_title(f"{kor.get(v, v)}\n({v})", fontsize=9)
        ax.set_xlabel(labels.get(v, ""), fontsize=8)
        lines.append(f"| `{v}` | {kor.get(v, '')} | {df.loc[y == 0, v].median():,.3f} | {df.loc[y == 1, v].median():,.3f} |")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "key_distributions.png", dpi=150)
    plt.close(fig)

    lines += ["", "그림: `default_rate_by_group.png`, `key_distributions.png`. 결측 비율 전체: `missing_rates.csv`.", ""]
    (OUT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
