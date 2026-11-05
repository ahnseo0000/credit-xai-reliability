"""데이터 로드, 정제, 파생 변수, 집단 변수, 분할 (T1-2, T1-3, T0-6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# ---------------------------------------------------------------- 변수 사전
def load_var_dict() -> pd.DataFrame:
    return pd.read_csv(C.VAR_DICT, encoding="utf-8-sig")


def model_features(var_dict: pd.DataFrame | None = None) -> list[str]:
    """variable_dict.csv에서 use == 사용인 변수 (성별 제외)."""
    vd = load_var_dict() if var_dict is None else var_dict
    return vd.loc[vd["use"] == "사용", "variable"].tolist()


# ---------------------------------------------------------------- 로드·정제
def load_raw(path=C.RAW_TRAIN) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """모형 입력용 정제 + 파생 변수. 원본은 바꾸지 않는다."""
    df = raw.copy()

    # 365243은 무직·연금 수령 표시값 → 결측
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(C.DAYS_EMPLOYED_ANOMALY, np.nan)

    # Y/N 플래그 → 1/0
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].map({"Y": 1, "N": 0}).astype(float)

    # 파생 변수 (variable_dict.csv 정의)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    # DAYS_EMPLOYED 결측 처리 후 계산 → 무직·연금 수령자는 결측
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    return df


# ---------------------------------------------------------------- 분할
def split_labels(df: pd.DataFrame) -> pd.Series:
    """TARGET 층화 7:1.5:1.5 분할. 값: train / valid / test."""
    idx = df.index
    holdout = C.VALID_SIZE + C.TEST_SIZE
    train_idx, rest_idx = train_test_split(
        idx, test_size=holdout, stratify=df[C.TARGET_COL], random_state=C.SEED
    )
    valid_idx, test_idx = train_test_split(
        rest_idx,
        test_size=C.TEST_SIZE / holdout,
        stratify=df.loc[rest_idx, C.TARGET_COL],
        random_state=C.SEED,
    )
    labels = pd.Series("train", index=idx, name="split")
    labels[valid_idx] = "valid"
    labels[test_idx] = "test"
    return labels


# ---------------------------------------------------------------- 집단 변수 (docs/interface.md)
def age_years(df: pd.DataFrame) -> pd.Series:
    return -df["DAYS_BIRTH"] / 365.25


def income_cuts(train_df: pd.DataFrame) -> tuple[float, float]:
    """학습셋 AMT_INCOME_TOTAL의 1/3, 2/3 분위수 (pandas 기본 linear 보간)."""
    q = train_df["AMT_INCOME_TOTAL"].quantile([1 / 3, 2 / 3])
    return float(q.iloc[0]), float(q.iloc[1])


def add_groups(df: pd.DataFrame, cuts: tuple[float, float]) -> pd.DataFrame:
    q1, q2 = cuts
    age = age_years(df)
    inc = df["AMT_INCOME_TOTAL"]
    return pd.DataFrame(
        {
            "group_gender": df["CODE_GENDER"].astype(str),
            "group_age": np.select([age < 35, age < 55], ["청년", "중년"], "장년"),
            "group_income": np.select([inc <= q1, inc <= q2], ["하", "중"], "상"),
        },
        index=df.index,
    )
