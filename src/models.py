"""전처리기, 확률 보정기, 평가 함수 (T1-4 ~ T1-7).

Streamlit에서 저장된 객체를 불러올 때 이 모듈의 클래스가 필요하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MIN_CATEGORY_COUNT = 100  # 이보다 적은 범주는 infrequent로 묶음


# ---------------------------------------------------------------- 전처리
class Winsorizer(BaseEstimator, TransformerMixin):
    """학습셋 분위수로 양 끝을 자른다 (로지스틱 baseline 이상치 처리). 결측은 그대로 둔다.

    고유값이 min_unique 이하인 변수(0/1 플래그, 신용조회 횟수 등)는 자르지 않는다.
    자르면 소수 범주가 사라져 상수가 될 수 있다."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995, min_unique: int = 50):
        self.lower = lower
        self.upper = upper
        self.min_unique = min_unique

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        n_unique = np.array([np.unique(col[~np.isnan(col)]).size for col in X.T])
        active = n_unique > self.min_unique
        self.lo_ = np.where(active, np.nanquantile(X, self.lower, axis=0), -np.inf)
        self.hi_ = np.where(active, np.nanquantile(X, self.upper, axis=0), np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def split_columns(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in features if c not in cats]
    return nums, cats


def make_preprocessor(df: pd.DataFrame, features: list[str], kind: str) -> ColumnTransformer:
    """kind='xgb': 수치 그대로(결측 유지) + 원-핫 전체.
    kind='lr': 수치 윈저화·중앙값 대치·표준화 + 원-핫(첫 범주 제거)."""
    nums, cats = split_columns(df, features)
    ohe = OneHotEncoder(
        drop="first" if kind == "lr" else None,
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_COUNT,
        sparse_output=False,
    )
    cat_pipe = Pipeline(
        [("fill", SimpleImputer(strategy="constant", fill_value="Missing")), ("ohe", ohe)]
    )
    if kind == "xgb":
        num_pipe = "passthrough"
    elif kind == "lr":
        num_pipe = Pipeline(
            [
                ("clip", Winsorizer()),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
    else:
        raise ValueError(kind)
    ct = ColumnTransformer(
        [("num", num_pipe, nums), ("cat", cat_pipe, cats)], verbose_feature_names_out=False
    )
    ct.set_output(transform="pandas")
    return ct


def feature_to_variable(feature_names: list[str], variables: list[str]) -> dict[str, str]:
    """전처리 후 컬럼명 → 원래 변수명. 원-핫 더미는 가장 긴 접두사 변수로 묶는다."""
    by_len = sorted(variables, key=len, reverse=True)
    mapping = {}
    for f in feature_names:
        if f in variables:
            mapping[f] = f
            continue
        mapping[f] = next(v for v in by_len if f.startswith(v + "_"))
    return mapping


# ---------------------------------------------------------------- 확률 보정
class Calibrator:
    """보정 전 XGBoost 확률 → 보정된 부도확률. method: 'isotonic' 또는 'platt'."""

    def __init__(self, method: str):
        if method not in ("isotonic", "platt"):
            raise ValueError(method)
        self.method = method

    def fit(self, p_raw, y):
        p_raw = np.asarray(p_raw, dtype=float)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model_.fit(p_raw, y)
        else:
            self.model_ = LogisticRegression(C=1e6, max_iter=1000)
            self.model_.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def predict(self, p_raw):
        p_raw = np.asarray(p_raw, dtype=float)
        if self.method == "isotonic":
            return self.model_.predict(p_raw)
        return self.model_.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


def _logit(p):
    return logit(np.clip(p, 1e-7, 1 - 1e-7))


# ---------------------------------------------------------------- 평가
def best_f1_threshold(y, p) -> float:
    """F1을 최대로 하는 컷오프 (검증셋에서 정하고 테스트셋에 적용)."""
    grid = np.linspace(0.01, 0.99, 197)
    scores = [f1_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


__all__ = [
    "Winsorizer",
    "Calibrator",
    "make_preprocessor",
    "feature_to_variable",
    "split_columns",
    "best_f1_threshold",
    "evaluate",
    "expit",
]
