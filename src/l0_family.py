"""변수 family 규칙: docs/variable_dict_l0.csv의 family 열을 채운다.

family는 변수가 재는 *신용 행동의 종류*다 (SHAP·반사실 설명을 묶어서 보고, 같은 정보를 여러 변수가 나눠 가진 경우를 합산해 해석하는 용도).
규칙은 (1) Tier1 변수는 이름으로 직접 지정, (2) Tier2 후보는 (테이블, 측정값)으로 지정한다. 어느 규칙에도 안 걸리면 오류를 낸다 (미분류를 남기지 않는다).

family 목록: 외부점수 / 연체이력 / 상환비율 / 소진율 / 부채규모 / 신용이력길이 / 계약기간 / 이용규모 / 신청행태 / 소득부담
"""
from __future__ import annotations

import pandas as pd

from . import l0_tier2 as T

TIER1 = {
    "EXT_SOURCE_MEAN": "외부점수", "EXT_SOURCE_MIN": "외부점수", "EXT_SOURCE_MAX": "외부점수", "EXT_SOURCE_PROD": "외부점수",
    "EXT_SOURCE_MISSING_CNT": "외부점수",
    "INCOME_PER_PERSON": "소득부담", "CREDIT_GOODS_RATIO": "소득부담",
    "ROW_MISSING_CNT": "신청행태", "DOC_SUBMITTED_CNT": "신청행태",
    "BUREAU_CNT": "이용규모", "BUREAU_ACTIVE_CNT": "이용규모", "BUREAU_CLOSED_CNT": "이용규모", "BUREAU_ACTIVE_RATIO": "이용규모",
    "BUREAU_TYPE_NUNIQUE": "이용규모", "BUREAU_OLDEST_DAYS": "신용이력길이",
    "BUREAU_MAX_DAY_OVERDUE": "연체이력", "BUREAU_OVERDUE_ACCT_CNT": "연체이력", "BUREAU_MAX_OVERDUE_AMT": "연체이력",
    "BUREAU_CREDIT_SUM": "부채규모", "BUREAU_DEBT_SUM": "부채규모", "BUREAU_DEBT_RATIO": "소진율", "BUREAU_ACTIVE_DEBT_RATIO": "소진율",
    "PREV_APP_CNT": "신청행태", "PREV_REFUSED_RATIO": "신청행태", "PREV_APPROVED_AMT_RATIO": "신청행태",
    "PREV_DAYS_SINCE_LAST": "신청행태", "PREV_MEAN_GAP_DAYS": "신청행태",
    "BB_WORST_3M": "연체이력", "BB_WORST_6M": "연체이력", "BB_WORST_12M": "연체이력",
    "BB_DPD30_CNT": "연체이력", "BB_DPD60_CNT": "연체이력", "BB_DPD90_CNT": "연체이력",
}

# (테이블, 측정값) → family. 측정값 'CNT'는 슬라이스 건수.
TIER2 = {
    "installments_payments": {
        "LATE_DAYS": "연체이력", "LATE_FLAG": "연체이력", "LATE7_FLAG": "연체이력", "LATE30_FLAG": "연체이력",
        "PAY_RATIO": "상환비율", "SHORT_AMT": "상환비율", "SHORT_FLAG": "상환비율", "UNPAID_FLAG": "상환비율",
        "INST_AMT": "부채규모", "CNT": "이용규모",
    },
    "credit_card_balance": {
        "UTIL": "소진율", "DRAW_RATIO": "소진율", "ATM_SHARE": "소진율", "DRAW_CNT": "이용규모", "BALANCE": "부채규모",
        "PAY_MIN_RATIO": "상환비율", "MIN_MISSED": "상환비율",
        "DPD": "연체이력", "DPD_DEF": "연체이력", "DPD_FLAG": "연체이력", "CNT": "이용규모",
    },
    "POS_CASH_balance": {
        "DPD": "연체이력", "DPD_DEF": "연체이력", "DPD_FLAG": "연체이력", "DPD_DEF_FLAG": "연체이력",
        "FUT_RATIO": "상환비율", "CNT_INST": "계약기간", "CNT": "이용규모",
    },
    "bureau": {
        "CREDIT_SUM": "부채규모", "DEBT_SUM": "부채규모", "DEBT_RATIO": "소진율",
        "DAY_OVERDUE": "연체이력", "OVERDUE_FLAG": "연체이력", "OVERDUE_AMT": "연체이력", "MAX_OVERDUE_AMT": "연체이력",
        "PROLONG": "계약기간", "ENDDATE_REMAIN": "계약기간", "AGE_DAYS": "신용이력길이", "CNT": "이용규모",
    },
    "previous_application": {
        "APP_AMT": "신청행태", "CREDIT_AMT": "신청행태", "CREDIT_APP_RATIO": "신청행태", "ANNUITY": "신청행태",
        "PAY_CNT": "계약기간", "DOWN_RATE": "신청행태", "INTEREST": "신청행태", "RECENCY": "신청행태",
        "REFUSED_FLAG": "신청행태", "APPROVED_FLAG": "신청행태", "CNT": "신청행태",
    },
    "bureau_balance": {
        "STATUS_NUM": "연체이력", "DPD1_FLAG": "연체이력", "DPD2_FLAG": "연체이력", "DPD3_FLAG": "연체이력",
        "CLOSED_FLAG": "신용이력길이", "UNKNOWN_FLAG": "신용이력길이", "CNT": "이용규모",
    },
}
_PFX = {s.pfx: s for s in T.SPECS.values()}
_STATS = ("MEAN", "SUM", "MAX", "MIN", "CNT")


def family_of(variable: str) -> str:
    if variable in TIER1:
        return TIER1[variable]
    pfx = variable.split("_", 1)[0]
    spec = _PFX.get(pfx)
    if spec is None:
        raise KeyError(f"family 규칙 없음(접두사): {variable}")
    meas_map = TIER2[spec.name]
    body = variable[len(pfx) + 1:]
    if body.endswith("_CNT") and not any(body.endswith(f"_{m}_CNT") for m in meas_map if m != "CNT"):
        return meas_map["CNT"]
    stat = body.rsplit("_", 1)[-1]
    core = body[: -(len(stat) + 1)]
    for m in sorted(meas_map, key=len, reverse=True):
        if m != "CNT" and (core == m or core.endswith("_" + m)):
            return meas_map[m]
    raise KeyError(f"family 규칙 없음(측정값): {variable}")


def add_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fam = [family_of(v) for v in out["variable"]]
    out["family"] = fam
    return out
