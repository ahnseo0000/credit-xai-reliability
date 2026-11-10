"""L0 Tier 2: 타깃 기반 EDA → 대량 후보 생성 (보조 6개 테이블).

1) 테이블마다 train 분할의 라벨로 부도율 EDA를 만든다 (results/l0/target_eda_<table>.md).
2) EDA에서 신호가 보인 축(상태·유형 구간)만 남기고, 집계대상 x 통계 x 시간창 x 상태분리 후보를 만든다.
   집계 자체는 라벨을 쓰지 않는다. 라벨은 EDA와 구간 선택에서 train 고객에만 쓴다.
3) 후보 평가(IV·단독 AUC·증분 AUC)와 ablation은 l0_tier2_run.py.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C

ID = C.ID_COL
L0 = C.RESULTS_DIR / "l0"
WIN = {"L3M": 3, "L6M": 6, "L12M": 12, "L24M": 24}
SIGNAL_MIN_DEV = 0.10      # 상태·유형 구간 유지 조건: 부도율 배수가 1에서 10% 이상 벗어남
SIGNAL_MIN_N = 500         # ... 그리고 구간 내 기록이 있는 train 고객 500명 이상
SIGNAL_MIN_N_BAD = 100     # 구간 안 불량 이벤트 고객의 배수는 100명 이상일 때만 판단에 쓴다


@dataclass
class Meas:
    ko: str
    unit: str
    kind: str  # flag | cont | amt


@dataclass
class Slice:
    name: str
    ko: str
    kind: str           # all | window | split | cross
    fn: Callable
    mset: str           # sets 이름


@dataclass
class Spec:
    name: str
    stage: int
    pfx: str
    table_ko: str
    path: str
    usecols: list
    dtype: dict
    prep: Callable
    measures: dict
    sets: dict          # mset -> [(measure, [stats])]
    slices: list
    bad: str
    key: str
    buckets: list       # (제목, 측정값, 통계, edges, 단위)
    chunk_rows: int = 1_000_000
    setup: Callable | None = None
    note: str = ""
    ctx: dict = field(default_factory=dict)


f32 = np.float32


def _flag(cond, valid=None):
    out = cond.astype(f32)
    if valid is not None:
        out = out.where(valid)
    return out.astype(f32)


def _days_win(col, n_months):
    lim = -30 * n_months
    return lambda c: (c[col] >= lim).to_numpy()


def _mb_win(n, offset=0):
    # 월 단위 잔액 테이블: MONTHS_BALANCE -1이 가장 최근 달 (bureau_balance는 0). 최근 n개월 = MB > -n - offset
    return lambda c: (c["MONTHS_BALANCE"] > -n - offset).to_numpy()


def win_slices(fn_for, mset="win"):
    return [Slice(k, f"최근 {n}개월", "window", fn_for(n), mset) for k, n in WIN.items()]


# ---------------------------------------------------------------- bureau
def _prep_bureau(c):
    c["DAY_OVERDUE"] = c["CREDIT_DAY_OVERDUE"].astype(f32)
    c["OVERDUE_FLAG"] = _flag(c["CREDIT_DAY_OVERDUE"] > 0)
    c["OVERDUE_AMT"] = c["AMT_CREDIT_SUM_OVERDUE"].astype(f32)
    c["MAX_OVERDUE_AMT"] = c["AMT_CREDIT_MAX_OVERDUE"].astype(f32)
    c["CREDIT_SUM"] = c["AMT_CREDIT_SUM"].astype(f32)
    c["DEBT_SUM"] = c["AMT_CREDIT_SUM_DEBT"].astype(f32)
    ok = c["AMT_CREDIT_SUM"] > 0
    c["DEBT_RATIO"] = (c["AMT_CREDIT_SUM_DEBT"] / c["AMT_CREDIT_SUM"]).where(ok).clip(0, 5).astype(f32)
    c["PROLONG"] = c["CNT_CREDIT_PROLONG"].astype(f32)
    c["ENDDATE_REMAIN"] = c["DAYS_CREDIT_ENDDATE"].astype(f32)
    c["AGE_DAYS"] = (-c["DAYS_CREDIT"]).astype(f32)
    c["BAD"] = c["OVERDUE_FLAG"]
    return c


_BUR_TYPES = {"CONSUMER": ("Consumer credit", "소비자 대출"), "CARD": ("Credit card", "신용카드"), "CAR": ("Car loan", "자동차 대출"),
              "MORTGAGE": ("Mortgage", "주택담보 대출"), "MICRO": ("Microloan", "소액 대출")}
_BUR_STATUS = {"ACTIVE": ("Active", "진행 중"), "CLOSED": ("Closed", "종료"), "BADDEBT": ("Bad debt", "부실 채권")}


def bureau_spec():
    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: _days_win("DAYS_CREDIT", n))
    for k, (v, ko) in _BUR_STATUS.items():
        sl.append(Slice(k, f"{ko} 상태", "split", (lambda v: lambda c: (c["CREDIT_ACTIVE"] == v).to_numpy())(v), "split"))
    for k, (v, ko) in _BUR_TYPES.items():
        sl.append(Slice(k, f"{ko} 상품", "split", (lambda v: lambda c: (c["CREDIT_TYPE"] == v).to_numpy())(v), "split"))
    sl.append(Slice("ACTIVE_L12M", "진행 중이면서 최근 12개월 개설", "cross",
                    lambda c: ((c["CREDIT_ACTIVE"] == "Active") & (c["DAYS_CREDIT"] >= -360)).to_numpy(), "split"))
    M = {
        "CREDIT_SUM": Meas("대출 금액", "통화 단위", "amt"), "DEBT_SUM": Meas("대출 잔액", "통화 단위", "amt"),
        "DEBT_RATIO": Meas("잔액/대출금액 비율", "비율", "cont"), "DAY_OVERDUE": Meas("현재 연체일수", "일", "cont"),
        "OVERDUE_FLAG": Meas("연체 중 대출", "건", "flag"), "OVERDUE_AMT": Meas("현재 연체금액", "통화 단위", "amt"),
        "MAX_OVERDUE_AMT": Meas("기록된 최대 연체금액", "통화 단위", "amt"), "PROLONG": Meas("대출 기간 연장 횟수", "회", "amt"),
        "ENDDATE_REMAIN": Meas("만기까지 남은 일수", "일", "cont"), "AGE_DAYS": Meas("개설 경과일", "일", "cont"),
    }
    S = {
        "full": [("CREDIT_SUM", ["mean", "sum", "max"]), ("DEBT_SUM", ["sum", "mean"]), ("DEBT_RATIO", ["mean", "max"]),
                 ("DAY_OVERDUE", ["max", "mean"]), ("OVERDUE_FLAG", ["mean", "sum"]), ("OVERDUE_AMT", ["sum"]),
                 ("MAX_OVERDUE_AMT", ["max"]), ("PROLONG", ["sum"]), ("ENDDATE_REMAIN", ["mean"]), ("AGE_DAYS", ["max", "mean"])],
        "win": [("DEBT_SUM", ["sum"]), ("DEBT_RATIO", ["mean"]), ("DAY_OVERDUE", ["max"]), ("OVERDUE_FLAG", ["mean", "sum"]), ("OVERDUE_AMT", ["sum"])],
        "split": [("DEBT_SUM", ["sum"]), ("DEBT_RATIO", ["mean"]), ("DAY_OVERDUE", ["max"]), ("OVERDUE_FLAG", ["mean", "sum"])],
    }
    B = [("현재 연체일수 최댓값별", "DAY_OVERDUE", "max", [0, 1, 8, 31, 91], "일"),
         ("타기관 대출 잔액 합계별", "DEBT_SUM", "sum", [0, 1, 1e5, 5e5, 1e6, 5e6], "통화 단위"),
         ("잔액/대출금액 비율 평균별", "DEBT_RATIO", "mean", [0, 0.01, 0.25, 0.5, 0.75, 1.0], "비율")]
    return Spec("bureau", 9, "BUR", "타 금융기관 대출 이력", "bureau.csv",
                [ID, "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_MAX_OVERDUE",
                 "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"],
                {ID: "int32", "CREDIT_ACTIVE": "category", "CREDIT_TYPE": "category", "DAYS_CREDIT": "float32", "CREDIT_DAY_OVERDUE": "float32",
                 "DAYS_CREDIT_ENDDATE": "float32", "AMT_CREDIT_MAX_OVERDUE": "float32", "CNT_CREDIT_PROLONG": "float32",
                 "AMT_CREDIT_SUM": "float32", "AMT_CREDIT_SUM_DEBT": "float32", "AMT_CREDIT_SUM_OVERDUE": "float32"},
                _prep_bureau, M, S, sl, "BAD", "DAY_OVERDUE", B,
                note="시간 기준 = DAYS_CREDIT(대출 개설일, 신청일 기준 음수). 불량 이벤트 = 현재 연체일수 > 0.")


# ---------------------------------------------------------------- previous_application
def _prep_prev(c):
    c["APP_AMT"] = c["AMT_APPLICATION"].astype(f32)
    c["CREDIT_AMT"] = c["AMT_CREDIT"].astype(f32)
    c["CREDIT_APP_RATIO"] = (c["AMT_CREDIT"] / c["AMT_APPLICATION"]).where(c["AMT_APPLICATION"] > 0).clip(0, 5).astype(f32)
    c["ANNUITY"] = c["AMT_ANNUITY"].astype(f32)
    c["PAY_CNT"] = c["CNT_PAYMENT"].astype(f32)
    c["DOWN_RATE"] = c["RATE_DOWN_PAYMENT"].astype(f32)
    c["INTEREST"] = c["RATE_INTEREST_PRIMARY"].astype(f32)
    c["RECENCY"] = (-c["DAYS_DECISION"]).astype(f32)
    c["REFUSED_FLAG"] = _flag(c["NAME_CONTRACT_STATUS"] == "Refused")
    c["APPROVED_FLAG"] = _flag(c["NAME_CONTRACT_STATUS"] == "Approved")
    c["BAD"] = c["REFUSED_FLAG"]
    return c


def prev_spec():
    def eq(col, v):
        return lambda c: (c[col] == v).to_numpy()

    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: _days_win("DAYS_DECISION", n))
    for k, v, ko in [("APPROVED", "Approved", "승인"), ("REFUSED", "Refused", "거절"), ("CANCELED", "Canceled", "취소"), ("UNUSED", "Unused offer", "미사용 제안")]:
        sl.append(Slice(k, f"{ko}된 신청", "split", eq("NAME_CONTRACT_STATUS", v), "split"))
    for k, v, ko in [("CASH", "Cash loans", "현금 대출"), ("CONSUMER", "Consumer loans", "소비자 대출"), ("REVOLVING", "Revolving loans", "회전 대출")]:
        sl.append(Slice(k, f"{ko} 신청", "split", eq("NAME_CONTRACT_TYPE", v), "split"))
    for k, v, ko in [("NEWCLI", "New", "신규 고객"), ("REPEAT", "Repeater", "재신청 고객")]:
        sl.append(Slice(k, f"{ko}으로 분류된 신청", "split", eq("NAME_CLIENT_TYPE", v), "split"))
    for v, ko in [("HC", "내부 정책(HC)"), ("LIMIT", "한도 초과(LIMIT)"), ("SCO", "점수 기준(SCO)"), ("SCOFR", "점수·사기 기준(SCOFR)"), ("VERIF", "검증 실패(VERIF)")]:
        sl.append(Slice(f"RJ_{v}", f"거절 사유 {ko}", "split", eq("CODE_REJECT_REASON", v), "split"))
    for k, v, ko in [("POS", "POS", "POS 상품"), ("CASHP", "Cash", "현금 상품"), ("CARDS", "Cards", "카드 상품")]:
        sl.append(Slice(f"PF_{k}", f"{ko} 포트폴리오", "split", eq("NAME_PORTFOLIO", v), "split"))
    for n in (6, 12, 24):
        sl.append(Slice(f"REFUSED_L{n}M", f"최근 {n}개월 거절", "cross",
                        (lambda n: lambda c: ((c["NAME_CONTRACT_STATUS"] == "Refused") & (c["DAYS_DECISION"] >= -30 * n)).to_numpy())(n), "split"))
    sl.append(Slice("APPROVED_L12M", "최근 12개월 승인", "cross",
                    lambda c: ((c["NAME_CONTRACT_STATUS"] == "Approved") & (c["DAYS_DECISION"] >= -360)).to_numpy(), "split"))
    M = {
        "APP_AMT": Meas("신청 금액", "통화 단위", "amt"), "CREDIT_AMT": Meas("승인(최종) 금액", "통화 단위", "amt"),
        "CREDIT_APP_RATIO": Meas("승인금액/신청금액 비율", "비율", "cont"), "ANNUITY": Meas("연금(월 상환액)", "통화 단위", "amt"),
        "PAY_CNT": Meas("상환 횟수", "회", "cont"), "DOWN_RATE": Meas("선납 비율", "비율", "cont"), "INTEREST": Meas("기본 금리", "비율", "cont"),
        "RECENCY": Meas("결정 경과일", "일", "cont"), "REFUSED_FLAG": Meas("거절", "건", "flag"), "APPROVED_FLAG": Meas("승인", "건", "flag"),
    }
    S = {
        "full": [("APP_AMT", ["mean", "sum", "max"]), ("CREDIT_AMT", ["mean", "sum"]), ("CREDIT_APP_RATIO", ["mean", "min"]), ("ANNUITY", ["mean", "max"]),
                 ("PAY_CNT", ["mean", "max"]), ("DOWN_RATE", ["mean"]), ("INTEREST", ["mean"]), ("RECENCY", ["min", "max"]),
                 ("REFUSED_FLAG", ["mean", "sum"]), ("APPROVED_FLAG", ["mean", "sum"])],
        "win": [("APP_AMT", ["mean", "sum"]), ("CREDIT_APP_RATIO", ["mean"]), ("PAY_CNT", ["mean"]), ("REFUSED_FLAG", ["mean", "sum"]), ("APPROVED_FLAG", ["mean", "sum"])],
        "split": [("APP_AMT", ["mean"]), ("CREDIT_APP_RATIO", ["mean"]), ("RECENCY", ["min"])],
    }
    B = [("과거 거절 횟수별", "REFUSED_FLAG", "sum", [0, 1, 2, 3, 5], "건"),
         ("승인금액/신청금액 비율 평균별", "CREDIT_APP_RATIO", "mean", [0, 0.5, 0.9, 1.0, 1.01], "비율"),
         ("신청 금액 평균별", "APP_AMT", "mean", [0, 5e4, 1e5, 2e5, 5e5], "통화 단위")]
    return Spec("previous_application", 10, "PRV", "Home Credit 과거 신청 이력", "previous_application.csv",
                [ID, "NAME_CONTRACT_TYPE", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY",
                 "NAME_CONTRACT_STATUS", "DAYS_DECISION", "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "CNT_PAYMENT"],
                {ID: "int32", "NAME_CONTRACT_TYPE": "category", "AMT_ANNUITY": "float32", "AMT_APPLICATION": "float32", "AMT_CREDIT": "float32",
                 "RATE_DOWN_PAYMENT": "float32", "RATE_INTEREST_PRIMARY": "float32", "NAME_CONTRACT_STATUS": "category", "DAYS_DECISION": "float32",
                 "CODE_REJECT_REASON": "category", "NAME_CLIENT_TYPE": "category", "NAME_PORTFOLIO": "category", "CNT_PAYMENT": "float32"},
                _prep_prev, M, S, sl, "BAD", "REFUSED_FLAG", B,
                note="시간 기준 = DAYS_DECISION(결정일). 불량 이벤트 = 거절.")


# ---------------------------------------------------------------- installments_payments
def _prep_inst(c):
    late = c["DAYS_ENTRY_PAYMENT"] - c["DAYS_INSTALMENT"]
    valid = late.notna()
    c["LATE_DAYS"] = late.astype(f32)
    c["LATE_FLAG"] = _flag(late > 0, valid)
    c["LATE7_FLAG"] = _flag(late > 7, valid)
    c["LATE30_FLAG"] = _flag(late > 30, valid)
    inst, pay = c["AMT_INSTALMENT"], c["AMT_PAYMENT"]
    c["PAY_RATIO"] = (pay / inst).where(inst > 0).clip(0, 5).astype(f32)
    c["SHORT_AMT"] = (inst - pay).clip(lower=0).astype(f32)
    c["SHORT_FLAG"] = _flag(pay < inst - 1, pay.notna())
    c["UNPAID_FLAG"] = _flag(c["DAYS_ENTRY_PAYMENT"].isna())
    c["INST_AMT"] = inst.astype(f32)
    c["BAD"] = c["LATE_FLAG"].fillna(0).astype(f32)
    return c


def inst_spec():
    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: _days_win("DAYS_INSTALMENT", n))
    sl += [
        Slice("LATE", "연체 납부 회차", "split", lambda c: (c["LATE_FLAG"] == 1).to_numpy(), "split"),
        Slice("LATE_L12M", "최근 12개월 중 연체 납부 회차", "cross", lambda c: ((c["LATE_FLAG"] == 1) & (c["DAYS_INSTALMENT"] >= -360)).to_numpy(), "split"),
        Slice("SHORT", "납부액 부족 회차", "split", lambda c: (c["SHORT_FLAG"] == 1).to_numpy(), "split"),
        Slice("SHORT_L12M", "최근 12개월 중 납부액 부족 회차", "cross", lambda c: ((c["SHORT_FLAG"] == 1) & (c["DAYS_INSTALMENT"] >= -360)).to_numpy(), "split"),
    ]
    M = {
        "LATE_DAYS": Meas("납부 지연일수(납부일−약정일)", "일", "cont"), "LATE_FLAG": Meas("연체 납부", "회", "flag"),
        "LATE7_FLAG": Meas("7일 초과 연체 납부", "회", "flag"), "LATE30_FLAG": Meas("30일 초과 연체 납부", "회", "flag"),
        "PAY_RATIO": Meas("납부액/약정액 비율", "비율", "cont"), "SHORT_AMT": Meas("납부액 부족분", "통화 단위", "amt"),
        "SHORT_FLAG": Meas("납부액 부족 회차", "회", "flag"), "UNPAID_FLAG": Meas("납부일 미기록 회차", "회", "flag"),
        "INST_AMT": Meas("약정 납부액", "통화 단위", "amt"),
    }
    S = {
        "full": [("LATE_DAYS", ["mean", "max"]), ("LATE_FLAG", ["mean", "sum"]), ("LATE7_FLAG", ["mean"]), ("LATE30_FLAG", ["sum"]),
                 ("PAY_RATIO", ["mean", "min"]), ("SHORT_AMT", ["sum", "max"]), ("SHORT_FLAG", ["mean"]), ("UNPAID_FLAG", ["sum"]),
                 ("INST_AMT", ["mean", "max"])],
        "win": [("LATE_DAYS", ["mean", "max"]), ("LATE_FLAG", ["mean", "sum"]), ("PAY_RATIO", ["mean", "min"]), ("SHORT_FLAG", ["mean"])],
        "split": [("LATE_DAYS", ["mean", "max"]), ("PAY_RATIO", ["mean"])],
    }
    B = [("납부 지연일수 최댓값별", "LATE_DAYS", "max", [-1e9, 0, 1, 8, 31, 91], "일"),
         ("연체 납부 비율별", "LATE_FLAG", "mean", [0, 1e-9, 0.1, 0.25, 0.5], "비율"),
         ("납부액/약정액 비율 최솟값별", "PAY_RATIO", "min", [0, 0.5, 0.9, 0.99, 1.0], "비율")]
    return Spec("installments_payments", 6, "INS", "할부 납부 이력", "installments_payments.csv",
                [ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
                {ID: "int32", "DAYS_INSTALMENT": "float32", "DAYS_ENTRY_PAYMENT": "float32", "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"},
                _prep_inst, M, S, sl, "BAD", "LATE_DAYS", B,
                note="시간 기준 = DAYS_INSTALMENT(약정 납부일). 불량 이벤트 = 납부일이 약정일보다 늦음.")


# ---------------------------------------------------------------- credit_card_balance
def _prep_card(c):
    lim = c["AMT_CREDIT_LIMIT_ACTUAL"]
    c["UTIL"] = (c["AMT_BALANCE"] / lim).where(lim > 0).clip(-1, 5).astype(f32)
    c["DRAW_RATIO"] = (c["AMT_DRAWINGS_CURRENT"] / lim).where(lim > 0).clip(0, 5).astype(f32)
    c["ATM_SHARE"] = (c["AMT_DRAWINGS_ATM_CURRENT"] / c["AMT_DRAWINGS_CURRENT"]).where(c["AMT_DRAWINGS_CURRENT"] > 0).clip(0, 1).astype(f32)
    mn = c["AMT_INST_MIN_REGULARITY"]
    c["PAY_MIN_RATIO"] = (c["AMT_PAYMENT_CURRENT"] / mn).where(mn > 0).clip(0, 10).astype(f32)
    c["MIN_MISSED"] = _flag(c["AMT_PAYMENT_CURRENT"].fillna(0) < mn, mn > 0)
    c["DPD"] = c["SK_DPD"].astype(f32)
    c["DPD_DEF"] = c["SK_DPD_DEF"].astype(f32)
    c["DPD_FLAG"] = _flag(c["SK_DPD"] > 0)
    c["BALANCE"] = c["AMT_BALANCE"].astype(f32)
    c["DRAW_CNT"] = c["CNT_DRAWINGS_CURRENT"].astype(f32)
    c["BAD"] = c["DPD_FLAG"]
    return c


def card_spec():
    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: (lambda c: (c["MONTHS_BALANCE"] >= -n).to_numpy()))
    for k, v, ko in [("ACTIVE", "Active", "활성"), ("COMPLETED", "Completed", "종료"), ("SIGNED", "Signed", "계약 체결")]:
        sl.append(Slice(k, f"{ko} 상태 월", "split", (lambda v: lambda c: (c["NAME_CONTRACT_STATUS"] == v).to_numpy())(v), "split"))
    sl += [Slice("DPD", "연체 발생 월", "split", lambda c: (c["SK_DPD"] > 0).to_numpy(), "split"),
           Slice("DPD_L12M", "최근 12개월 중 연체 발생 월", "cross", lambda c: ((c["SK_DPD"] > 0) & (c["MONTHS_BALANCE"] >= -12)).to_numpy(), "split")]
    M = {
        "UTIL": Meas("한도 대비 잔액 비율", "비율", "cont"), "DRAW_RATIO": Meas("한도 대비 인출액 비율", "비율", "cont"),
        "ATM_SHARE": Meas("현금서비스 인출 비중", "비율", "cont"), "PAY_MIN_RATIO": Meas("최소납부액 대비 납부액 비율", "비율", "cont"),
        "MIN_MISSED": Meas("최소납부액 미달 월", "개월", "flag"), "DPD": Meas("연체일수", "일", "cont"), "DPD_DEF": Meas("유의미 연체일수", "일", "cont"),
        "DPD_FLAG": Meas("연체 발생 월", "개월", "flag"), "BALANCE": Meas("카드 잔액", "통화 단위", "amt"), "DRAW_CNT": Meas("월 인출 횟수", "회", "cont"),
    }
    S = {
        "full": [("UTIL", ["mean", "max"]), ("DPD", ["max", "mean"]), ("DPD_DEF", ["max"]), ("DPD_FLAG", ["mean", "sum"]),
                 ("DRAW_RATIO", ["mean", "max"]), ("PAY_MIN_RATIO", ["mean", "min"]), ("MIN_MISSED", ["mean", "sum"]),
                 ("BALANCE", ["mean", "max"]), ("ATM_SHARE", ["mean"]), ("DRAW_CNT", ["mean"])],
        "win": [("UTIL", ["mean", "max"]), ("DPD", ["max"]), ("DPD_FLAG", ["mean", "sum"]), ("PAY_MIN_RATIO", ["mean", "min"]),
                ("MIN_MISSED", ["mean"]), ("DRAW_RATIO", ["mean"])],
        "split": [("UTIL", ["mean", "max"]), ("DPD", ["max"])],
    }
    B = [("연체일수 최댓값별", "DPD", "max", [0, 1, 8, 31, 91], "일"),
         ("한도 대비 잔액 비율 평균별", "UTIL", "mean", [-1, 0.01, 0.25, 0.5, 0.75, 1.0], "비율"),
         ("최소납부액 미달 월 수별", "MIN_MISSED", "sum", [0, 1, 2, 4, 8], "개월")]
    return Spec("credit_card_balance", 7, "CCB", "신용카드 월별 잔액 이력", "credit_card_balance.csv",
                [ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
                 "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "CNT_DRAWINGS_CURRENT", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"],
                {ID: "int32", "MONTHS_BALANCE": "int16", "AMT_BALANCE": "float32", "AMT_CREDIT_LIMIT_ACTUAL": "float32",
                 "AMT_DRAWINGS_ATM_CURRENT": "float32", "AMT_DRAWINGS_CURRENT": "float32", "AMT_INST_MIN_REGULARITY": "float32",
                 "AMT_PAYMENT_CURRENT": "float32", "CNT_DRAWINGS_CURRENT": "float32", "NAME_CONTRACT_STATUS": "category",
                 "SK_DPD": "float32", "SK_DPD_DEF": "float32"},
                _prep_card, M, S, sl, "BAD", "DPD", B,
                note="시간 기준 = MONTHS_BALANCE(−1이 가장 최근 달). 불량 이벤트 = 연체일수(SK_DPD) > 0.")


# ---------------------------------------------------------------- POS_CASH_balance
def _prep_pos(c):
    c["DPD"] = c["SK_DPD"].astype(f32)
    c["DPD_DEF"] = c["SK_DPD_DEF"].astype(f32)
    c["DPD_FLAG"] = _flag(c["SK_DPD"] > 0)
    c["DPD_DEF_FLAG"] = _flag(c["SK_DPD_DEF"] > 0)
    c["FUT_RATIO"] = (c["CNT_INSTALMENT_FUTURE"] / c["CNT_INSTALMENT"]).where(c["CNT_INSTALMENT"] > 0).clip(0, 1).astype(f32)
    c["CNT_INST"] = c["CNT_INSTALMENT"].astype(f32)
    c["BAD"] = c["DPD_FLAG"]
    return c


def pos_spec():
    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: (lambda c: (c["MONTHS_BALANCE"] >= -n).to_numpy()))
    for k, v, ko in [("ACTIVE", "Active", "진행 중"), ("COMPLETED", "Completed", "종료"), ("RETURNED", "Returned to the store", "반품")]:
        sl.append(Slice(k, f"{ko} 상태 월", "split", (lambda v: lambda c: (c["NAME_CONTRACT_STATUS"] == v).to_numpy())(v), "split"))
    sl += [Slice("DPD", "연체 발생 월", "split", lambda c: (c["SK_DPD"] > 0).to_numpy(), "split"),
           Slice("DPD_L12M", "최근 12개월 중 연체 발생 월", "cross", lambda c: ((c["SK_DPD"] > 0) & (c["MONTHS_BALANCE"] >= -12)).to_numpy(), "split")]
    M = {
        "DPD": Meas("연체일수", "일", "cont"), "DPD_DEF": Meas("유의미 연체일수", "일", "cont"), "DPD_FLAG": Meas("연체 발생 월", "개월", "flag"),
        "DPD_DEF_FLAG": Meas("유의미 연체 발생 월", "개월", "flag"), "FUT_RATIO": Meas("남은 회차/전체 회차 비율", "비율", "cont"),
        "CNT_INST": Meas("전체 약정 회차", "회", "cont"),
    }
    S = {
        "full": [("DPD", ["max", "mean"]), ("DPD_DEF", ["max", "mean"]), ("DPD_FLAG", ["mean", "sum"]), ("DPD_DEF_FLAG", ["sum"]),
                 ("FUT_RATIO", ["mean", "min"]), ("CNT_INST", ["mean", "max"])],
        "win": [("DPD", ["max", "mean"]), ("DPD_FLAG", ["mean", "sum"]), ("DPD_DEF", ["max"]), ("FUT_RATIO", ["mean"])],
        "split": [("DPD", ["max", "mean"]), ("DPD_FLAG", ["mean"])],
    }
    B = [("연체일수 최댓값별", "DPD", "max", [0, 1, 8, 31, 91], "일"),
         ("연체 발생 월 수별", "DPD_FLAG", "sum", [0, 1, 2, 4, 8], "개월"),
         ("남은 회차 비율 평균별", "FUT_RATIO", "mean", [0, 0.1, 0.3, 0.5, 0.8], "비율")]
    return Spec("POS_CASH_balance", 8, "POS", "POS·현금대출 월별 잔액 이력", "POS_CASH_balance.csv",
                [ID, "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"],
                {ID: "int32", "MONTHS_BALANCE": "int16", "CNT_INSTALMENT": "float32", "CNT_INSTALMENT_FUTURE": "float32",
                 "NAME_CONTRACT_STATUS": "category", "SK_DPD": "float32", "SK_DPD_DEF": "float32"},
                _prep_pos, M, S, sl, "BAD", "DPD", B,
                note="시간 기준 = MONTHS_BALANCE(−1이 가장 최근 달). 불량 이벤트 = 연체일수(SK_DPD) > 0.")


# ---------------------------------------------------------------- bureau_balance (계좌→고객 매핑 필요)
_BB_VAL = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": 0}


def _setup_bb(spec, app_ids):
    b = pd.read_csv(C.RAW_DIR / "bureau.csv", usecols=["SK_ID_BUREAU", ID], dtype="int32")
    b = b.drop_duplicates("SK_ID_BUREAU").sort_values("SK_ID_BUREAU")
    spec.ctx["bu_ids"] = b["SK_ID_BUREAU"].to_numpy()
    spec.ctx["bu_cur"] = b[ID].to_numpy()


def _prep_bb(c):
    st = c["STATUS"].astype(str)
    num = st.map(_BB_VAL).astype(f32)  # X는 NaN
    c["STATUS_NUM"] = num
    c["DPD1_FLAG"] = _flag(num >= 1, num.notna())
    c["DPD2_FLAG"] = _flag(num >= 2, num.notna())
    c["DPD3_FLAG"] = _flag(num >= 3, num.notna())
    c["CLOSED_FLAG"] = _flag(st == "C")
    c["UNKNOWN_FLAG"] = _flag(st == "X")
    c["BAD"] = c["DPD2_FLAG"].fillna(0).astype(f32)
    return c


def bb_spec():
    sl = [Slice("ALL", "전체 기간", "all", lambda c: np.ones(len(c), bool), "full")]
    sl += win_slices(lambda n: (lambda c: (c["MONTHS_BALANCE"] > -n).to_numpy()))
    M = {
        "STATUS_NUM": Meas("월별 연체 등급(0 정상~5 심각)", "등급", "cont"), "DPD1_FLAG": Meas("연체 1등급 이상 월", "개월", "flag"),
        "DPD2_FLAG": Meas("31일 이상 연체 월", "개월", "flag"), "DPD3_FLAG": Meas("61일 이상 연체 월", "개월", "flag"),
        "CLOSED_FLAG": Meas("종료 상태 월", "개월", "flag"), "UNKNOWN_FLAG": Meas("상태 불명 월", "개월", "flag"),
    }
    S = {
        "full": [("STATUS_NUM", ["mean", "max"]), ("DPD1_FLAG", ["mean", "sum"]), ("DPD2_FLAG", ["mean", "sum"]), ("DPD3_FLAG", ["sum"]),
                 ("CLOSED_FLAG", ["mean"]), ("UNKNOWN_FLAG", ["mean"])],
        "win": [("STATUS_NUM", ["mean", "max"]), ("DPD1_FLAG", ["mean", "sum"]), ("DPD2_FLAG", ["mean", "sum"]), ("DPD3_FLAG", ["sum"]),
                ("CLOSED_FLAG", ["mean"]), ("UNKNOWN_FLAG", ["mean"])],
    }
    B = [("월별 연체 등급 최댓값별", "STATUS_NUM", "max", [0, 1, 2, 3, 4, 5], "등급"),
         ("31일 이상 연체 월 수별", "DPD2_FLAG", "sum", [0, 1, 2, 4, 8], "개월"),
         ("연체 1등급 이상 월 비율별", "DPD1_FLAG", "mean", [0, 1e-9, 0.05, 0.1, 0.25], "비율")]
    return Spec("bureau_balance", 11, "BBL", "타기관 대출 월별 상태 이력", "bureau_balance.csv",
                ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
                {"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"},
                _prep_bb, M, S, sl, "BAD", "STATUS_NUM", B, chunk_rows=2_000_000, setup=_setup_bb,
                note="시간 기준 = MONTHS_BALANCE(0이 가장 최근 달). 상태 C는 0등급, X는 집계에서 제외. 불량 이벤트 = 31일 이상 연체 월.")


SPECS = {s.name: s for s in (bureau_spec(), prev_spec(), inst_spec(), card_spec(), pos_spec(), bb_spec())}
TABLE_ORDER = ["installments_payments", "credit_card_balance", "POS_CASH_balance", "bureau", "previous_application"]  # bureau_balance는 Tier2에서 제외(Tier1에 이미 있음)


# ---------------------------------------------------------------- 집계기
def _positions(spec: Spec, c: pd.DataFrame, app_ids: pd.Index) -> np.ndarray:
    if spec.name == "bureau_balance":
        ids = c["SK_ID_BUREAU"].to_numpy()
        j = np.searchsorted(spec.ctx["bu_ids"], ids)
        j = np.minimum(j, len(spec.ctx["bu_ids"]) - 1)
        ok = spec.ctx["bu_ids"][j] == ids
        cur = np.where(ok, spec.ctx["bu_cur"][j], -1)
        return np.where(ok, app_ids.get_indexer(cur), -1)
    return app_ids.get_indexer(c[ID])


def run_pass(spec: Spec, app_ids: pd.Index, slices: list[Slice], cols: list[str], finalize, mem_budget=4.0e8, nrows=None):
    """슬라이스 묶음마다 파일을 한 번 읽어 (합계·건수·최대·최소)를 고객별로 누적한다. 라벨은 쓰지 않는다.
    finalize(slice, acc) → 저장할 객체. 묶음 크기는 메모리 예산으로 정한다."""
    if spec.setup is not None and not spec.ctx:
        spec.setup(spec, app_ids)
    n, M = len(app_ids), len(cols)
    gsize = max(1, int(mem_budget // (n * (M * 16 + 4))))
    out = {}
    path = C.RAW_DIR / spec.path
    for g0 in range(0, len(slices), gsize):
        grp = slices[g0:g0 + gsize]
        acc = {s.name: dict(rows=np.zeros(n, np.int32), sum=np.zeros((n, M), f32), cnt=np.zeros((n, M), np.int32),
                            max=np.full((n, M), -np.inf, f32), min=np.full((n, M), np.inf, f32)) for s in grp}
        reader = pd.read_csv(path, usecols=spec.usecols, dtype=spec.dtype, chunksize=spec.chunk_rows, nrows=nrows)
        for chunk in reader:
            pos = _positions(spec, chunk, app_ids)
            keep = pos >= 0
            if not keep.all():
                chunk, pos = chunk[keep].reset_index(drop=True), pos[keep]
            chunk = spec.prep(chunk)
            vals = chunk[cols]
            for s in grp:
                m = s.fn(chunk)
                if not m.any():
                    continue
                sub, p = vals[m], pos[m]
                g = sub.groupby(p)
                idx = g.size().index.to_numpy()
                a = acc[s.name]
                a["rows"][idx] += g.size().to_numpy().astype(np.int32)
                a["sum"][idx] += g.sum().to_numpy(f32)
                a["cnt"][idx] += g.count().to_numpy(np.int32)
                a["max"][idx] = np.fmax(a["max"][idx], g.max().to_numpy(f32))
                a["min"][idx] = np.fmin(a["min"][idx], g.min().to_numpy(f32))
            del chunk, vals
        for s in grp:
            out[s.name] = finalize(s, acc.pop(s.name))
        gc.collect()
    return out


def stat_array(acc, j, stat):
    cnt = acc["cnt"][:, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "mean":
            v = acc["sum"][:, j] / cnt
        elif stat == "sum":
            v = acc["sum"][:, j].copy()
        elif stat == "max":
            v = acc["max"][:, j].copy()
        else:
            v = acc["min"][:, j].copy()
    if stat != "sum":
        v[cnt == 0] = np.nan
    return v.astype(f32)


# ---------------------------------------------------------------- EDA (train 라벨만)
def _fmt_edges(edges, unit):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"[{lo:g}, {hi:g})")
    out.append(f"[{edges[-1]:g}, ∞)")
    return out


def _md_table(df: pd.DataFrame) -> list[str]:
    L = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for r in df.itertuples(index=False):
        L.append("| " + " | ".join(x for x in r) + " |")
    return L


def _rate_rows(groups: dict[str, np.ndarray], y: np.ndarray, base: float) -> pd.DataFrame:
    rows = []
    for k, m in groups.items():
        n = int(m.sum())
        r = float(y[m].mean()) if n else np.nan
        rows.append({"구간": k, "train 고객 수": n, "부도율": r, "전체 대비 배수": r / base if n else np.nan})
    return pd.DataFrame(rows)


def _show(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["train 고객 수"] = d["train 고객 수"].map(lambda v: f"{v:,}")
    d["부도율"] = d["부도율"].map(lambda v: "-" if pd.isna(v) else f"{v:.2%}")
    d["전체 대비 배수"] = d["전체 대비 배수"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return d


def _interp(df: pd.DataFrame, min_n=300) -> str:
    d = df[(df["train 고객 수"] >= min_n) & df["부도율"].notna()]
    if len(d) < 2:
        return "해석: 표본이 충분한 구간이 2개 미만이라 비교하지 않는다."
    hi, lo = d.loc[d["부도율"].idxmax()], d.loc[d["부도율"].idxmin()]
    return (f"해석: 부도율은 `{hi['구간']}`에서 가장 높고({hi['부도율']:.2%}, 전체의 {hi['전체 대비 배수']:.2f}배, {int(hi['train 고객 수']):,}명), "
            f"`{lo['구간']}`에서 가장 낮다({lo['부도율']:.2%}, {lo['전체 대비 배수']:.2f}배, {int(lo['train 고객 수']):,}명). "
            f"이는 관측된 차이이며 인과를 뜻하지 않는다.")


def eda_pass(spec: Spec, app_ids: pd.Index, y: np.ndarray, train: np.ndarray, nrows=None):
    """1차 집계(EDA용) + 표 생성. 반환: (markdown, 유지할 슬라이스 이름 목록, 슬라이스 판단표)"""
    cols = list(dict.fromkeys([spec.bad, spec.key] + [b[1] for b in spec.buckets]))
    cidx = {c: i for i, c in enumerate(cols)}
    bi = cidx[spec.bad]

    def fin(s: Slice, acc):
        d = {"rows": acc["rows"].copy(), "bad": acc["sum"][:, bi].copy()}
        for title, meas, stat, edges, unit in spec.buckets:
            v = stat_array(acc, cidx[meas], stat)
            v[acc["rows"] == 0] = np.nan
            d[title] = v
        return d

    res = run_pass(spec, app_ids, spec.slices, cols, fin, nrows=nrows)
    yt = y[train]
    base = float(yt.mean())
    L = [f"# 타깃 EDA: {spec.name} ({spec.table_ko})", "",
         f"- **라벨 사용 범위: train 분할 {int(train.sum()):,}명만.** valid/test 라벨은 쓰지 않았다. 전체 train 부도율 {base:.2%}.",
         f"- 단위는 고객(SK_ID_CURR). 한 고객의 기록 여러 건은 묶어서 본다. {spec.note}",
         "- 배수 = 해당 구간 부도율 / 전체 train 부도율. 인과가 아니라 관측된 연관이다."]
    has_all = res["ALL"]["rows"][train] > 0
    L += ["", "## 1. 기록 있음 vs 없음", ""]
    t = _rate_rows({"기록 있음": has_all, "기록 없음": ~has_all}, yt, base)
    L += _md_table(_show(t)) + ["", _interp(t), ""]

    # 시간창
    L += ["## 2. 최근 N개월 구간의 신호 차이", "",
          f"각 시간창 안에서 불량 이벤트가 한 번이라도 있는 고객 / 창 안에 기록은 있으나 불량 이벤트 없는 고객 / 창 안에 기록이 없는 고객의 부도율.", ""]
    wrows = []
    for s in [x for x in spec.slices if x.kind in ("all", "window")]:
        r, b = res[s.name]["rows"][train], res[s.name]["bad"][train]
        g = {"불량 이벤트 있음": b > 0, "기록 있음·불량 없음": (r > 0) & ~(b > 0), "창 안 기록 없음": r == 0}
        tt = _rate_rows(g, yt, base)
        row = {"시간창": s.ko}
        for k, (_, rr) in zip(g, tt.iterrows()):
            row[f"{k} 명수"] = f"{int(rr['train 고객 수']):,}"
            row[f"{k} 부도율"] = "-" if pd.isna(rr["부도율"]) else f"{rr['부도율']:.2%}"
        a, c_ = tt.iloc[0]["부도율"], tt.iloc[1]["부도율"]
        row["불량 있음/없음 배수"] = "-" if pd.isna(a) or pd.isna(c_) or c_ == 0 else f"{a / c_:.2f}"
        wrows.append(row)
    L += _md_table(pd.DataFrame(wrows)) + ["", ]
    ratios = [(r["시간창"], r["불량 있음/없음 배수"]) for r in wrows if r["불량 있음/없음 배수"] != "-"]
    if ratios:
        best = max(ratios, key=lambda x: float(x[1]))
        L += [f"해석: 불량 이벤트 유무에 따른 부도율 배수는 {', '.join(f'{k} {v}' for k, v in ratios)}이며, 가장 큰 창은 `{best[0]}`이다.", ""]

    # 상태·유형 + 교차
    splits = [s for s in spec.slices if s.kind in ("split", "cross")]
    keep = [s.name for s in spec.slices if s.kind in ("all", "window")]
    judge = []
    if splits:
        L += ["## 3. 상태·유형 구간별 부도율", "",
              "구간에 해당하는 기록이 한 건이라도 있는 고객의 부도율. 마지막 두 열은 그 구간 안에서 불량 이벤트도 있는 고객.", ""]
        srows = []
        for s in splits:
            r, b = res[s.name]["rows"][train], res[s.name]["bad"][train]
            m1, m2 = r > 0, b > 0
            n1, n2 = int(m1.sum()), int(m2.sum())
            r1 = float(yt[m1].mean()) if n1 else np.nan
            r2 = float(yt[m2].mean()) if n2 else np.nan
            dev = max(abs(r1 / base - 1) if n1 >= SIGNAL_MIN_N else 0, abs(r2 / base - 1) if n2 >= SIGNAL_MIN_N_BAD else 0)
            kept = dev >= SIGNAL_MIN_DEV and n1 >= SIGNAL_MIN_N
            if kept:
                keep.append(s.name)
            srows.append({"구간": s.ko, "고객 수": f"{n1:,}", "부도율": "-" if n1 == 0 else f"{r1:.2%}", "배수": "-" if n1 == 0 else f"{r1 / base:.2f}",
                          "불량 포함 고객 수": f"{n2:,}", "불량 포함 부도율": "-" if n2 == 0 else f"{r2:.2%}", "불량 포함 배수": "-" if n2 == 0 else f"{r2 / base:.2f}",
                          "후보 생성": "유지" if kept else "제외(신호 약함/표본 적음)"})
            judge.append({"slice": s.name, "ko": s.ko, "n_train": n1, "rate": r1, "ratio": r1 / base if n1 else np.nan, "n_bad_train": n2,
                          "rate_bad": r2, "max_dev": dev, "kept": kept})
        L += _md_table(pd.DataFrame(srows)) + [""]
        jd = pd.DataFrame(judge)
        jd2 = jd[jd.n_train >= SIGNAL_MIN_N]
        if len(jd2):
            top = jd2.loc[jd2["max_dev"].idxmax()]
            L += [f"해석: 유지 기준은 (구간 부도율 배수 또는 구간 내 불량 포함 고객의 배수[{SIGNAL_MIN_N_BAD}명 이상일 때])가 1에서 {SIGNAL_MIN_DEV:.0%} 이상 벗어나고 구간 고객 {SIGNAL_MIN_N}명 이상. "
                  f"{len(splits)}개 중 {int(jd.kept.sum())}개를 유지했고, 가장 크게 벗어난 구간은 `{top['ko']}`(배수 {top['ratio']:.2f})이다.", ""]
    # 구간 표
    L += ["## 4. 값 구간별 부도율 (전체 기간 집계값)", ""]
    for title, meas, stat, edges, unit in spec.buckets:
        arr = res["ALL"][title][train]
        none = np.isnan(arr)
        lab = _fmt_edges(edges, unit)
        code = np.searchsorted(np.asarray(edges, float), np.nan_to_num(arr, nan=-1e18), side="right") - 1
        g = {"기록 없음/값 없음": none}
        for i, nm in enumerate(lab):
            g[f"{nm}"] = (~none) & (code == i)
        g = {k: v for k, v in g.items() if v.sum() > 0}
        t = _rate_rows(g, yt, base)
        L += [f"### {title} ({unit})", ""] + _md_table(_show(t)) + ["", _interp(t), ""]
    return "\n".join(L) + "\n", keep, pd.DataFrame(judge)


# ---------------------------------------------------------------- 후보 생성
_STAT_KO = {"mean": "평균", "sum": "합계", "max": "최댓값", "min": "최솟값"}


def cand_name(spec: Spec, s: Slice, meas: str, stat: str) -> str:
    return f"{spec.pfx}_{s.name}_{meas}_{stat.upper()}"


def build_candidates(spec: Spec, app_ids: pd.Index, keep_slices: list[str], nrows=None):
    """(후보 DataFrame[float32, 인덱스=SK_ID_CURR], 등록부 DataFrame, 규칙 설명)"""
    slices = [s for s in spec.slices if s.name in keep_slices]
    assert slices[0].name == "ALL"
    cols = list(dict.fromkeys(m for s in slices for m, _ in spec.sets[s.mset]))
    cidx = {c: i for i, c in enumerate(cols)}
    store = {}
    reg = []
    has_any = {}

    def fin(s: Slice, acc):
        if s.name == "ALL":
            has_any["v"] = acc["rows"] > 0
        cnt_name = f"{spec.pfx}_{s.name}_CNT"
        store[cnt_name] = acc["rows"].astype(f32)
        reg.append((cnt_name, f"{s.ko} 건수", f"{spec.table_ko}에서 {s.ko}에 해당하는 기록의 건수이다 (기록이 없으면 0).",
                    "건", s.name, "CNT", "CNT"))
        for meas, stats in spec.sets[s.mset]:
            mm = spec.measures[meas]
            for st in stats:
                v = stat_array(acc, cidx[meas], st)
                if st == "sum" and mm.kind == "flag":
                    v = np.where(has_any["v"], v, np.nan).astype(f32)
                nm = cand_name(spec, s, meas, st)
                store[nm] = v
                unit = mm.unit if st != "mean" or mm.kind != "flag" else "비율"
                what = f"{mm.ko} {_STAT_KO[st] if not (st == 'sum' and mm.kind == 'flag') else '횟수'}"
                reg.append((nm, f"{s.ko} {what}", f"{spec.table_ko}에서 {s.ko} 기록만 모아 계산한 {mm.ko}의 "
                            f"{'발생 횟수' if (st == 'sum' and mm.kind == 'flag') else '발생 비율' if (st == 'mean' and mm.kind == 'flag') else _STAT_KO[st]}이다"
                            f"{' (해당 기록이 없으면 비어 있음)' if not (st == 'sum' and mm.kind == 'flag') else ''}.",
                            unit, s.name, meas, st))
        return None

    run_pass(spec, app_ids, slices, cols, fin, nrows=nrows)
    feat = pd.DataFrame(store, index=app_ids)
    feat.index.name = ID
    assert not feat.index.has_duplicates and len(feat) == len(app_ids)
    regdf = pd.DataFrame(reg, columns=["variable", "korean_name", "description", "unit", "slice", "measure", "stat"])
    regdf["use"] = "사용"
    regdf["var_group"] = spec.name + "_" + regdf["slice"]
    regdf["note"] = "L0 Tier2 후보 (" + spec.name + ")"
    regdf["source"] = spec.name
    regdf["stage"] = spec.stage
    return feat, regdf
