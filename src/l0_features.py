"""L0 신용 데이터 구축: 보조 테이블 집계 피처 (SK_ID_CURR 기준).

집계는 라벨(TARGET)을 쓰지 않는다. 라벨을 쓰는 통계(IV 등)는 l0_run에서 train 분할로만 계산한다.
모든 피처는 아래 REGISTRY에 한국어 설명과 함께 등록되며, 등록되지 않은 변수는 학습에 쓰지 않는다.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

ID = C.ID_COL
L0_DIR = C.RESULTS_DIR / "l0"


@dataclass(frozen=True)
class Feat:
    name: str
    korean: str
    desc: str
    unit: str
    group: str
    stage: int
    source: str


def _f(stage, source, group, rows):
    return [Feat(n, k, d, u, group, stage, source) for n, k, d, u in rows]


REGISTRY: list[Feat] = (
    _f(1, "application_train", "ext_source", [
        ("EXT_SOURCE_MEAN", "외부 신용평가 점수 평균", "외부 신용평가 점수 1~3 중 값이 있는 것의 평균이다.", "0~1"),
        ("EXT_SOURCE_MIN", "외부 신용평가 점수 최솟값", "외부 신용평가 점수 1~3 중 값이 있는 것의 최솟값이다.", "0~1"),
        ("EXT_SOURCE_MAX", "외부 신용평가 점수 최댓값", "외부 신용평가 점수 1~3 중 값이 있는 것의 최댓값이다.", "0~1"),
        ("EXT_SOURCE_PROD", "외부 신용평가 점수 곱", "외부 신용평가 점수 1~3 중 값이 있는 것을 모두 곱한 값이다.", "0~1"),
        ("EXT_SOURCE_MISSING_CNT", "외부 신용평가 점수 결측 개수", "외부 신용평가 점수 1~3 중 값이 없는 것의 개수이다.", "개"),
    ])
    + _f(1, "application_train", "loan_burden", [
        ("INCOME_PER_PERSON", "1인당 소득", "연소득을 가족 수로 나눈 값이다.", "통화 단위"),
        ("CREDIT_GOODS_RATIO", "상품가 대비 대출 금액", "대출 금액을 구매 상품 가격으로 나눈 값이다.", "배"),
    ])
    + _f(1, "application_train", "", [
        ("ROW_MISSING_CNT", "신청서 결측 개수", "모형 입력 신청서 변수 51개 중 값이 비어 있는 항목의 개수이다.", "개"),
        ("DOC_SUBMITTED_CNT", "제출 서류 개수", "서류 2~21번 중 제출한 서류의 개수이다.", "개"),
    ])
    + _f(2, "bureau", "bureau_account", [
        ("BUREAU_CNT", "타기관 대출 건수", "타 금융기관 대출 기록의 총 건수이다 (기록이 없으면 0).", "건"),
        ("BUREAU_ACTIVE_CNT", "타기관 진행 중 대출 수", "타 금융기관 대출 중 진행 중인 건수이다.", "건"),
        ("BUREAU_CLOSED_CNT", "타기관 종료 대출 수", "타 금융기관 대출 중 종료된 건수이다.", "건"),
        ("BUREAU_ACTIVE_RATIO", "타기관 진행 중 대출 비율", "타 금융기관 대출 중 진행 중인 건의 비율이다.", "비율"),
        ("BUREAU_TYPE_NUNIQUE", "타기관 대출 상품 종류 수", "타 금융기관 대출의 상품 종류가 몇 가지인지 센 값이다.", "개"),
        ("BUREAU_OLDEST_DAYS", "최초 대출 개설 경과일", "신청일 기준 가장 오래된 타기관 대출이 개설된 지 며칠 지났는지이다.", "일"),
    ])
    + _f(2, "bureau", "bureau_overdue", [
        ("BUREAU_MAX_DAY_OVERDUE", "타기관 최대 연체일수", "타 금융기관 대출 중 현재 연체일수의 최댓값이다.", "일"),
        ("BUREAU_OVERDUE_ACCT_CNT", "타기관 연체 대출 수", "타 금융기관 대출 중 현재 연체 중인 건수이다.", "건"),
        ("BUREAU_MAX_OVERDUE_AMT", "타기관 최대 연체금액", "타 금융기관 대출에서 기록된 연체금액의 최댓값이다.", "통화 단위"),
    ])
    + _f(2, "bureau", "bureau_utilization", [
        ("BUREAU_CREDIT_SUM", "타기관 대출 금액 합계", "타 금융기관 대출 금액의 합계이다.", "통화 단위"),
        ("BUREAU_DEBT_SUM", "타기관 대출 잔액 합계", "타 금융기관 대출의 현재 잔액 합계이다.", "통화 단위"),
        ("BUREAU_DEBT_RATIO", "타기관 대출 소진율", "타 금융기관 대출 잔액 합계를 대출 금액 합계로 나눈 값이다.", "비율"),
        ("BUREAU_ACTIVE_DEBT_RATIO", "타기관 진행 중 대출 소진율", "진행 중인 타기관 대출만의 잔액 합계를 대출 금액 합계로 나눈 값이다.", "비율"),
    ])
    + _f(3, "previous_application", "prev_application", [
        ("PREV_APP_CNT", "과거 신청 횟수", "Home Credit에 과거 신청한 건수이다 (기록이 없으면 0).", "건"),
        ("PREV_REFUSED_RATIO", "과거 신청 거절 비율", "과거 신청 중 거절된 건의 비율이다.", "비율"),
        ("PREV_APPROVED_AMT_RATIO", "요청액 대비 승인액", "승인된 과거 신청의 승인 금액을 요청 금액으로 나눈 값의 평균이다.", "비율"),
        ("PREV_DAYS_SINCE_LAST", "마지막 과거 신청 경과일", "가장 최근 과거 신청의 결정일로부터 신청일까지 지난 일수이다.", "일"),
        ("PREV_MEAN_GAP_DAYS", "과거 신청 평균 간격", "과거 신청 결정일 사이의 평균 간격이다 (신청이 2건 미만이면 없음).", "일"),
    ])
    + _f(4, "bureau_balance", "bureau_history", [
        ("BB_WORST_3M", "최근 3개월 최악 연체 등급", "타기관 대출의 최근 3개월 월별 상태 중 가장 나쁜 연체 등급이다 (0 정상 ~ 5 심각).", "등급"),
        ("BB_WORST_6M", "최근 6개월 최악 연체 등급", "타기관 대출의 최근 6개월 월별 상태 중 가장 나쁜 연체 등급이다 (0 정상 ~ 5 심각).", "등급"),
        ("BB_WORST_12M", "최근 12개월 최악 연체 등급", "타기관 대출의 최근 12개월 월별 상태 중 가장 나쁜 연체 등급이다 (0 정상 ~ 5 심각).", "등급"),
        ("BB_DPD30_CNT", "30일 초과 연체 월 수", "타기관 대출 이력에서 31일 이상 연체였던 달의 수이다.", "개월"),
        ("BB_DPD60_CNT", "60일 초과 연체 월 수", "타기관 대출 이력에서 61일 이상 연체였던 달의 수이다.", "개월"),
        ("BB_DPD90_CNT", "90일 초과 연체 월 수", "타기관 대출 이력에서 91일 이상 연체였던 달의 수이다.", "개월"),
    ])
)
STAGE_NAMES = {1: "application_derived", 2: "bureau", 3: "previous_application", 4: "bureau_balance"}


def names_for(stage: int) -> list[str]:
    return [f.name for f in REGISTRY if f.stage == stage]


def dict_rows(stages: list[int]) -> pd.DataFrame:
    """variable_dict.csv와 같은 열 + source, stage. 새 변수는 모두 use=사용."""
    return pd.DataFrame(
        [
            {"variable": f.name, "korean_name": f.korean, "description": f.desc, "unit": f.unit, "use": "사용",
             "var_group": f.group, "note": f"L0 집계 ({f.source})", "source": f.source, "stage": f.stage}
            for f in REGISTRY
            if f.stage in stages
        ]
    )


# ---------------------------------------------------------------- 공통
def _qs(s: pd.Series) -> str:
    q = s.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    return " / ".join(f"{v:,.0f}" for v in q.to_numpy())


def _miss_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    L = ["| 컬럼 | 결측 비율 |", "| --- | --- |"]
    for c in cols:
        L.append(f"| `{c}` | {df[c].isna().mean():.1%} |")
    return L


def _check_keys(feat: pd.DataFrame, app_ids: pd.Index, name: str) -> pd.DataFrame:
    """집계 결과를 신청서 ID로 맞추고 중복 키를 점검한다 (중복이면 예외)."""
    if feat.index.has_duplicates:
        raise RuntimeError(f"{name}: 집계 결과에 중복 SK_ID_CURR {int(feat.index.duplicated().sum())}개")
    feat = feat.reindex(app_ids)
    if len(feat) != len(app_ids) or feat.index.has_duplicates:
        raise RuntimeError(f"{name}: 집계 행 수 {len(feat)} != 신청서 {len(app_ids)}")
    return feat


# ---------------------------------------------------------------- 1단계 신청서 파생
def build_stage1(df: pd.DataFrame, base_feats: list[str], app_ids: pd.Index) -> tuple[pd.DataFrame, str]:
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    all_na = ext.isna().all(axis=1)
    fam = df["CNT_FAM_MEMBERS"].where(df["CNT_FAM_MEMBERS"] > 0)
    goods = df["AMT_GOODS_PRICE"].where(df["AMT_GOODS_PRICE"] > 0)
    docs = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    out = pd.DataFrame(
        {
            "EXT_SOURCE_MEAN": ext.mean(axis=1),
            "EXT_SOURCE_MIN": ext.min(axis=1),
            "EXT_SOURCE_MAX": ext.max(axis=1),
            "EXT_SOURCE_PROD": ext.prod(axis=1, skipna=True).where(~all_na),
            "EXT_SOURCE_MISSING_CNT": ext.isna().sum(axis=1),
            "INCOME_PER_PERSON": df["AMT_INCOME_TOTAL"] / fam,
            "CREDIT_GOODS_RATIO": df["AMT_CREDIT"] / goods,
            "ROW_MISSING_CNT": df[base_feats].isna().sum(axis=1),
            "DOC_SUBMITTED_CNT": df[docs].sum(axis=1),
        }
    )
    out.index = df[ID].to_numpy()
    out = _check_keys(out, app_ids, "stage1")
    L = [
        "# EDA: application_train 파생 (1단계)",
        "",
        f"- 행 {len(df):,}개 (키 SK_ID_CURR당 1행). 새 변수 {out.shape[1]}개.",
        f"- 외부 점수 3개가 모두 없는 행: {int(all_na.sum()):,}개 (`EXT_SOURCE_MEAN/MIN/MAX/PROD`는 결측).",
        f"- `CNT_FAM_MEMBERS` ≤ 0: {int((df['CNT_FAM_MEMBERS'] <= 0).sum())}건, `AMT_GOODS_PRICE` ≤ 0 또는 결측: {int(goods.isna().sum()):,}건 (해당 비율은 결측 처리).",
        "- 이미 모형 입력에 있어 새로 만들지 않은 것: 연금/소득(`ANNUITY_INCOME_RATIO`), 대출액/소득(`CREDIT_INCOME_RATIO`), "
        "대출액/연금(`CREDIT_TERM`의 역수, 트리 모형에서 정보 동일), 등록·신분증·전화번호 변경 경과일(`DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`).",
        "- 연령(`DAYS_BIRTH`)이 들어가는 비율은 만들지 않았다.",
        "",
        "## 새 변수 결측 비율",
        "",
        *_miss_table(out, list(out.columns)),
        "",
    ]
    return out, "\n".join(L)


# ---------------------------------------------------------------- 2단계 bureau
def build_stage2(app_ids: pd.Index) -> tuple[pd.DataFrame, str]:
    cols = [ID, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_TYPE"]
    b = pd.read_csv(
        C.RAW_DIR / "bureau.csv", usecols=cols,
        dtype={ID: "int32", "SK_ID_BUREAU": "int32", "CREDIT_ACTIVE": "category", "CREDIT_CURRENCY": "category",
               "DAYS_CREDIT": "int32", "CREDIT_DAY_OVERDUE": "int32", "CREDIT_TYPE": "category"},
    )
    n_all = len(b)
    per_key = b.groupby(ID).size()
    n_not_train = int((~b[ID].isin(app_ids)).sum())
    anomalies = {
        "AMT_CREDIT_SUM_DEBT < 0 (음수 잔액)": int((b["AMT_CREDIT_SUM_DEBT"] < 0).sum()),
        "AMT_CREDIT_SUM == 0": int((b["AMT_CREDIT_SUM"] == 0).sum()),
        "DAYS_CREDIT > 0 (미래 개설일)": int((b["DAYS_CREDIT"] > 0).sum()),
        "CREDIT_DAY_OVERDUE > 0": int((b["CREDIT_DAY_OVERDUE"] > 0).sum()),
        "CREDIT_CURRENCY != currency 1": int((b["CREDIT_CURRENCY"] != "currency 1").sum()),
        "CREDIT_ACTIVE 값": b["CREDIT_ACTIVE"].value_counts().to_dict(),
    }
    miss = _miss_table(b, cols[2:])
    b = b[b[ID].isin(app_ids)].copy()
    b["_active"] = b["CREDIT_ACTIVE"] == "Active"
    b["_closed"] = b["CREDIT_ACTIVE"] == "Closed"
    b["_over"] = b["CREDIT_DAY_OVERDUE"] > 0
    b["_sum_act"] = b["AMT_CREDIT_SUM"].where(b["_active"])
    b["_debt_act"] = b["AMT_CREDIT_SUM_DEBT"].where(b["_active"])
    g = b.groupby(ID)
    out = pd.DataFrame(
        {
            "BUREAU_CNT": g.size(),
            "BUREAU_ACTIVE_CNT": g["_active"].sum(),
            "BUREAU_CLOSED_CNT": g["_closed"].sum(),
            "BUREAU_TYPE_NUNIQUE": g["CREDIT_TYPE"].nunique(),
            "BUREAU_OLDEST_DAYS": -g["DAYS_CREDIT"].min(),
            "BUREAU_MAX_DAY_OVERDUE": g["CREDIT_DAY_OVERDUE"].max(),
            "BUREAU_OVERDUE_ACCT_CNT": g["_over"].sum(),
            "BUREAU_MAX_OVERDUE_AMT": g["AMT_CREDIT_MAX_OVERDUE"].max(),
            "BUREAU_CREDIT_SUM": g["AMT_CREDIT_SUM"].sum(min_count=1),
            "BUREAU_DEBT_SUM": g["AMT_CREDIT_SUM_DEBT"].sum(min_count=1),
        }
    )
    sum_act = g["_sum_act"].sum(min_count=1)
    debt_act = g["_debt_act"].sum(min_count=1)
    out["BUREAU_ACTIVE_RATIO"] = out["BUREAU_ACTIVE_CNT"] / out["BUREAU_CNT"]
    out["BUREAU_DEBT_RATIO"] = out["BUREAU_DEBT_SUM"] / out["BUREAU_CREDIT_SUM"].where(out["BUREAU_CREDIT_SUM"] > 0)
    out["BUREAU_ACTIVE_DEBT_RATIO"] = debt_act / sum_act.where(sum_act > 0)
    out = out[names_for(2)]
    out = _check_keys(out, app_ids, "stage2")
    out["BUREAU_CNT"] = out["BUREAU_CNT"].fillna(0)  # 기록 없음 = 0건, 나머지는 결측 유지
    L = [
        "# EDA: bureau (2단계)",
        "",
        f"- 전체 {n_all:,}행, 고유 SK_ID_CURR {per_key.shape[0]:,}개. 학습 신청서에 없는 키의 행: {n_not_train:,}행 (제외).",
        f"- 키당 행 수 분포 (최소/25%/50%/75%/90%/99%/최대): {_qs(per_key)}",
        f"- 학습 신청서 {len(app_ids):,}명 중 bureau 기록이 있는 고객: {int((out['BUREAU_CNT'] > 0).sum()):,}명 ({(out['BUREAU_CNT'] > 0).mean():.1%}).",
        "",
        "## 이상값·점검",
        "",
        *[f"- {k}: {v:,}" if isinstance(v, int) else f"- {k}: {v}" for k, v in anomalies.items()],
        "",
        "## 사용 컬럼 결측 비율 (전체 파일 기준)",
        "",
        *miss,
        "",
        "## 결정 (보수적 처리)",
        "",
        "- 기록이 없는 고객: `BUREAU_CNT`만 0, 나머지는 결측 (0과 \"기록 없음\"을 구분하기 위해).",
        "- 소진율은 분모(대출 금액 합)가 0 이하이면 결측. 음수 잔액·1 초과 비율은 자르지 않고 그대로 둔다 (트리 모형).",
        "- 통화가 다른 대출도 금액을 환산하지 않고 그대로 합산했다 (비중이 매우 작음, 위 점검 참고).",
        "",
        "## 새 변수 결측 비율 (학습 신청서 기준)",
        "",
        *_miss_table(out, list(out.columns)),
        "",
    ]
    return out, "\n".join(L)


# ---------------------------------------------------------------- 3단계 previous_application
def build_stage3(app_ids: pd.Index) -> tuple[pd.DataFrame, str]:
    cols = ["SK_ID_PREV", ID, "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION",
            "DAYS_FIRST_DRAWING", "DAYS_LAST_DUE"]
    p = pd.read_csv(
        C.RAW_DIR / "previous_application.csv", usecols=cols,
        dtype={"SK_ID_PREV": "int32", ID: "int32", "NAME_CONTRACT_STATUS": "category", "DAYS_DECISION": "int32"},
    )
    n_all = len(p)
    per_key = p.groupby(ID).size()
    n_not_train = int((~p[ID].isin(app_ids)).sum())
    anomalies = {
        "DAYS_DECISION > 0": int((p["DAYS_DECISION"] > 0).sum()),
        "AMT_APPLICATION == 0": int((p["AMT_APPLICATION"] == 0).sum()),
        "AMT_CREDIT 결측": int(p["AMT_CREDIT"].isna().sum()),
        "DAYS_FIRST_DRAWING == 365243 (표시값, 이번 단계 미사용)": int((p["DAYS_FIRST_DRAWING"] == 365243).sum()),
        "DAYS_LAST_DUE == 365243 (표시값, 이번 단계 미사용)": int((p["DAYS_LAST_DUE"] == 365243).sum()),
        "NAME_CONTRACT_STATUS 값": p["NAME_CONTRACT_STATUS"].value_counts().to_dict(),
    }
    miss = _miss_table(p, cols[2:])
    p = p[p[ID].isin(app_ids)].copy()
    p["_ref"] = p["NAME_CONTRACT_STATUS"] == "Refused"
    ok = (p["NAME_CONTRACT_STATUS"] == "Approved") & (p["AMT_APPLICATION"] > 0)
    p["_ratio"] = (p["AMT_CREDIT"] / p["AMT_APPLICATION"]).where(ok)
    p = p.sort_values([ID, "DAYS_DECISION"])
    p["_gap"] = p.groupby(ID)["DAYS_DECISION"].diff()
    g = p.groupby(ID)
    out = pd.DataFrame(
        {
            "PREV_APP_CNT": g.size(),
            "PREV_REFUSED_RATIO": g["_ref"].mean(),
            "PREV_APPROVED_AMT_RATIO": g["_ratio"].mean(),
            "PREV_DAYS_SINCE_LAST": -g["DAYS_DECISION"].max(),
            "PREV_MEAN_GAP_DAYS": g["_gap"].mean(),
        }
    )[names_for(3)]
    out = _check_keys(out, app_ids, "stage3")
    out["PREV_APP_CNT"] = out["PREV_APP_CNT"].fillna(0)
    L = [
        "# EDA: previous_application (3단계)",
        "",
        f"- 전체 {n_all:,}행, 고유 SK_ID_CURR {per_key.shape[0]:,}개. 학습 신청서에 없는 키의 행: {n_not_train:,}행 (제외).",
        f"- 키당 행 수 분포 (최소/25%/50%/75%/90%/99%/최대): {_qs(per_key)}",
        f"- 학습 신청서 중 과거 신청 기록이 있는 고객: {int((out['PREV_APP_CNT'] > 0).sum()):,}명 ({(out['PREV_APP_CNT'] > 0).mean():.1%}).",
        "",
        "## 이상값·점검",
        "",
        *[f"- {k}: {v:,}" if isinstance(v, int) else f"- {k}: {v}" for k, v in anomalies.items()],
        "",
        "## 사용 컬럼 결측 비율 (전체 파일 기준)",
        "",
        *miss,
        "",
        "## 결정 (보수적 처리)",
        "",
        "- 기록이 없는 고객: `PREV_APP_CNT`만 0, 나머지는 결측.",
        "- 요청액 대비 승인액은 상태가 Approved이고 요청액 > 0인 신청만 평균한다.",
        "- 거절 비율의 분모는 모든 상태(Approved, Refused, Canceled, Unused offer)의 신청 수이다.",
        "- \"최근 신청 간격\"은 해석이 둘이라 둘 다 만들었다: 마지막 신청 경과일(`PREV_DAYS_SINCE_LAST`)과 신청 간 평균 간격(`PREV_MEAN_GAP_DAYS`).",
        "- 이번 단계에서는 `DAYS_FIRST_DRAWING` 등 365243 표시값 컬럼을 쓰지 않는다.",
        "",
        "## 새 변수 결측 비율 (학습 신청서 기준)",
        "",
        *_miss_table(out, list(out.columns)),
        "",
    ]
    return out, "\n".join(L)


# ---------------------------------------------------------------- 4단계 bureau_balance (청크)
STATUS_CATS = ["0", "1", "2", "3", "4", "5", "C", "X"]
# 코드 → 연체 등급. C(종료)=0으로 본다. X(알 수 없음)=-1은 어떤 집계에도 넣지 않는다.
STATUS_VAL = np.array([0, 1, 2, 3, 4, 5, 0, -1], dtype=np.int8)


def build_stage4(app_ids: pd.Index, chunk_rows: int = 2_000_000) -> tuple[pd.DataFrame, str]:
    cat = pd.CategoricalDtype(STATUS_CATS)
    parts = []
    n_rows = 0
    status_counts = np.zeros(len(STATUS_CATS), dtype=np.int64)
    m_min, m_max = 0, 0
    reader = pd.read_csv(
        C.RAW_DIR / "bureau_balance.csv", chunksize=chunk_rows,
        dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": cat},
    )
    for ch in reader:
        n_rows += len(ch)
        codes = ch["STATUS"].cat.codes.to_numpy()
        status_counts += np.bincount(codes, minlength=len(STATUS_CATS))
        val = STATUS_VAL[codes]
        mo = ch["MONTHS_BALANCE"].to_numpy()
        m_min, m_max = min(m_min, int(mo.min())), max(m_max, int(mo.max()))
        d = pd.DataFrame({"SK_ID_BUREAU": ch["SK_ID_BUREAU"].to_numpy()})
        d["n"] = np.int32(1)
        for w in (3, 6, 12):
            d[f"w{w}"] = np.where((mo > -w) & (val >= 0), val, -1).astype(np.int8)
        d["c30"] = (val >= 2).astype(np.int16)
        d["c60"] = (val >= 3).astype(np.int16)
        d["c90"] = (val >= 4).astype(np.int16)
        agg = d.groupby("SK_ID_BUREAU").agg(
            n=("n", "sum"), w3=("w3", "max"), w6=("w6", "max"), w12=("w12", "max"),
            c30=("c30", "sum"), c60=("c60", "sum"), c90=("c90", "sum"),
        )
        parts.append(agg)
        del ch, d, agg
        gc.collect()
    acct = pd.concat(parts)
    del parts
    acct = acct.groupby(level=0).agg(
        n=("n", "sum"), w3=("w3", "max"), w6=("w6", "max"), w12=("w12", "max"),
        c30=("c30", "sum"), c60=("c60", "sum"), c90=("c90", "sum"),
    )
    rows_per_acct = acct["n"]
    bmap = pd.read_csv(C.RAW_DIR / "bureau.csv", usecols=[ID, "SK_ID_BUREAU"], dtype="int32")
    if bmap["SK_ID_BUREAU"].duplicated().any():
        raise RuntimeError("bureau의 SK_ID_BUREAU가 중복됨")
    acct = acct.join(bmap.set_index("SK_ID_BUREAU"), how="left")
    n_unmapped = int(acct[ID].isna().sum())
    acct = acct.dropna(subset=[ID])
    acct[ID] = acct[ID].astype("int32")
    acct = acct[acct[ID].isin(app_ids)]
    for w in ("w3", "w6", "w12"):
        acct[w] = acct[w].where(acct[w] >= 0)  # -1(해당 기간에 유효한 상태 없음) → 결측
    g = acct.groupby(ID)
    out = pd.DataFrame(
        {
            "BB_WORST_3M": g["w3"].max(),
            "BB_WORST_6M": g["w6"].max(),
            "BB_WORST_12M": g["w12"].max(),
            "BB_DPD30_CNT": g["c30"].sum(),
            "BB_DPD60_CNT": g["c60"].sum(),
            "BB_DPD90_CNT": g["c90"].sum(),
        }
    )[names_for(4)]
    out = _check_keys(out, app_ids, "stage4")
    L = [
        "# EDA: bureau_balance (4단계)",
        "",
        f"- 전체 {n_rows:,}행을 {chunk_rows:,}행 청크로 읽었다 (dtype downcast: SK_ID_BUREAU int32, MONTHS_BALANCE int16, STATUS category).",
        f"- 고유 SK_ID_BUREAU {len(rows_per_acct):,}개. bureau에 없는 계좌: {n_unmapped:,}개 (제외).",
        f"- 계좌당 행 수 분포 (최소/25%/50%/75%/90%/99%/최대): {_qs(rows_per_acct)}",
        f"- MONTHS_BALANCE 범위: {m_min} ~ {m_max}",
        "",
        "## STATUS 분포",
        "",
        "| STATUS | 행 수 | 비율 |",
        "| --- | --- | --- |",
        *[f"| {s} | {int(c):,} | {c / n_rows:.2%} |" for s, c in zip(STATUS_CATS, status_counts)],
        "",
        "## 결정 (보수적 처리)",
        "",
        "- STATUS 0~5는 연체 등급 그대로(0=정상, 1=1~30일, 2=31~60일, 3=61~90일, 4=91~120일, 5=120일 초과·상각). C(종료)는 0으로, X(알 수 없음)는 어떤 집계에도 넣지 않는다.",
        "- 최근 N개월 = MONTHS_BALANCE > −N (0이 가장 최근 달). 해당 기간에 유효한 상태가 없으면 결측.",
        "- DPD 30/60/90 횟수 = 등급 ≥2 / ≥3 / ≥4인 달의 전체 이력 합계 (31일·61일·91일 이상 연체).",
        "- 고객 단위 집계: 최악 등급은 계좌 간 최댓값, 횟수는 계좌 간 합계. 청크 경계에서 한 계좌가 잘려도 합치도록 계좌별 부분 집계를 모아 다시 집계했다.",
        f"- bureau_balance 기록이 있는 학습 고객: {int(out['BB_WORST_12M'].notna().sum()):,}명 (12개월 최악 등급 기준).",
        "",
        "## 새 변수 결측 비율 (학습 신청서 기준)",
        "",
        *_miss_table(out, list(out.columns)),
        "",
    ]
    return out, "\n".join(L)
