"""L2 공통: 동결된 v2 모형 입력 행렬을 한 번 만들어 results/l2/raw/ 에 캐시한다.

캐시를 만들 때만 L0 집계(무거움)를 읽는다. 이후 스크립트는 캐시와 동결 모형만 쓴다.
모형·전처리·변수 목록은 results/v2_frozen/ 을 읽기만 하고 수정하지 않는다.
test는 SHAP·중요도 등 '설명'에만 쓴다. 변수·모형·하이퍼파라미터를 test로 바꾸지 않는다.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from .models import feature_to_variable

FROZEN = C.RESULTS_DIR / "v2_frozen"
L2 = C.RESULTS_DIR / "l2"
L2RAW = L2 / "raw"
RUNLOG = L2 / "RUNLOG.md"


def log(msg: str, decision: bool = False):
    L2.mkdir(parents=True, exist_ok=True)
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    print(line, flush=True)
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_cache():
    from .l0_v2 import load_state

    L2RAW.mkdir(parents=True, exist_ok=True)
    feat = json.loads((FROZEN / "features.json").read_text(encoding="utf-8"))
    base_sel, l0_sel = feat["application_features"], feat["l0_features"]
    df, app_ids, split, base_all, names, meta, E, design = load_state()
    pre = joblib.load(FROZEN / "models" / "xgb_preprocessor.joblib")
    cols = np.array([names.index(n) for n in l0_sel])
    out = {}
    for s in ("train", "valid", "test"):
        m = (split == s).to_numpy()
        xb = pre.transform(df.loc[m, base_sel]).to_numpy(np.float32)
        rows = np.flatnonzero(m)
        X = np.hstack([xb, E[np.ix_(rows, cols)]]).astype(np.float32)
        out[s] = X
        np.save(L2RAW / f"X_{s}.npy", X)
        np.save(L2RAW / f"y_{s}.npy", df.loc[m, C.TARGET_COL].to_numpy())
        np.save(L2RAW / f"ids_{s}.npy", df.loc[m, C.ID_COL].to_numpy())
    col_names = list(pre.get_feature_names_out()) + l0_sel
    (L2RAW / "col_names.json").write_text(json.dumps(col_names, ensure_ascii=False), encoding="utf-8")
    # test의 사람이 읽는 원본 변수값 (전처리 전)
    mt = (split == "test").to_numpy()
    vals = pd.concat([df.loc[mt, [C.ID_COL] + base_sel].reset_index(drop=True),
                      pd.DataFrame(E[np.ix_(np.flatnonzero(mt), cols)], columns=l0_sel)], axis=1)
    vals.to_pickle(L2RAW / "test_values.pkl")
    return out


def load_cache():
    feat = json.loads((FROZEN / "features.json").read_text(encoding="utf-8"))
    variables = feat["application_features"] + feat["l0_features"]
    col_names = json.loads((L2RAW / "col_names.json").read_text(encoding="utf-8"))
    f2v = feature_to_variable(col_names, variables)
    d = {s: {k: np.load(L2RAW / f"{k}_{s}.npy", allow_pickle=False) for k in ("X", "y", "ids")} for s in ("train", "valid", "test")}
    return d, col_names, variables, f2v


def load_model() -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(str(FROZEN / "models" / "xgb_v2.json"))
    return m


# family 최종 분류 (2026-11-12 재분류). L0 변수는 docs/variable_dict_l0.csv의 family_final 열(옛 family 열은 그대로 둠)을 읽는다.
# 신청서 변수는 L0 사전에 없어서 아래 상수로 둔다. 호출 결과를 보기 전에 정하고 이후 바꾸지 않는다.
APP_FAMILY = {
    "EXT_SOURCE_1": "외부점수", "EXT_SOURCE_2": "외부점수", "EXT_SOURCE_3": "외부점수",
    # 대출조건
    "AMT_CREDIT": "대출조건", "AMT_ANNUITY": "대출조건", "AMT_GOODS_PRICE": "대출조건", "CREDIT_TERM": "대출조건",
    # 소득·상환부담
    "AMT_INCOME_TOTAL": "소득·상환부담", "CREDIT_INCOME_RATIO": "소득·상환부담", "ANNUITY_INCOME_RATIO": "소득·상환부담",
    # 직업·고용
    "NAME_INCOME_TYPE": "직업·고용", "OCCUPATION_TYPE": "직업·고용", "ORGANIZATION_TYPE": "직업·고용", "DAYS_EMPLOYED": "직업·고용",
    # 생활안정
    "DAYS_REGISTRATION": "생활안정", "DAYS_ID_PUBLISH": "생활안정", "DAYS_LAST_PHONE_CHANGE": "생활안정", "OWN_CAR_AGE": "생활안정",
    # 인구통계·주변 (대리 후보: L3_SPEC 7절 B 목록과 함께 모니터링)
    "NAME_EDUCATION_TYPE": "인구통계·주변(대리 후보)", "NAME_FAMILY_STATUS": "인구통계·주변(대리 후보)",
    "REGION_POPULATION_RELATIVE": "인구통계·주변(대리 후보)", "REGION_RATING_CLIENT_W_CITY": "인구통계·주변(대리 후보)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "인구통계·주변(대리 후보)",
    "FLAG_DOCUMENT_3": "신청행태",
}
FAMILY_NOTE = {"FLAG_DOCUMENT_3": "의미 불명, 해석 주의"}


def family_map(variables: list[str]) -> dict[str, str]:
    """변수 -> family(최종). L0는 docs/variable_dict_l0.csv의 family_final, 신청서는 APP_FAMILY."""
    d = pd.read_csv(C.DOCS_DIR / "variable_dict_l0.csv", encoding="utf-8-sig").set_index("variable")["family_final"].to_dict()
    out = {v: APP_FAMILY.get(v) or d.get(v) for v in variables}
    miss = [v for v, f in out.items() if not isinstance(f, str)]
    assert not miss, f"family 미분류: {miss}"
    return out


if __name__ == "__main__":
    build_cache()
    log("L2 캐시 생성 완료 (results/l2/raw, git 제외)")
