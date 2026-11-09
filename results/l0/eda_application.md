# EDA: application_train 파생 (1단계)

- 행 307,511개 (키 SK_ID_CURR당 1행). 새 변수 9개.
- 외부 점수 3개가 모두 없는 행: 172개 (`EXT_SOURCE_MEAN/MIN/MAX/PROD`는 결측).
- `CNT_FAM_MEMBERS` ≤ 0: 0건, `AMT_GOODS_PRICE` ≤ 0 또는 결측: 278건 (해당 비율은 결측 처리).
- 이미 모형 입력에 있어 새로 만들지 않은 것: 연금/소득(`ANNUITY_INCOME_RATIO`), 대출액/소득(`CREDIT_INCOME_RATIO`), 대출액/연금(`CREDIT_TERM`의 역수, 트리 모형에서 정보 동일), 등록·신분증·전화번호 변경 경과일(`DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`).
- 연령(`DAYS_BIRTH`)이 들어가는 비율은 만들지 않았다.

## 새 변수 결측 비율

| 컬럼 | 결측 비율 |
| --- | --- |
| `EXT_SOURCE_MEAN` | 0.1% |
| `EXT_SOURCE_MIN` | 0.1% |
| `EXT_SOURCE_MAX` | 0.1% |
| `EXT_SOURCE_PROD` | 0.1% |
| `EXT_SOURCE_MISSING_CNT` | 0.0% |
| `INCOME_PER_PERSON` | 0.0% |
| `CREDIT_GOODS_RATIO` | 0.1% |
| `ROW_MISSING_CNT` | 0.0% |
| `DOC_SUBMITTED_CNT` | 0.0% |
