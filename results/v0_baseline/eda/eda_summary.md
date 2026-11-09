# EDA 요약 (T1-1)

`python -m src.eda`로 생성. 원본 `application_train.csv` 전체 기준.

## 1. 규모와 부도율

- 행 307,511개, 원본 컬럼 122개, 모형 입력 변수 53개 (파생 4개 포함, 성별 제외)
- 부도(TARGET=1) 24,825건, 부도율 **8.07%**

## 2. 결측 비율 (모형 입력 변수, 결측 있는 것만)

| 변수 | 한국어 | 결측 비율 |
| --- | --- | --- |
| `OWN_CAR_AGE` | 차량 연식 | 66.0% |
| `EXT_SOURCE_1` | 외부 신용평가 점수 1 | 56.4% |
| `OCCUPATION_TYPE` | 직업 | 31.3% |
| `EXT_SOURCE_3` | 외부 신용평가 점수 3 | 19.8% |
| `EMPLOYED_AGE_RATIO` | 나이 대비 근속 비율 | 18.0% |
| `DAYS_EMPLOYED` | 근속 기간 | 18.0% |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | 최근 1년 신용조회 수 | 13.5% |
| `AMT_REQ_CREDIT_BUREAU_QRT` | 최근 3개월 신용조회 수 | 13.5% |
| `AMT_REQ_CREDIT_BUREAU_MON` | 최근 1개월 신용조회 수 | 13.5% |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | 최근 1주 신용조회 수 | 13.5% |
| `AMT_REQ_CREDIT_BUREAU_DAY` | 최근 1일 신용조회 수 | 13.5% |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | 최근 1시간 신용조회 수 | 13.5% |
| `OBS_30_CNT_SOCIAL_CIRCLE` | 주변인 30일 연체 관측 수 | 0.3% |
| `DEF_60_CNT_SOCIAL_CIRCLE` | 주변인 60일 연체 발생 수 | 0.3% |
| `OBS_60_CNT_SOCIAL_CIRCLE` | 주변인 60일 연체 관측 수 | 0.3% |
| `DEF_30_CNT_SOCIAL_CIRCLE` | 주변인 30일 연체 발생 수 | 0.3% |
| `EXT_SOURCE_2` | 외부 신용평가 점수 2 | 0.2% |
| `AMT_GOODS_PRICE` | 구매 상품 가격 | 0.1% |
| `CREDIT_TERM` | 상환 속도 | 0.0% |
| `ANNUITY_INCOME_RATIO` | 소득 대비 상환액 (DSR 유사) | 0.0% |
| `AMT_ANNUITY` | 정기 상환액 | 0.0% |
| `CNT_FAM_MEMBERS` | 가족 수 | 0.0% |
| `DAYS_LAST_PHONE_CHANGE` | 휴대폰 변경 경과일 | 0.0% |

- `DAYS_EMPLOYED` 365243(무직·연금 수령 표시값) 55,374건은 결측으로 바꿨다. 그래서 `DAYS_EMPLOYED`, `EMPLOYED_AGE_RATIO`의 결측 비율에 포함된다.

## 3. 이상치 메모

- `AMT_INCOME_TOTAL`: 중앙값 147,150, 99.5% 분위 630,000, 최댓값 117,000,000. 로지스틱 baseline에서는 0.5%/99.5% 윈저화, XGBoost는 그대로.
- `CODE_GENDER = XNA` 4건 (추출·성별 분석에서 제외)
- `NAME_FAMILY_STATUS = Unknown` 2건 (원-핫 시 infrequent로 묶임)

## 4. 집단별 부도율

소득 경계값(학습셋 기준): q1 = 117,000.0, q2 = 180,000.0

**성별**

| 집단 | 인원 | 부도율 |
| --- | --- | --- |
| F | 202,448 | 7.00% |
| M | 105,059 | 10.14% |

**연령**

| 집단 | 인원 | 부도율 |
| --- | --- | --- |
| 청년 | 84,662 | 10.90% |
| 중년 | 154,451 | 7.79% |
| 장년 | 68,398 | 5.22% |

**소득**

| 집단 | 인원 | 부도율 |
| --- | --- | --- |
| 하 | 103,156 | 8.41% |
| 중 | 112,428 | 8.52% |
| 상 | 91,927 | 7.15% |

## 5. 주요 변수: 부도/정상 중앙값

| 변수 | 한국어 | 정상 중앙값 | 부도 중앙값 |
| --- | --- | --- | --- |
| `EXT_SOURCE_1` | 외부 신용평가 점수 1 | 0.517 | 0.362 |
| `EXT_SOURCE_2` | 외부 신용평가 점수 2 | 0.574 | 0.440 |
| `EXT_SOURCE_3` | 외부 신용평가 점수 3 | 0.546 | 0.379 |
| `DAYS_BIRTH` | 나이 | -15,877.000 | -14,282.000 |
| `AMT_INCOME_TOTAL` | 연소득 | 148,500.000 | 135,000.000 |
| `CREDIT_INCOME_RATIO` | 소득 대비 대출 금액 | 3.267 | 3.253 |
| `ANNUITY_INCOME_RATIO` | 소득 대비 상환액 (DSR 유사) | 0.162 | 0.169 |
| `DAYS_EMPLOYED` | 근속 기간 | -1,691.000 | -1,230.000 |

그림: `default_rate_by_group.png`, `key_distributions.png`. 결측 비율 전체: `missing_rates.csv`.
