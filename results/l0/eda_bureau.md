# EDA: bureau (2단계)

- 전체 1,716,428행, 고유 SK_ID_CURR 305,811개. 학습 신청서에 없는 키의 행: 251,103행 (제외).
- 키당 행 수 분포 (최소/25%/50%/75%/90%/99%/최대): 1 / 2 / 4 / 8 / 11 / 21 / 116
- 학습 신청서 307,511명 중 bureau 기록이 있는 고객: 263,491명 (85.7%).

## 이상값·점검

- AMT_CREDIT_SUM_DEBT < 0 (음수 잔액): 8,418
- AMT_CREDIT_SUM == 0: 66,582
- DAYS_CREDIT > 0 (미래 개설일): 0
- CREDIT_DAY_OVERDUE > 0: 4,217
- CREDIT_CURRENCY != currency 1: 1,408
- CREDIT_ACTIVE 값: {'Closed': 1079273, 'Active': 630607, 'Sold': 6527, 'Bad debt': 21}

## 사용 컬럼 결측 비율 (전체 파일 기준)

| 컬럼 | 결측 비율 |
| --- | --- |
| `CREDIT_ACTIVE` | 0.0% |
| `CREDIT_CURRENCY` | 0.0% |
| `DAYS_CREDIT` | 0.0% |
| `CREDIT_DAY_OVERDUE` | 0.0% |
| `AMT_CREDIT_MAX_OVERDUE` | 65.5% |
| `AMT_CREDIT_SUM` | 0.0% |
| `AMT_CREDIT_SUM_DEBT` | 15.0% |
| `CREDIT_TYPE` | 0.0% |

## 결정 (보수적 처리)

- 기록이 없는 고객: `BUREAU_CNT`만 0, 나머지는 결측 (0과 "기록 없음"을 구분하기 위해).
- 소진율은 분모(대출 금액 합)가 0 이하이면 결측. 음수 잔액·1 초과 비율은 자르지 않고 그대로 둔다 (트리 모형).
- 통화가 다른 대출도 금액을 환산하지 않고 그대로 합산했다 (비중이 매우 작음, 위 점검 참고).

## 새 변수 결측 비율 (학습 신청서 기준)

| 컬럼 | 결측 비율 |
| --- | --- |
| `BUREAU_CNT` | 0.0% |
| `BUREAU_ACTIVE_CNT` | 14.3% |
| `BUREAU_CLOSED_CNT` | 14.3% |
| `BUREAU_ACTIVE_RATIO` | 14.3% |
| `BUREAU_TYPE_NUNIQUE` | 14.3% |
| `BUREAU_OLDEST_DAYS` | 14.3% |
| `BUREAU_MAX_DAY_OVERDUE` | 14.3% |
| `BUREAU_OVERDUE_ACCT_CNT` | 14.3% |
| `BUREAU_MAX_OVERDUE_AMT` | 40.2% |
| `BUREAU_CREDIT_SUM` | 14.3% |
| `BUREAU_DEBT_SUM` | 16.7% |
| `BUREAU_DEBT_RATIO` | 17.0% |
| `BUREAU_ACTIVE_DEBT_RATIO` | 33.8% |
