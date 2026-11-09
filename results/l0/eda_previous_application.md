# EDA: previous_application (3단계)

- 전체 1,670,214행, 고유 SK_ID_CURR 338,857개. 학습 신청서에 없는 키의 행: 256,513행 (제외).
- 키당 행 수 분포 (최소/25%/50%/75%/90%/99%/최대): 1 / 2 / 4 / 7 / 10 / 20 / 77
- 학습 신청서 중 과거 신청 기록이 있는 고객: 291,057명 (94.6%).

## 이상값·점검

- DAYS_DECISION > 0: 0
- AMT_APPLICATION == 0: 392,402
- AMT_CREDIT 결측: 1
- DAYS_FIRST_DRAWING == 365243 (표시값, 이번 단계 미사용): 934,444
- DAYS_LAST_DUE == 365243 (표시값, 이번 단계 미사용): 211,221
- NAME_CONTRACT_STATUS 값: {'Approved': 1036781, 'Canceled': 316319, 'Refused': 290678, 'Unused offer': 26436}

## 사용 컬럼 결측 비율 (전체 파일 기준)

| 컬럼 | 결측 비율 |
| --- | --- |
| `NAME_CONTRACT_STATUS` | 0.0% |
| `AMT_APPLICATION` | 0.0% |
| `AMT_CREDIT` | 0.0% |
| `DAYS_DECISION` | 0.0% |
| `DAYS_FIRST_DRAWING` | 40.3% |
| `DAYS_LAST_DUE` | 40.3% |

## 결정 (보수적 처리)

- 기록이 없는 고객: `PREV_APP_CNT`만 0, 나머지는 결측.
- 요청액 대비 승인액은 상태가 Approved이고 요청액 > 0인 신청만 평균한다.
- 거절 비율의 분모는 모든 상태(Approved, Refused, Canceled, Unused offer)의 신청 수이다.
- "최근 신청 간격"은 해석이 둘이라 둘 다 만들었다: 마지막 신청 경과일(`PREV_DAYS_SINCE_LAST`)과 신청 간 평균 간격(`PREV_MEAN_GAP_DAYS`).
- 이번 단계에서는 `DAYS_FIRST_DRAWING` 등 365243 표시값 컬럼을 쓰지 않는다.

## 새 변수 결측 비율 (학습 신청서 기준)

| 컬럼 | 결측 비율 |
| --- | --- |
| `PREV_APP_CNT` | 0.0% |
| `PREV_REFUSED_RATIO` | 5.4% |
| `PREV_APPROVED_AMT_RATIO` | 6.0% |
| `PREV_DAYS_SINCE_LAST` | 5.4% |
| `PREV_MEAN_GAP_DAYS` | 22.4% |
