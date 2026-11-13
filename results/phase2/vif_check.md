# Phase 2 점검: 스코어카드 22개 변수의 상관과 VIF (train)

- 계산 대상: 스코어카드가 실제로 쓰는 WoE 변환값 22개 열, train 215,257건. **test는 읽지 않았다.** 구간화는 p2_scorecard와 같은 규칙으로 train에서 다시 만들었다.
- 기준: 쌍 |상관| > 0.7, VIF ≥ 5 주의, ≥ 10 문제 (VIF = 상관행렬 역행렬의 대각, 표준화 WoE 기준). **이 문서는 점검과 제안만 한다. 모형은 바꾸지 않았다.**
- 이전 선택 단계에서는 같은 family 안의 상관만 정리했다(|상관|>0.7). 이번에는 family 간 상관을 처음 본다.

## 1. |상관| > 0.7 쌍

| 변수 A | 변수 B | family A | family B | 같은 family | 상관 |
| --- | --- | --- | --- | --- | --- |
| INS_ALL_PAY_RATIO_MEAN | INS_ALL_LATE_FLAG_MEAN | 상환비율 | 연체이력 | 아니오 (family 간) | +0.725 |

참고: 기준 아래라도 상관이 큰 상위 10쌍.

| 변수 A | 변수 B | family A | family B | 상관 |
| --- | --- | --- | --- | --- |
| INS_ALL_PAY_RATIO_MEAN | INS_ALL_LATE_FLAG_MEAN | 상환비율 | 연체이력 | +0.725 |
| INS_L24M_PAY_RATIO_MEAN | INS_L24M_LATE_DAYS_MAX | 상환비율 | 연체이력 | +0.682 |
| INS_L24M_PAY_RATIO_MEAN | INS_ALL_PAY_RATIO_MEAN | 상환비율 | 상환비율 | +0.657 |
| BUREAU_ACTIVE_RATIO | BUREAU_ACTIVE_CNT | 이용규모 | 이용규모 | +0.564 |
| INS_L24M_LATE_DAYS_MAX | INS_L6M_LATE_DAYS_MAX | 연체이력 | 연체이력 | +0.551 |
| BUR_ALL_DEBT_RATIO_MAX | BUR_ACTIVE_DEBT_RATIO_MEAN | 소진율 | 소진율 | +0.548 |
| BUREAU_OLDEST_DAYS | BUREAU_ACTIVE_RATIO | 신용이력길이 | 이용규모 | +0.528 |
| INS_L24M_LATE_DAYS_MAX | INS_ALL_LATE_FLAG_MEAN | 연체이력 | 연체이력 | +0.515 |
| BUR_ALL_DEBT_RATIO_MAX | BUREAU_ACTIVE_CNT | 소진율 | 이용규모 | +0.511 |
| INS_L24M_PAY_RATIO_MEAN | INS_ALL_LATE_FLAG_MEAN | 상환비율 | 연체이력 | +0.472 |

## 2. 변수별 VIF (큰 순)

| 변수 | family | IV | 계수 | VIF | 가장 상관 큰 상대 | |상관| |
| --- | --- | --- | --- | --- | --- | --- |
| INS_ALL_PAY_RATIO_MEAN | 상환비율 | 0.055 | -0.059 | 3.27 | INS_ALL_LATE_FLAG_MEAN | 0.725 |
| INS_L24M_PAY_RATIO_MEAN | 상환비율 | 0.074 | -0.043 | 2.96 | INS_L24M_LATE_DAYS_MAX | 0.682 |
| INS_L24M_LATE_DAYS_MAX | 연체이력 | 0.093 | -0.362 | 2.64 | INS_L24M_PAY_RATIO_MEAN | 0.682 |
| INS_ALL_LATE_FLAG_MEAN | 연체이력 | 0.062 | -0.233 | 2.60 | INS_ALL_PAY_RATIO_MEAN | 0.725 |
| BUR_ALL_DEBT_RATIO_MAX | 소진율 | 0.143 | -0.338 | 2.38 | BUR_ACTIVE_DEBT_RATIO_MEAN | 0.548 |
| BUREAU_ACTIVE_RATIO | 이용규모 | 0.085 | -0.168 | 2.22 | BUREAU_ACTIVE_CNT | 0.564 |
| BUREAU_ACTIVE_CNT | 이용규모 | 0.055 | -0.178 | 2.21 | BUREAU_ACTIVE_RATIO | 0.564 |
| BUR_ACTIVE_DEBT_RATIO_MEAN | 소진율 | 0.093 | -0.355 | 1.77 | BUR_ALL_DEBT_RATIO_MAX | 0.548 |
| BUREAU_OLDEST_DAYS | 신용이력길이 | 0.074 | -0.071 | 1.64 | BUREAU_ACTIVE_RATIO | 0.528 |
| INS_L6M_LATE_DAYS_MAX | 연체이력 | 0.064 | -0.418 | 1.46 | INS_L24M_LATE_DAYS_MAX | 0.551 |
| EXT_SOURCE_MEAN | 외부점수 | 0.611 | -0.728 | 1.24 | BUR_ALL_DEBT_RATIO_MAX | 0.210 |
| DAYS_EMPLOYED | 직업·고용 | 0.113 | -0.469 | 1.20 | OCCUPATION_TYPE | 0.260 |
| OCCUPATION_TYPE | 직업·고용 | 0.075 | -0.386 | 1.17 | DAYS_EMPLOYED | 0.260 |
| NAME_EDUCATION_TYPE | 인구통계·주변(대리 후보) | 0.048 | -0.495 | 1.12 | OCCUPATION_TYPE | 0.254 |
| INS_ALL_INST_AMT_MAX | 부채규모 | 0.036 | -0.460 | 1.11 | INS_ALL_PAY_RATIO_MEAN | 0.202 |
| DAYS_LAST_PHONE_CHANGE | 생활안정 | 0.045 | -0.320 | 1.11 | BUREAU_OLDEST_DAYS | 0.199 |
| DAYS_ID_PUBLISH | 생활안정 | 0.037 | -0.190 | 1.09 | DAYS_EMPLOYED | 0.223 |
| CREDIT_TERM | 대출조건 | 0.143 | -0.271 | 1.07 | DAYS_EMPLOYED | 0.120 |
| PREV_REFUSED_RATIO | 신청행태 | 0.068 | -0.487 | 1.07 | INS_ALL_LATE_FLAG_MEAN | 0.143 |
| REGION_RATING_CLIENT_W_CITY | 인구통계·주변(대리 후보) | 0.052 | -0.422 | 1.06 | EXT_SOURCE_MEAN | 0.196 |
| PREV_APPROVED_AMT_RATIO | 신청행태 | 0.052 | -0.359 | 1.06 | PREV_REFUSED_RATIO | 0.126 |
| CREDIT_GOODS_RATIO | 대출조건 | 0.069 | -0.620 | 1.04 | CREDIT_TERM | 0.100 |

VIF 최대 3.27, 중앙값 1.22, 주의(≥5) 0개, 문제(≥10) 0개.

## 3. 제안 (적용하지 않음)

VIF가 5를 넘는 변수는 없다. 위 쌍 상관은 해석상 중복 신호이므로 필요하면 IV가 낮은 쪽을 후보로 삼을 수 있다.
- INS_ALL_PAY_RATIO_MEAN ↔ INS_ALL_LATE_FLAG_MEAN (상관 +0.72): IV가 낮은 INS_ALL_PAY_RATIO_MEAN (IV 0.055)가 후보.

## 4. 해석상 주의

- WoE 변환값의 상관은 구간화 결과에 의존한다. 원 변수 값의 상관과 다를 수 있다.
- VIF는 선형 모형(스코어카드)의 계수 안정성 지표이다. GBM에서는 같은 정보가 중복되어도 예측에는 문제가 없지만 SHAP 기여가 변수 사이로 나뉘어 해석이 흔들릴 수 있다.
- 계수가 모두 음수(부호 정상)라는 점은 이미 확인했다. VIF가 낮으면 이 부호가 다른 변수 때문에 억지로 만들어진 값은 아니라는 근거가 된다.
