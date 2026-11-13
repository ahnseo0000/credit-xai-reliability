# Phase 2 추가: 외부점수 제외 모델 (valid)

- **해석 기준(결과 보기 전에 RUNLOG에 기록): valid AUC 하락폭, 상위 SHAP 요인과 family 변화를 기술적으로 보고한다. 우열 판정이나 가설 검정은 하지 않는다.**
- train 215,257건으로 학습, valid 46,127건으로 평가. **test는 읽지 않았다.** v2 동결 파일(모델, 변수 목록, 하이퍼파라미터)은 읽기만 했다.
- 제외한 변수 7개: EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_MEAN, EXT_SOURCE_MIN, EXT_SOURCE_MAX, EXT_SOURCE_PROD (남은 변수 93개). 이름에 EXT_SOURCE가 들어간 변수가 전부이며, 외부 점수를 재료로 쓰는 비율·상호작용 변수는 v2 100개 안에 없었다 (`excluded_vars.csv`).
- 새 모형은 v2와 같은 하이퍼파라미터·시드 42·scale_pos_weight 1.0·조기 종료 150(valid AUC), 튜닝 없음. best_iteration 1154 (v2 1405).

## 1. valid 지표

| 모형 | 변수 | AUC | Gini | KS | Brier | ECE(10분위) | 1분위 lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 (100변수) | 100 | 0.7898 | 0.5796 | 0.4390 | 0.0657 | 0.0024 | 3.80 |
| 외부점수 제외 (93변수) | 93 | 0.7685 | 0.5369 | 0.4020 | 0.0673 | 0.0033 | 3.52 |
| 차이 (제외 − v2) | | -0.0213 | -0.0427 | -0.0370 | +0.0016 | +0.0008 | -0.28 |

- **AUC 차이(제외 − v2) = -0.0213, 95% 구간 [-0.0254, -0.0174]** (같은 재표본 쌍 부트스트랩, B=1000, 시드 42). AUC 하락 0.0213는 v2 AUC의 2.7%, 모형 AUC 구간: v2 [0.7827, 0.7970], 제외 [0.7610, 0.7759].
- 참고(스코어카드·22변수 GBM과의 위치): 외부점수를 쓰는 22변수 GBM AUC 0.7749, 스코어카드 0.7544 (compare_valid_v2.md). 외부점수 제외 모형의 AUC는 0.7685.

## 2. SHAP 상위 15개 변수 (valid, 정확한 TreeSHAP, mean|SHAP| = log-odds 단위)

SHAP은 L2와 같은 방식이지만 test 대신 valid 46,127건에서 계산했다 (두 모형 모두). 변수별 값은 원-핫 열을 합친 것이다.

| 순위 | v2 변수 | family | mean abs SHAP | 새 모형 순위 | | 순위 | 새 모형 변수 | family | mean abs SHAP | v2 순위 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | EXT_SOURCE_MEAN (제외됨) | 외부점수 | 0.3678 | 제외 | | 1 | DAYS_EMPLOYED | 직업·고용 | 0.1611 | 5 |
| 2 | CREDIT_TERM | 대출조건 | 0.1162 | 3 | | 2 | BUR_ALL_DEBT_RATIO_MAX | 소진율 | 0.1298 | 12 |
| 3 | AMT_ANNUITY | 대출조건 | 0.1091 | 8 | | 3 | CREDIT_TERM | 대출조건 | 0.1238 | 2 |
| 4 | POS_ALL_FUT_RATIO_MEAN | 상환비율 | 0.1080 | 6 | | 4 | CREDIT_GOODS_RATIO | 대출조건 | 0.1164 | 6 |
| 5 | DAYS_EMPLOYED | 직업·고용 | 0.1071 | 1 | | 5 | NAME_EDUCATION_TYPE | 인구통계·주변(대리 후보) | 0.1093 | 9 |
| 6 | CREDIT_GOODS_RATIO | 대출조건 | 0.0937 | 4 | | 6 | POS_ALL_FUT_RATIO_MEAN | 상환비율 | 0.1031 | 4 |
| 7 | EXT_SOURCE_1 (제외됨) | 외부점수 | 0.0881 | 제외 | | 7 | PREV_APPROVED_AMT_RATIO | 신청행태 | 0.0941 | 8 |
| 8 | PREV_APPROVED_AMT_RATIO | 신청행태 | 0.0875 | 7 | | 8 | AMT_ANNUITY | 대출조건 | 0.0932 | 3 |
| 9 | NAME_EDUCATION_TYPE | 인구통계·주변(대리 후보) | 0.0869 | 5 | | 9 | BUREAU_MAX_OVERDUE_AMT | 연체이력 | 0.0894 | 28 |
| 10 | EXT_SOURCE_MAX (제외됨) | 외부점수 | 0.0841 | 제외 | | 10 | DAYS_LAST_PHONE_CHANGE | 생활안정 | 0.0862 | 54 |
| 11 | INS_ALL_INST_AMT_MEAN | 부채규모 | 0.0808 | 14 | | 11 | REGION_RATING_CLIENT_W_CITY | 인구통계·주변(대리 후보) | 0.0846 | 27 |
| 12 | BUR_ALL_DEBT_RATIO_MAX | 소진율 | 0.0777 | 2 | | 12 | OCCUPATION_TYPE | 직업·고용 | 0.0845 | 16 |
| 13 | POS_ACTIVE_CNT | 이용규모 | 0.0729 | 16 | | 13 | PREV_REFUSED_RATIO | 신청행태 | 0.0821 | 19 |
| 14 | OWN_CAR_AGE | 생활안정 | 0.0728 | 18 | | 14 | INS_ALL_INST_AMT_MEAN | 부채규모 | 0.0800 | 11 |
| 15 | EXT_SOURCE_3 (제외됨) | 외부점수 | 0.0642 | 제외 | | 15 | BUREAU_OLDEST_DAYS | 신용이력길이 | 0.0800 | 74 |

- **v2 상위 10위 변수 중 새 모형에서도 상위 10위 안: 7개 / 10개.** v2 상위 10위 중 제외된(외부점수) 변수 3개(EXT_SOURCE_MEAN, EXT_SOURCE_1, EXT_SOURCE_MAX)는 당연히 빠졌고, 남은 7개 중 새 상위 10위에 남은 변수는 7개: CREDIT_TERM, AMT_ANNUITY, POS_ALL_FUT_RATIO_MEAN, DAYS_EMPLOYED, CREDIT_GOODS_RATIO, PREV_APPROVED_AMT_RATIO, NAME_EDUCATION_TYPE.
- v2 상위 10위 중 새 모형에서 10위 밖으로 간 변수(제외된 변수 제외): 없음
- 새 모형 상위 10위에 새로 들어온 변수(v2 상위 10위 밖): BUR_ALL_DEBT_RATIO_MAX(12위→2위), BUREAU_MAX_OVERDUE_AMT(28위→9위), DAYS_LAST_PHONE_CHANGE(54위→10위)
- 참고: valid로 다시 계산한 v2 상위 10위와 L2(test 기준) v2 상위 10위의 겹침은 10/10 (L2 상위 10위: EXT_SOURCE_MEAN, CREDIT_TERM, AMT_ANNUITY, POS_ALL_FUT_RATIO_MEAN, DAYS_EMPLOYED, CREDIT_GOODS_RATIO, EXT_SOURCE_1, NAME_EDUCATION_TYPE, PREV_APPROVED_AMT_RATIO, EXT_SOURCE_MAX).
- 새 모형 mean|SHAP| 합 3.605, v2 3.729. v2에서 외부점수 7개 변수의 mean|SHAP| 합은 0.735 (19.7%), `EXT_SOURCE_MEAN` 단독 0.368.

## 3. family별 평균 |SHAP| (valid, 행별 family 합의 mean|.|; L2 family_importance와 같은 정의)

| family | 변수 수 (v2→새) | v2 값 | v2 비중 | v2 순위 | 새 값 | 새 비중 | 새 순위 | 값 변화 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 외부점수 | 7→0 | 0.5390 | 25.7% | 1 | - | - | - | - |
| 대출조건 | 5→5 | 0.2147 | 10.2% | 2 | 0.2246 | 11.4% | 3 | +0.0099 |
| 연체이력 | 22→22 | 0.2098 | 10.0% | 3 | 0.2506 | 12.7% | 2 | +0.0408 |
| 소진율 | 16→16 | 0.1906 | 9.1% | 4 | 0.2648 | 13.4% | 1 | +0.0741 |
| 직업·고용 | 4→4 | 0.1688 | 8.0% | 5 | 0.2235 | 11.3% | 4 | +0.0547 |
| 신청행태 | 8→8 | 0.1389 | 6.6% | 6 | 0.1719 | 8.7% | 6 | +0.0330 |
| 인구통계·주변(대리 후보) | 5→5 | 0.1295 | 6.2% | 7 | 0.1916 | 9.7% | 5 | +0.0621 |
| 상환비율 | 10→10 | 0.1199 | 5.7% | 8 | 0.1193 | 6.0% | 10 | -0.0006 |
| 부채규모 | 8→8 | 0.1135 | 5.4% | 9 | 0.1197 | 6.1% | 9 | +0.0063 |
| 이용규모 | 6→6 | 0.1109 | 5.3% | 10 | 0.1370 | 6.9% | 8 | +0.0261 |
| 생활안정 | 4→4 | 0.0945 | 4.5% | 11 | 0.1377 | 7.0% | 7 | +0.0432 |
| 소득·상환부담 | 4→4 | 0.0526 | 2.5% | 12 | 0.0527 | 2.7% | 12 | +0.0001 |
| 신용이력길이 | 1→1 | 0.0149 | 0.7% | 13 | 0.0800 | 4.1% | 11 | +0.0650 |

- 값은 SHAP이 log-odds 척도라 합이 1이 아니다. 비중은 family 합 값의 비중이다. 외부점수 family는 새 모형에서 사라졌다 (변수 7→0). 다른 family의 값은 외부점수가 하던 일을 다른 변수가 나눠 받은 몫을 포함한다.
- 변수별 mean|SHAP|의 단순 합(상쇄 없음) 열은 `family_shap.csv`에 같이 저장했다.

## 4. 한계

- 이 문서는 기술 보고이다. 우열 판정이나 가설 검정은 하지 않았다.
- 단일 학습·단일 valid 분할의 결과이고 튜닝은 하지 않았다 (v2와 같은 조건 유지). 조기 종료에 valid를 쓰는 점도 v2와 같다.
- SHAP 순위는 두 모형의 학습 방식 차이(외부점수가 없을 때 다른 변수가 대신 쓰이는 정도)를 포함한다. 순위 변화가 곧 해당 변수의 인과적 중요도 변화는 아니다.
- SHAP은 test가 아니라 valid에서 계산했다 (test 사용 금지). L2의 test 기준 값과 완전히 같지는 않다.
