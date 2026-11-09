# 모델링 결과 (T1-3 ~ T1-7)

`python -m src.train`으로 생성.

## 분할

- TARGET 층화 7:1.5:1.5, `random_state=42`: train 215,257 / valid 46,127 / test 46,127
- 소득 3분위 경계값(학습셋): q1 = 117,000.0, q2 = 180,000.0

## 성능 비교 (테스트셋)

컷오프는 검증셋에서 F1이 최대가 되는 값으로 정해 테스트셋에 적용했다. 부도율 8%라 0.5 컷오프는 쓰지 않는다.

| 모델 | AUC | 정확도 | F1 | Brier | 컷오프 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 로지스틱 회귀 | 0.7491 | 0.8466 | 0.2920 | 0.0687 | 0.150 | 37,593 | 4,810 | 2,265 | 1,459 |
| XGBoost (보정 후) | 0.7695 | 0.8651 | 0.3146 | 0.0672 | 0.170 | 38,478 | 3,925 | 2,296 | 1,428 |

## XGBoost

- `scale_pos_weight` = 1.00 (가중치 없음), early stopping 150라운드 (검증셋 AUC)
- best iteration = 1054, 검증셋 AUC = 0.7648
- 하이퍼파라미터는 고정값 (5-fold 튜닝은 범위에서 제외): `{'n_estimators': 3000, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 20, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 5.0, 'tree_method': 'hist', 'early_stopping_rounds': 150, 'random_state': 42}`
- 결측은 XGBoost가 자체 처리, 범주형은 원-핫 (100건 미만 범주는 infrequent로 묶음)

## 확률 보정 (T1-6)

- 검증셋 안 5-fold Brier: isotonic 0.06733, Platt 0.06732
- **platt** 채택, 검증셋 전체로 다시 학습. isotonic은 계단함수라 부도확률이 정확히 0%·100%인 고객과 동점이 생긴다 (첫 실행에서 테스트셋 고유값 103개, 0%가 72명). 사용자·LLM에 보여줄 값으로 부적절하고 Brier 차이도 미미해 Platt을 쓴다.
- 검증셋을 early stopping과 보정에 함께 썼다. 테스트셋은 어디에도 쓰지 않았다.
- TreeSHAP은 보정 전 XGBoost의 log-odds 기준이다. 보정은 단조변환이라 고객 간 순위는 유지된다.

| 모델 | 테스트 Brier | 평균 예측 | 실제 부도율 |
| --- | --- | --- | --- |
| XGBoost 보정 전 | 0.06718 | 0.0805 | 0.0807 |
| XGBoost 보정 후 (platt) | 0.06718 | 0.0806 | 0.0807 |
| 로지스틱 | 0.06867 | 0.0808 | 0.0807 |

그림: `calibration_curve.png`

## 로지스틱 baseline (T1-4)

- 전처리: 0.5%/99.5% 윈저화(고유값 50개 초과 연속형만) → 중앙값 대치 → 표준화, 범주형은 원-핫(첫 범주 제거). 계수는 표준화 단위.
- 유의(p<0.05) 계수 64개 / 전체 129개. 전체 표: `logit_coefficients.csv`
- Wald 검정에서 뺀 컬럼 (예측용 모형에는 포함):
  - 1이 50건 미만인 희소 더미 (완전분리로 계수 발산): `NAME_INCOME_TYPE_infrequent_sklearn`, `NAME_FAMILY_STATUS_infrequent_sklearn`
  - 완전 선형종속 (피벗 QR): `NAME_FAMILY_STATUS_Separated`

|z| 상위 15개:

| 컬럼 | 한국어 | 계수 | 오즈비 | z | p |
| --- | --- | --- | --- | --- | --- |
| `EXT_SOURCE_3` | 외부 신용평가 점수 3 | -0.457 | 0.633 | -56.6 | 0.0e+00 |
| `EXT_SOURCE_2` | 외부 신용평가 점수 2 | -0.390 | 0.677 | -47.9 | 0.0e+00 |
| `EXT_SOURCE_1` | 외부 신용평가 점수 1 | -0.168 | 0.845 | -19.7 | 3.4e-86 |
| `AMT_GOODS_PRICE` | 구매 상품 가격 | -0.953 | 0.385 | -16.9 | 3.5e-64 |
| `DAYS_EMPLOYED` | 근속 기간 | +0.147 | 1.158 | +13.5 | 1.6e-41 |
| `CREDIT_TERM` | 상환 속도 | -0.199 | 0.820 | -12.0 | 3.8e-33 |
| `FLAG_OWN_CAR` | 자동차 보유 | -0.105 | 0.900 | -11.2 | 6.9e-29 |
| `AMT_CREDIT` | 대출 금액 | +0.699 | 2.011 | +10.7 | 1.0e-26 |
| `OCCUPATION_TYPE_Drivers` | 직업 | +0.593 | 1.809 | +8.7 | 3.5e-18 |
| `FLAG_WORK_PHONE` | 추가 전화 제공 | +0.075 | 1.077 | +8.5 | 1.9e-17 |
| `OCCUPATION_TYPE_Low-skill Laborers` | 직업 | +0.773 | 2.167 | +8.2 | 2.3e-16 |
| `ANNUITY_INCOME_RATIO` | 소득 대비 상환액 (DSR 유사) | +0.225 | 1.253 | +7.4 | 1.7e-13 |
| `OCCUPATION_TYPE_Laborers` | 직업 | +0.462 | 1.587 | +7.3 | 3.0e-13 |
| `DAYS_ID_PUBLISH` | 신분증 재발급 경과일 | +0.056 | 1.057 | +6.4 | 2.1e-10 |
| `NAME_CONTRACT_TYPE_Revolving loans` | 대출 유형 | -0.256 | 0.775 | -6.3 | 3.5e-10 |

## VIF 진단

VIF > 10인 컬럼 39개 (5만 건 표본). 전체 표: `vif.csv`

주요 원인 (구조적 중복이라 예상된 결과):

- `ORGANIZATION_TYPE_XNA`, `FLAG_EMP_PHONE`, `NAME_INCOME_TYPE_Pensioner`, `DAYS_EMPLOYED`: 연금 수령·무직자를 서로 다른 방식으로 표시한 같은 정보
- `CNT_FAM_MEMBERS` = `CNT_CHILDREN` + (기혼이면 2, 아니면 1): 혼인 상태 더미와 합쳐 완전 선형종속. Wald 검정에서는 피벗 QR로 종속 컬럼 하나를 자동으로 뺐다.
- `OBS_30` ↔ `OBS_60`, `REGION_RATING_CLIENT` ↔ `_W_CITY`: 거의 같은 변수 쌍
- `AMT_CREDIT` ↔ `AMT_GOODS_PRICE` ↔ 파생 비율 변수: 대출 금액과 상품 가격은 거의 같다. 그래서 로지스틱에서 두 계수 부호가 반대로 나온다 (개별 계수 해석 불가).
- 범주형 더미: 기준 범주가 작을 때 생기는 더미 간 VIF

로지스틱 계수는 이 변수들에 대해 개별 해석하지 않는다. XGBoost는 다중공선성에 예측 성능은 영향을 덜 받지만, SHAP 기여도가 상관된 변수끼리 나뉠 수 있다 (`docs/interface.md`의 `var_group` 참고).

| 컬럼 | VIF |
| --- | --- |
| `CNT_CHILDREN` | ∞ (완전 선형종속) |
| `NAME_FAMILY_STATUS_Separated` | ∞ (완전 선형종속) |
| `CNT_FAM_MEMBERS` | ∞ (완전 선형종속) |
| `NAME_FAMILY_STATUS_Widow` | ∞ (완전 선형종속) |
| `NAME_FAMILY_STATUS_Single / not married` | ∞ (완전 선형종속) |
| `ORGANIZATION_TYPE_XNA` | 8,712.4 |
| `FLAG_EMP_PHONE` | 7,395.2 |
| `NAME_INCOME_TYPE_Pensioner` | 1,184.2 |
| `NAME_EDUCATION_TYPE_Secondary / secondary special` | 356.9 |
| `NAME_EDUCATION_TYPE_Higher education` | 318.6 |
| `OBS_60_CNT_SOCIAL_CIRCLE` | 318.4 |
| `OBS_30_CNT_SOCIAL_CIRCLE` | 318.0 |
| `ORGANIZATION_TYPE_Business Entity Type 3` | 156.8 |
| `ORGANIZATION_TYPE_Self-employed` | 97.7 |
| `AMT_CREDIT` | 58.1 |
| `NAME_EDUCATION_TYPE_Incomplete higher` | 57.7 |
| `ORGANIZATION_TYPE_Other` | 46.9 |
| `AMT_GOODS_PRICE` | 43.1 |
| `ORGANIZATION_TYPE_Medicine` | 32.9 |
| `ORGANIZATION_TYPE_Business Entity Type 2` | 30.8 |
| `ORGANIZATION_TYPE_Government` | 29.5 |
| `ORGANIZATION_TYPE_School` | 25.7 |
| `NAME_HOUSING_TYPE_House / apartment` | 25.5 |
| `ORGANIZATION_TYPE_Trade: type 7` | 23.5 |
| `NAME_EDUCATION_TYPE_Lower secondary` | 23.5 |
| `ORGANIZATION_TYPE_Kindergarten` | 20.6 |
| `ORGANIZATION_TYPE_Construction` | 19.2 |
| `ORGANIZATION_TYPE_Business Entity Type 1` | 18.4 |
| `ORGANIZATION_TYPE_Transport: type 4` | 17.0 |
| `CREDIT_INCOME_RATIO` | 16.5 |
| `ANNUITY_INCOME_RATIO` | 13.9 |
| `AMT_ANNUITY` | 13.1 |
| `NAME_HOUSING_TYPE_With parents` | 12.6 |
| `ORGANIZATION_TYPE_Industry: type 9` | 11.1 |
| `REGION_RATING_CLIENT_W_CITY` | 11.1 |
| `REGION_RATING_CLIENT` | 11.0 |
| `ORGANIZATION_TYPE_Security` | 11.0 |
| `ORGANIZATION_TYPE_Trade: type 3` | 10.9 |
| `ORGANIZATION_TYPE_Industry: type 3` | 10.6 |
