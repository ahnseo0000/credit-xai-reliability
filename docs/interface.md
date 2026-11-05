# 모델링 → LLM 인터페이스

모델링(XAI) 단계가 LLM 설명 생성 단계에 넘기는 입력 형식이다. 이 형식이 고정되어 있으면 LLM 파이프라인은 가짜 SHAP 데이터로 먼저 개발할 수 있다.

## 파일

- 경로: `results/llm_input.csv`
- 인코딩: UTF-8
- 한 행 = 한 샘플 (층화 추출된 300건)

## 컬럼

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `sample_id` | int | 샘플 ID. 원본 `SK_ID_CURR` 값 |
| `pd` | float (0~1) | 보정된 부도확률 (isotonic 또는 Platt 보정 후) |
| `group_gender` | str | 성별 집단 (`CODE_GENDER`). **모형 입력에는 사용하지 않음**, 집단 분석용 |
| `group_age` | str | 연령 구간 (`DAYS_BIRTH` 기준) |
| `group_income` | str | 소득 구간 (`AMT_INCOME_TOTAL` 분위) |
| `feat__<변수명>` | 원본 타입 | 해당 변수의 원본 피처값 (전처리 전 값, 결측은 빈칸) |
| `shap__<변수명>` | float | 해당 변수의 SHAP 값. 보정 전 XGBoost의 **log-odds 기준** |

## 규칙

- `<변수명>`은 Home Credit 원본 영어 컬럼명을 그대로 쓴다. 예: `feat__EXT_SOURCE_2`, `shap__EXT_SOURCE_2`
- 모든 모형 입력 변수는 `feat__`와 `shap__` 컬럼이 한 쌍으로 존재해야 한다.
- 변수의 한국어 설명은 `docs/variable_dict.csv`에 둔다 (예: `EXT_SOURCE_2` → 외부 신용평가 점수 2).
- 보정은 단조변환이므로 log-odds 기준 SHAP의 변수 순위는 보정 후에도 유지된다.

## 예시

```csv
sample_id,pd,group_gender,group_age,group_income,feat__EXT_SOURCE_2,shap__EXT_SOURCE_2,feat__DAYS_BIRTH,shap__DAYS_BIRTH
100002,0.183,M,20-29,Q2,0.263,0.412,-9461,0.157
```
