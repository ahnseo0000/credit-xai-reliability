# v2 동결 — 사용 안내

동결된 신용평가 모형(XGBoost, 입력 변수 100개)과 그 재현·사용에 필요한 것을 모아 둔 폴더다. 이 폴더의 모형·변수·전처리는 바꾸지 않는다 (git 태그 `v2-frozen`).

## 무엇이 들어 있나

| 파일 | 내용 |
| --- | --- |
| `models/xgb_v2.json` | XGBoost 모형. 나무 1,406개(조기 종료 지점)까지만 쓴다 — `predict` 때 `iteration_range=(0, best_iteration+1)` 또는 `XGBClassifier.predict_proba` 사용 |
| `models/xgb_preprocessor.joblib` | 신청서 변수 전처리 (train에서만 적합). 수치는 그대로, 범주는 원-핫 |
| `models/logit_champion.joblib` | 비교용 로지스틱 챔피언 (C=0.01, 결측 표시 열 포함) |
| `features.json` | 입력 변수 목록과 순서 (신청서 24 + L0 76 = 100), 제외된 변수 |
| `variable_dict_v2_application.csv`, `variable_dict_v2_l0.csv` | 변수 사전 (한국어 이름·설명·단위·family) |
| `config.json` | 시드·하이퍼파라미터·라이브러리 버전·분할 정보 |
| `test_final/` | test를 단 한 번 평가한 결과 (AUC 0.7900 / 로지스틱 0.7757). 이 결과로 어떤 결정도 하지 않았다 |
| `cell_sizes_18.csv` | 300건 추출용 18칸 크기표 |
| `k_sweep.*` | 변수 개수 K를 정한 근거 (valid만 사용) |
| `raw/test_predictions.csv` | test 고객별 확률 (git 제외) |

## 알아 둘 것

- **보정 없음**: Platt 보정은 valid에서 이득이 없어 쓰지 않는다. `pd`는 모형 출력 확률 그대로다.
- **성별·연령·소득은 모형 입력이 아니다.** 집단 분석용 값은 `results/l0/raw/group_vars.csv`에 따로 있다.
- **분할**: `SK_ID_CURR` 시드 42, 7:1.5:1.5 (train 215,257 / valid 46,127 / test 46,127).
- **L0 변수 재생성**: L0 변수는 보조 테이블 집계로 만들고(`results/l0/*.parquet`, git 제외), 재생성은 `python -m src.l0_features`, `python -m src.l0_tier2_run --reuse`.
- **안전 경로**: 문제가 생기면 태그 `l0-tier2-complete`의 190개 변수 모형(`results/l0_v2/raw/model_190.json`)이 이전 동결 후보다.

## 쓰는 법 (예측)

```python
import json, joblib, numpy as np, xgboost as xgb
feat = json.load(open("results/v2_frozen/features.json", encoding="utf-8"))
pre = joblib.load("results/v2_frozen/models/xgb_preprocessor.joblib")
model = xgb.XGBClassifier(); model.load_model("results/v2_frozen/models/xgb_v2.json")
# X = [pre.transform(신청서 변수 24개) ; L0 변수 76개]  (열 순서는 features.json)
# pd = model.predict_proba(X)[:, 1]
```

L2 이후 산출물(설명·SHAP·300건 추출)은 `results/l2/`에 있고, 이 폴더를 읽기만 한다.
