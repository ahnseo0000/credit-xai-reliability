# credit-xai-llm

Home Credit 데이터로 XGBoost 신용평가 모델을 만들고, Gemini가 생성한 한국어 설명이 SHAP 근거와 얼마나 일치하는지 통계적으로 검증한 뒤 Streamlit 서비스로 보여주는 프로젝트입니다.

- 팀: 안서영(팀장), 서민솔
- 발표: 11/17 (진리장학)

## 핵심 질문

> SHAP 정보 없이 생성한 LLM 설명은 모델 근거와 얼마나 어긋나고, SHAP을 주면 어떤 유형의 오류가 남는가?

## 폴더 구성

```
credit-xai-llm/
├── data/        # 원본 데이터 (커밋하지 않음)
├── notebooks/   # EDA·실험 노트북
├── src/         # 공용 파이썬 코드 (전처리, 모델, SHAP, LLM 호출, 분석)
├── app/         # Streamlit 앱
├── results/     # 모델 버전별 폴더(v0_baseline/ ...): 산출물, 결과표, 그림 (*/raw/는 커밋하지 않음)
└── docs/        # 인터페이스 정의, 변수 사전 등 문서
```

## 설치

Python 3.12 기준입니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY 값 입력
```

- macOS에서는 XGBoost가 OpenMP를 필요로 합니다: `brew install libomp`
- Gemini API 키는 Google AI Studio에서 발급합니다. `.env`는 절대 커밋하지 않습니다.
- 저장된 모델(`results/<버전>/models/*.joblib`)은 패키지 버전에 묶여 있으니 `requirements.txt`의 고정 버전을 그대로 쓰세요.

## 실행

```bash
python -m src.eda     # T1-1 EDA → results/<버전>/eda/
python -m src.train   # T1-2~8 전처리·분할·로지스틱·XGBoost·보정·비교·저장 (약 4분)
```

버전은 `src/config.py`의 `MODEL_VERSION`으로 정한다 (현재 `v0_baseline`).

| 산출물 (`results/<버전>/` 아래) | 내용 |
| --- | --- |
| `eda/eda_summary.md` | 부도율, 결측 비율, 집단별 부도율, 주요 변수 분포 |
| `modeling/modeling_summary.md` | 성능 비교, 보정, 로지스틱 계수·Wald, VIF |
| `modeling/credit_metrics.md` | KS, Gini, 십분위 lift, ECE, 부트스트랩 AUC |
| `models/xgb_bundle.joblib` | 전처리기 + XGBoost + Platt 보정기 + 더미→변수 매핑 (Streamlit·SHAP용) |
| `models/lr_pipeline.joblib` | 로지스틱 baseline 파이프라인 |
| `raw/` (git 제외) | 분할 ID(`splits.csv`), 테스트셋 예측값(`test_predictions.csv`) |

코드 구성: `src/config.py`(경로·상수), `src/data.py`(정제·파생 변수·집단·분할), `src/models.py`(전처리기·보정기·평가), `src/eda.py`, `src/train.py`.

## 데이터

Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)에서 `application_train.csv`(와 이후 단계의 보조 테이블)를 내려받아 `data/raw/`에 넣습니다.

```
data/raw/application_train.csv
```

`data/` 아래 파일은 `.gitignore`로 제외되어 레포에 올라가지 않습니다.

## 분석 5단계

1. **모델링**: 전처리, 로지스틱 회귀 baseline, XGBoost 학습, 확률 보정
2. **XAI**: TreeSHAP, Feature Importance·Permutation Importance 순위 비교, SHAP 부트스트랩
3. **LLM 설명**: Gemini로 SHAP 제공/미제공 두 조건의 한국어 설명 생성 및 파싱
4. **통계 분석**: 일치도 검정, 불일치 유형화, 집단·공정성 분석
5. **서비스**: Streamlit 개인 화면 + 분석 결과 탭

모델링 → LLM 단계의 입력 형식은 [docs/interface.md](docs/interface.md)를 따릅니다.
