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
├── results/     # 모델 산출물, 분석 결과표, 그림 (results/raw/는 커밋하지 않음)
└── docs/        # 인터페이스 정의, 변수 사전 등 문서
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY 값 입력
```

Gemini API 키는 Google AI Studio에서 발급합니다. `.env`는 절대 커밋하지 않습니다.

## 데이터

Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)에서 `application_train.csv`를 내려받아 `data/`에 넣습니다.

```
data/application_train.csv
```

`data/` 아래 파일은 `.gitignore`로 제외되어 레포에 올라가지 않습니다.

## 분석 5단계

1. **모델링**: 전처리, 로지스틱 회귀 baseline, XGBoost 학습, 확률 보정
2. **XAI**: TreeSHAP, Feature Importance·Permutation Importance 순위 비교, SHAP 부트스트랩
3. **LLM 설명**: Gemini로 SHAP 제공/미제공 두 조건의 한국어 설명 생성 및 파싱
4. **통계 분석**: 일치도 검정, 불일치 유형화, 집단·공정성 분석
5. **서비스**: Streamlit 개인 화면 + 분석 결과 탭

모델링 → LLM 단계의 입력 형식은 [docs/interface.md](docs/interface.md)를 따릅니다.
