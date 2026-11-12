# credit-xai-llm

Home Credit 데이터로 XGBoost 신용평가 모델을 만들고, Gemini가 생성한 한국어 설명이 SHAP 근거와 얼마나 일치하는지 통계적으로 검증한 뒤 Streamlit 서비스로 보여주는 프로젝트입니다.

- 팀: 안서영(팀장), 서민솔
- 발표: 11/17 (진리장학)

## 핵심 질문

> SHAP 정보 없이 생성한 LLM 설명은 모델 근거와 얼마나 어긋나고, SHAP을 주면 어떤 유형의 오류가 남는가?

## 현재 모델

**현재 모델은 v2(동결, 입력 변수 100개)** 입니다. 동결 산출물은 `results/v2_frozen/`(태그 `v2-frozen`)이고 사용법은 [results/v2_frozen/README.md](results/v2_frozen/README.md)에 있습니다. `results/v0_baseline/`과 `results/v1_*/`는 첫 모델과 비교 실험을 **재현·비교하기 위한 참고용**이며 서비스·설명 생성에는 v2를 씁니다.

## 폴더 구성

```
credit-xai-llm/
├── data/        # 원본 데이터 (커밋하지 않음)
├── notebooks/   # EDA·실험 노트북
├── src/         # 공용 파이썬 코드 (전처리, 모델, SHAP, LLM 호출, 분석)
├── app/         # Streamlit 앱
├── results/     # v0_baseline·v1_*(재현용 참고), l0/(변수 만들기), v2_frozen/(현재 모델), l2/(설명 산출물), 팀 전달용 l3_handoff/(git 제외) (*/raw/는 커밋하지 않음)
└── docs/        # 인터페이스 정의, 변수 사전 등 문서
```

## 설치

Python 3.13.1 기준입니다 (결과를 재현한 환경: Windows 11).

```bash
python3.13 -m venv .venv          # Windows: py -3.13 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # Windows: set PYTHONUTF8=1 (PowerShell: $env:PYTHONUTF8=1) 후 실행
cp .env.example .env              # GEMINI_API_KEY 값 입력
```

- **L3(LLM 설명 생성)에는 `google-genai`, `python-dotenv`, `pandas`만 필요합니다.** 나머지 패키지는 모델링·L0·L2 단계용입니다.
- **알려진 문제**: `shap` 0.51.0은 numba와 numpy 2.5 충돌로 import에 실패합니다(L3에는 불필요). Windows에서는 `requirements.txt`의 한글 주석 때문에 `PYTHONUTF8=1`이 필요합니다.

- macOS에서는 XGBoost가 OpenMP를 필요로 합니다: `brew install libomp`
- Gemini API 키는 Google AI Studio에서 발급합니다. `.env`는 절대 커밋하지 않습니다.
- 저장된 모델(`results/<버전>/models/*.joblib`)은 패키지 버전에 묶여 있으니 `requirements.txt`의 고정 버전을 그대로 쓰세요.

## 실행

```bash
python -m src.eda     # T1-1 EDA → results/<버전>/eda/
python -m src.train   # T1-2~8 전처리·분할·로지스틱·XGBoost·보정·비교·저장 (약 4분)
```

버전은 `src/config.py`의 `MODEL_VERSION`으로 정한다 (현재 `v2`). 위 두 명령은 v0 기준 첫 모델(재현용)을 만든다.

| 산출물 (`results/<버전>/` 아래) | 내용 |
| --- | --- |
| `eda/eda_summary.md` | 부도율, 결측 비율, 집단별 부도율, 주요 변수 분포 |
| `modeling/modeling_summary.md` | 성능 비교, 보정, 로지스틱 계수·Wald, VIF |
| `modeling/credit_metrics.md` | KS, Gini, 십분위 lift, ECE, 부트스트랩 AUC, Hosmer-Lemeshow, isotonic vs Platt, PSI |
| `models/xgb_bundle.joblib` | 전처리기 + XGBoost + Platt 보정기 + 더미→변수 매핑 (Streamlit·SHAP용) |
| `models/lr_pipeline.joblib` | 로지스틱 baseline 파이프라인 |
| `raw/` (git 제외) | 분할 ID(`splits.csv`), 테스트셋 예측값(`test_predictions.csv`) |

코드 구성: `src/config.py`(경로·상수), `src/data.py`(정제·파생 변수·집단·분할), `src/models.py`(전처리기·보정기·평가), `src/eda.py`, `src/train.py`.

### L0(보조 테이블 변수) · v2 동결 · L2(설명 산출물) 실행 명령

실행 기록은 `results/l0/RUNLOG.md`, `results/l2/RUNLOG.md`에 있습니다. 보조 테이블 CSV 6종이 `data/raw/`에 있어야 합니다. 큰 중간 산출물(`*.parquet`, `*/raw/`)은 git에 없어 아래 순서로 다시 만듭니다.

```bash
# L0 Tier 1: 보조 테이블 집계 변수 (단계별 ablation)
python -m src.l0_run --build-only        # 집계·EDA만 (학습 없음). 전체 실행: python -m src.l0_run
# L0 Tier 2: 타깃 EDA → 후보 대량 생성 → train/valid 선별 (테이블별 로컬 커밋)
python -m src.l0_tier2_run               # 특정 테이블: --tables installments_payments, 집계 재사용: --reuse
# v2 동결: 190개 모형 점검 → gain 중요도 K-sweep → 보정 확인·로지스틱·test 최종 1회·동결·태그
python -m src.l0_v2 --phase timing
python -m src.l0_v2 --phase sweep
python -m src.l0_v2 --phase final
# L2: 입력 캐시 → 300건 추출 → 전역 중요도(SHAP·내장 FI·Permutation·family) → LLM 입력 → SHAP 부트스트랩
python -m src.l2_common
python -m src.l2_sample
python -m src.l2_shap                    # Permutation을 재사용하려면 --reuse-perm
python -m src.l2_llm_input
python -m src.l2_boot --b-max 100        # 이어하기 가능, 시간이 걸린다(1회 약 4분)
# L3 팀 전달 패키지 (행 단위 데이터, zip은 git 제외)
python -m src.l3_handoff
```

L3 사양은 [docs/L3_SPEC.md](docs/L3_SPEC.md)에 있습니다.

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
