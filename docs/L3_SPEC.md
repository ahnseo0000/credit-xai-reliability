# L3 인수 사양서 (설명 생성)

L2 산출물(모형·SHAP·300건)을 받아 LLM 설명을 만드는 쪽을 위한 문서다. 근거는 `docs/PROJECT_PLAN.md`의 L3·L4 항목과 `docs/interface.md`이다.
**[확인 필요]** 표시는 계획서에 정해진 값이 없어 인수 측이 정하거나 팀이 확정해야 하는 항목이다. 이 문서는 기본값을 제안할 뿐이다.

## 0. 받는 것

| 파일 | 내용 |
| --- | --- |
| `llm_input_300_full.jsonl` | 300명, 한 줄 = 한 명. 아래 1절 구조 (**행 단위 데이터: git에 올리지 않는다, zip으로만 전달**) |
| `variable_dict.csv`, `variable_dict_v2_l0.csv` | 한국어 변수 사전 (사실성·환각 대조용) |

모형: v2 동결 XGBoost (입력 100개, 태그 `v2-frozen`). `pd`는 보정 없는 모형 확률이다. SHAP 단위는 log-odds이고 `기준값 + 변수별 SHAP 합 = 모형 log-odds`가 성립한다 (파일 생성 시 300명 모두 확인).

## 1. 입력 레코드 구조

```
sample_id
cell            { cell_id, pd_bin(저/경계/고), weight }
model_output    { pd, base_value_logodds, shap_sum_logodds, cutoff_rule, cutoff_pd, decision(거절/승인) }
features[100]   |SHAP| 큰 순서. { abs_rank, variable, korean_name, family, description, unit,
                  value(원값, 결측은 null), value_text(읽기용), shap, direction }
applicant_info  { group_gender, group_age, age_years, group_income }   ← RQ6용, 프롬프트에 넣지 않는다
analysis_only   { target }                                              ← 프롬프트에 넣지 않는다
```

- **임시 거절선**: test 전체(46,127건) `pd`의 q80 = `cutoff_pd`. `pd >= cutoff_pd`이면 `거절`, 아니면 `승인`. L5에서 손익 기반 거절선이 정해지면 바뀐다 (그때 `decision`만 다시 계산하면 된다).
- **`value_text` 규칙**: 날짜 기준 변수(`DAYS_EMPLOYED` 등)는 "약 N년 전/후" (30일 미만 "N일", 1년 미만 "약 N개월"), 결측은 "정보 없음", `DAYS_EMPLOYED` 결측은 "재직 정보 없음". 지연일수·간격 같은 *기간* 변수는 시점이 아니므로 "N일 지연/일찍 납부"처럼 쓴다. 범주형은 원본 값(영문) 그대로이다.
- **`DAYS_EMPLOYED` 결측의 뜻**: 원본의 무직·연금 수령 표시값(365243)을 결측으로 바꾼 것이다. 정말 정보가 없는 것이 아니다. 설명에 "재직 정보 없음"으로 나온다.
- **방향**: `shap > 0`이면 부도 위험 증가(거절 쪽), `< 0`이면 감소. 절댓값 1e-4 이하는 "영향 거의 없음".
- `weight` = (test의 칸 크기) / (추출 건수). 전체 모집단 추정에만 쓰고, 집단·구간 간 비교는 가중치 없이 한다.
- `applicant_info`는 모형 입력이 아니다. 성별·연령은 RQ6(보호변수 언급률)과 L4 집단 분석에서만 쓴다.

## 2. 실험 조건

| 조건 | 프롬프트에 주는 것 |
| --- | --- |
| **A. SHAP 미제공** | `pd`, `decision`, `cutoff_pd`, 변수 100개의 `korean_name`·`value_text` (값 전체, 순서는 변수 사전 순서로 섞어 SHAP 순위가 새지 않게) |
| **B. SHAP 제공** | A + 각 변수의 `shap`·`direction` + 기준값 (|SHAP| 순서) |

- 본 실험: 300명 × 2조건 = 600건 (계획서 L3-6). 두 조건의 프롬프트 문구 차이는 SHAP 관련 문장 외에는 없게 한다.
- 파일럿(L3-2): 프롬프트 3종 × 모델 2종, 50건. **[확인 필요]** 모델 2종과 프롬프트 3종의 구체 내용, 파일럿 50건 선택 방법(300건 일부 사용 여부).
- 자기일관성(L3-7): 동일 입력 3회 반복 (반복 번호를 캐시 키에 넣는다).
- 두 조건 모두 `applicant_info`와 `target`은 넣지 않는다. **[확인 필요]** RQ6를 "성별·연령을 주지 않았는데도 언급하는가"로 볼지, 일부 조건에서 주고 볼지 팀 합의가 필요하다 (기본값: 주지 않음).

## 3. 출력 JSON 스키마 (제36조의2 제1항 4요소)

LLM이 아래 JSON 하나만 출력하게 한다 (설명 문장 한국어).

```json
{
  "result": {"decision": "거절|승인", "summary": "평가 결과 한 문장"},
  "main_reasons": [
    {"rank": 1, "variable": "변수 사전의 variable 이름", "direction": "위험 증가|위험 감소", "reason_text": "고객에게 보이는 문장"}
  ],
  "basis_info_overview": "사용된 기초정보 개요 (어떤 종류의 정보를 봤는지)",
  "improvement_direction": [{"variable": "...", "suggestion": "..."}],
  "other_factors": [{"name": "목록 밖에서 언급한 요인", "reason_text": "..."}]
}
```

- `main_reasons`는 정확히 4개 (주요 판단 기준 상위 4개), 방향 포함.
- `main_reasons[].variable`은 입력에 있는 변수 이름만 쓰게 지시한다. 목록에 없는 요인은 `other_factors`에 쓰게 하고, **제거하지 않는다** (환각 측정용, L3-3).
- `improvement_direction`은 L2-10 반사실과 대조하기 위한 필드이다. 바꿀 수 없는 변수(나이·성별 등)에 대한 제안 여부는 L4 실행가능성 축에서 평가한다.
- 스키마는 `decision`이 입력의 `decision`과 같은지, `main_reasons`가 4개인지, 필수 키가 있는지를 검사한다.

## 4. 함수 인터페이스 (제안)

```python
# src/l3_llm.py
def build_prompt(record: dict, condition: str, prompt_id: str) -> str:
    """condition = 'A' | 'B'. 2절 규칙대로 record에서 허용된 필드만 꺼내 프롬프트를 만든다 (applicant_info, target 사용 금지)."""

def call_llm(prompt: str, model: str, repeat_idx: int = 0, *, use_cache: bool = True) -> dict:
    """반환: {'text': 원문, 'cached': bool, 'latency_s': float, 'attempts': int, 'tokens_in': int, 'tokens_out': int, 'error': str | None}"""

def parse_output(text: str) -> tuple[dict | None, list[str]]:
    """JSON 추출·파싱 + 스키마 검사. 반환: (파싱 결과 또는 None, 위반 목록)"""

def postprocess(parsed: dict, record: dict, var_dict: pd.DataFrame) -> dict:
    """5절 후처리 결과(변수 매칭, 환각 후보, 보호변수 언급, 수치 대조)를 붙여 돌려준다."""

def run_batch(records: list[dict], condition: str, model: str, prompt_id: str, repeats: int = 1) -> pd.DataFrame:
    """레코드 × 반복을 돌리고 결과 표(한 행 = 한 호출)를 results/l3/ 에 저장한다."""
```

## 5. 호출 설정

- `temperature = 0`, 최대 출력 토큰·`top_p` 등 나머지 설정도 기록한다. **[확인 필요]** 모델 이름과 버전. 사용 SDK는 `requirements.txt`의 `google-genai`(Gemini)이다.
- **캐시**: 키 = sha256(모델, 설정, 프롬프트 전문, 반복 번호). 같은 키는 API를 다시 부르지 않는다. 디스크에 jsonl 또는 sqlite로 둔다 (`results/l3/cache/`, git 제외).
- **재시도**: 429·5xx·네트워크 오류는 지수 백오프(최대 3~5회). 파싱·스키마 실패는 같은 프롬프트로 최대 2회 재시도하고 시도 횟수를 기록한다. 재시도해도 실패한 건은 빼지 않고 `failed`로 남긴다.
- **호출 간격**: 최소 간격(예: 1초 이상, 요금제 한도에 맞춤)을 둔다. **[확인 필요]**
- **기록**: 호출마다 시각, 모델, 프롬프트 id, 조건, 샘플 id, 반복 번호, 캐시 여부, 지연, 시도 횟수, 토큰 수, 원문 응답, 오류를 남긴다. 원문은 지우지 않는다.

## 6. 후처리 (L3-4, L4 입력)

1. JSON 파싱, 스키마 위반 목록 기록.
2. `main_reasons[].variable`과 `other_factors[].name`을 변수 사전의 `variable`·`korean_name`에 대조해 매칭 (완전 일치 → 한국어 이름 일치 → 필요하면 임베딩 유사도). 매칭 실패는 **환각 후보**로 표시만 하고 지우지 않는다.
3. 보호변수 언급(성별·연령·지역) 탐지: `reason_text`·`other_factors`·`summary` 전체에서 키워드로 찾는다 (RQ6). 키워드 목록은 결과를 보기 전에 고정한다.
4. 사실성: 설명 속 수치를 `features[].value_text`·`value`와 대조.
5. 충실성(L4): 조건 B에서 LLM이 든 상위 4개와 `abs_rank` 순위 비교. 목록에 없는 변수는 최하위 동순위.

## 7. 시험 10건 (L3-5) — 본 실험 전에 반드시

- 실제 SHAP 대신 **가짜 SHAP**(예: 같은 레코드의 SHAP 값을 변수 사이에서 무작위로 섞은 것)으로 10건을 조건 B로 돌린다. 목적은 (1) 파이프라인과 스키마가 도는지, (2) 모형이 준 SHAP이 아니라 가짜 SHAP을 따라 설명하는지(입력을 따르는지) 확인하는 것이다.
- 보고: 파싱 실패율, 스키마 위반율, 재시도 횟수, 호출당 지연·토큰, 가짜 SHAP 상위 4개와 응답 상위 4개 일치도.
- 통과 기준 **[확인 필요]** (제안: 재시도 후 파싱 실패 0건, 스키마 위반 1건 이하). 통과 전에는 600건 본 실험을 시작하지 않는다.

## 8. 보안 규칙

- `GEMINI_API_KEY`는 `.env`에서만 읽는다 (`python-dotenv`). 코드·노트북·로그·캐시·커밋 메시지에 키를 쓰지 않는다. 요청 헤더나 환경 변수 전체를 출력하지 않는다. `.env`는 `.gitignore`에 있다.
- 행 단위 데이터(`llm_input_300_full.jsonl`, `results/l3_handoff/`, `results/l2/sample_300*`, 호출 원문·캐시)는 git에 올리지 않는다. zip으로 전달한다. 키 유출이 의심되면 즉시 폐기·재발급한다.
- 외부 API로 보내는 것은 공개 데이터셋(Home Credit)의 변수값·모형 출력뿐이다. 실제 개인 정보를 넣지 않는다. `applicant_info`와 `target`은 외부로 보내지 않는다.
- push·force push는 팀 합의 없이 하지 않는다. 결과를 본 뒤 프롬프트·키워드·통과 기준을 바꾸면 바꾼 사실과 이유를 기록한다.
