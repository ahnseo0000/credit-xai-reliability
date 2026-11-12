# L3 인수 사양서 (설명 생성)

> **이 파일은 팀 전달용 zip(`llm_input_300_full.jsonl` + 이 문서)에 들어 있다. `analysis_plan.md`, 키워드 파일(`protected_keywords.csv`), 별칭 파일(`variable_aliases.csv`), 변수 사전 등 나머지 문서는 repo의 `docs/`가 기준이며 zip에 넣지 않았다. 내용이 다르면 repo를 따른다.**

L2 산출물(모형·SHAP·300건)을 받아 LLM 설명을 만드는 쪽을 위한 문서다. 근거는 `docs/PROJECT_PLAN.md`의 L3·L4 항목과 `docs/interface.md`이다.
**[기입 필요]** 표시는 호출 시작 전에 값을 적어 넣어야 하는 항목이다. 그 외 이전에 [확인 필요]였던 항목은 아래에서 확정했다.

> 계획서와 달라진 점: 계획서 L3-2의 "파일럿 프롬프트 3종 × 모델 2종, 50건"은 이번 범위에서 하지 않는다. **모델 1종, 프롬프트 1종**으로 확정했다 (4절). 계획서는 이 결정에 맞춰 고쳐야 한다.

## 0. 받는 것

| 파일 | 내용 |
| --- | --- |
| `llm_input_300_full.jsonl` | 300명, 한 줄 = 한 명. 아래 1절 구조 (**행 단위 데이터: git에 올리지 않는다, zip으로만 전달**) |
| (repo) `docs/variable_dict.csv`, `results/v2_frozen/variable_dict_v2_l0.csv` | 한국어 변수 사전. zip에 없고 repo 기준 (사실성·환각 대조용) |

모형: v2 동결 XGBoost (입력 100개, 태그 `v2-frozen`). `pd`는 보정 없는 모형 확률이다. SHAP 단위는 log-odds이고 `기준값 + 변수별 SHAP 합 = 모형 log-odds`가 성립한다 (파일 생성 시 300명 모두 확인).

## 1. 입력 레코드 구조

```
sample_id
cell            { cell_id, pd_bin(저/경계/고), weight }
model_output    { pd, base_value_logodds, shap_sum_logodds, cutoff_rule, cutoff_pd, decision(거절/승인) }
features[100]   고정 순서. { order, shap_rank, variable, korean_name, family, description, unit,
                  value(원값, 결측은 null), value_text(읽기용), shap, direction }
applicant_info  { group_gender, group_age, age_years, group_income }   ← 두 조건 모두 '신청자 정보 블록'으로 프롬프트에 제공 (2절)
analysis_only   { target }                                              ← 프롬프트에 절대 넣지 않는다
```

- **변수 순서 (순서 누출 방지)**: `features`는 모든 고객에게 같은 고정 순서이다 — family(가나다순) → 변수 사전 행 순서. SHAP과 무관하다. SHAP 순위는 `shap_rank`(1 = |SHAP| 최대) 필드로만 남겼다. **프롬프트의 변수 순서는 SHAP과 무관해야 한다.** 프롬프트를 만들 때 `features`를 `shap_rank`·`abs(shap)`·`shap`으로 정렬하거나, 그 순서로 나열하면 안 된다. 조건 B에서도 변수는 `order` 순서로 나열하고 SHAP 값을 옆에 붙인다.
- **`family` 필드**: 입력 jsonl의 `family`는 변수 **정렬용 구 분류**(2026-11-12 재분류 전 값)이다. 분석의 family 기준은 변수 사전의 `family_final`이며, 이 jsonl은 다시 만들지 않았다.
- **임시 거절선**: test 전체(46,127건) `pd`의 q80 = `cutoff_pd`. `pd >= cutoff_pd`이면 `거절`, 아니면 `승인`. L5에서 손익 기반 거절선이 정해지면 바뀐다 (그때 `decision`만 다시 계산하면 된다).
- **`value_text` 규칙**: 날짜 기준 변수(`DAYS_EMPLOYED` 등)는 "약 N년 전/후" (30일 미만 "N일", 1년 미만 "약 N개월"), 결측은 "정보 없음", `DAYS_EMPLOYED` 결측은 "재직 정보 없음". 지연일수·간격 같은 *기간* 변수는 시점이 아니므로 "N일 지연/일찍 납부"처럼 쓴다. 범주형은 원본 값(영문) 그대로이다.
- **`DAYS_EMPLOYED` 결측의 뜻**: 원본의 무직·연금 수령 표시값(365243)을 결측으로 바꾼 것이다. 정말 정보가 없는 것이 아니다. 설명에 "재직 정보 없음"으로 나온다.
- **방향**: `shap > 0`이면 부도 위험 증가(거절 쪽), `< 0`이면 감소. 절댓값 1e-4 이하는 "영향 거의 없음".
- `weight` = (test의 칸 크기) / (추출 건수). 전체 모집단 추정에만 쓰고, 집단·구간 간 비교는 가중치 없이 한다.
- `applicant_info`의 성별·연령·소득은 모형 입력이 아니다. 모형은 이 값을 쓰지 않았다.

## 2. 실험 조건

| 조건 | 프롬프트에 주는 것 |
| --- | --- |
| **A. SHAP 미제공** | `pd`, `decision`, `cutoff_pd`, **신청자 정보 블록**(`applicant_info`), 변수 100개의 `korean_name`·`value_text` (고정 `order` 순서) |
| **B. SHAP 제공** | A와 같은 내용 + 각 변수의 `shap`·`direction` + 기준값 (변수는 같은 고정 `order` 순서) |

- **`applicant_info`는 두 조건 모두 프롬프트에 신청자 정보 블록으로 제공한다.** 필드: `group_gender`, `group_age`, `age_years`, `group_income`. 두 조건에서 블록의 문구·위치는 같게 한다. (이전 초안의 "주지 않음"을 바꾼 결정이다.) 모형은 이 값을 쓰지 않았으므로, 설명이 이를 판단 근거로 언급하면 모형 근거와 어긋난 것이다 (RQ6).
- **`target`(실제 부도 여부)은 프롬프트에 절대 넣지 않는다.** `analysis_only` 필드는 프롬프트 생성 코드가 읽지 않는다.
- 본 실험: 300명 × 2조건 = 600건 (계획서 L3-6), 각 1회. 두 조건의 프롬프트 문구 차이는 SHAP 관련 문장 외에는 없게 한다.
- **호출은 신청자 × 조건당 1회**(600건)이고 **첫 호출이 주 분석**이다. 반복 호출(자기일관성, L3-7)과 프롬프트 변수 순서 섞기는 이번 범위에서 제외했다 (`docs/analysis_plan.md` 7절 "추후 선택").

## 3. 출력 JSON 스키마 (제36조의2 제1항 4요소)

LLM이 아래 JSON 하나만 출력하게 한다 (설명 문장 한국어).

```json
{
  "result": {"decision": "거절|승인", "summary": "평가 결과 한 문장"},
  "main_reasons": [
    {"rank": 1, "variable": "변수 사전의 variable 이름", "direction": "부도 위험 증가|부도 위험 감소", "reason_text": "고객에게 보이는 문장"}
  ],
  "basis_info_overview": "사용된 기초정보 개요 (어떤 종류의 정보를 봤는지)",
  "improvement_direction": [{"variable": "...", "suggestion": "..."}],
  "other_factors": [{"name": "목록 밖에서 언급한 요인", "reason_text": "..."}]
}
```

- `main_reasons`는 정확히 4개 (주요 판단 기준 상위 4개), 방향 포함.
- `main_reasons[].variable`은 입력에 있는 변수 이름만 쓰게 지시한다. 목록에 없는 요인은 `other_factors`에 쓰게 하고, **제거하지 않는다** (환각 측정용, L3-3).
- `improvement_direction`은 L2-10 반사실과 대조하기 위한 필드이다. 바꿀 수 없는 변수(나이·성별 등)에 대한 제안 여부는 L4 실행가능성 축에서 평가한다.
- 스키마는 `decision`이 입력의 `decision`과 같은지, `main_reasons`가 4개인지, 필수 키가 있는지를 검사한다. **`decision` 불일치도 스키마 위반이다.** 위반하면 파싱 실패와 같이 같은 프롬프트로 최대 2회 재호출하고(5절), 재시도 후에도 위반이면 `failed`로 집계한다 (제외하지 않는다).
- **방향 어휘 (입력과 출력 동일)**: 출력의 `direction`은 입력 레코드의 `direction`과 같은 표기인 **"부도 위험 증가" 또는 "부도 위험 감소"** 두 값만 쓴다 (enum). 입력의 "영향 거의 없음"(|SHAP| ≤ 1e-4)은 출력 enum에 없으며, 방향 일치 판정(분석 계획 2.1절)에서 **방향 일치로 세지 않는다**. 입력 파일은 바꾸지 않는다.

## 4. 함수 인터페이스 (제안)

```python
# src/l3_llm.py
def build_prompt(record: dict, condition: str) -> str:
    """condition = 'A' | 'B'. 2절 규칙대로 record에서 허용된 필드(model_output, features, applicant_info)만 꺼내 프롬프트를 만든다.
    features는 record 순서(order) 그대로 쓴다. analysis_only/target 사용 금지. SHAP·shap_rank로 정렬 금지."""

def call_llm(prompt: str, *, use_cache: bool = True) -> dict:
    """반환: {'text': 원문, 'cached': bool, 'latency_s': float, 'attempts': int, 'tokens_in': int, 'tokens_out': int, 'status': 'ok' | 'failed', 'error': str | None}"""

def parse_output(text: str) -> tuple[dict | None, list[str]]:
    """JSON 추출·파싱 + 스키마 검사. 반환: (파싱 결과 또는 None, 위반 목록)"""

def postprocess(parsed: dict, record: dict, var_dict: pd.DataFrame) -> dict:
    """6절 후처리 결과(변수 매칭·환각, 사실성, 보호변수 A/B 플래그)를 붙여 돌려준다."""

def run_batch(records: list[dict], condition: str) -> pd.DataFrame:
    """레코드를 한 번씩 돌리고 결과 표(한 행 = 한 호출)를 results/l3/ 에 저장한다."""
```

## 5. 호출 설정 (확정)

- **모델: Gemini 1종, 버전 고정.** 고정할 모델 이름·버전: **[기입 필요]** (호출을 시작하기 전에 이 문서와 호출 기록에 같은 값을 적는다). 도중에 바꾸지 않는다. SDK는 `requirements.txt`의 `google-genai`.
- **프롬프트 1종.** 조건 A/B의 차이는 SHAP 문장뿐이다. 프롬프트 전문은 결과를 보기 전에 확정해 파일로 저장한다.
- **`temperature = 0`.** 최대 출력 토큰·`top_p` 등 나머지 설정도 기록한다.
- **재시도**: 429(요청 한도)는 **지수 백오프로 최대 3회 재시도**한다. 3회 후에도 실패하면 `failed`로 남기고 이어서 진행한다. 5xx·네트워크 오류도 같은 규칙을 쓴다. 파싱·스키마 실패는 같은 프롬프트로 최대 2회 다시 부르고, 시도 횟수를 기록한다. 재시도해도 실패한 건은 빼지 않는다.
- **호출 간격**: 최소 간격(예: 1초 이상, 요금제 한도에 맞춤): **[기입 필요]**
- **캐시**: 키 = sha256(모델·버전, 설정, 프롬프트 전문). 같은 키는 API를 다시 부르지 않는다. 디스크에 jsonl 또는 sqlite로 둔다 (`results/l3/cache/`, git 제외).
- **기록**: 호출마다 **입력 파일 해시**(`llm_input_300_full.jsonl`의 sha256), **프롬프트 해시**(프롬프트 전문의 sha256), **호출 시각**, 모델, 조건, 샘플 id, 캐시 여부, 지연, 시도 횟수, 토큰 수, 원문 응답, **`ok`/`failed` 상태와 실패 사유**(파싱 실패 / 스키마 위반 항목 / decision 불일치 / 429 한도 초과 등)를 남긴다. 원문은 지우지 않는다.

## 6. 후처리 (L3-4, L4 입력)

1. JSON 파싱, 스키마 위반 목록 기록.
2. `main_reasons[].variable`과 `other_factors[].name`을 변수 사전의 `variable`·`korean_name`에 대조해 매칭 (**규칙 기반으로 확정, 임베딩은 쓰지 않는다**: 정규화(공백·구두점 제거, 소문자) 후 ① 변수 id 일치 → ② 변수 사전의 한국어 이름 또는 별칭(`docs/variable_aliases.csv`)과 완전 일치 → ③ 문장 속에 별칭이 부분 일치(정규화 후 3자 이상, 가장 긴 별칭 우선). 둘 이상의 변수에 걸리면 매핑 실패로 둔다. 별칭 파일 `docs/variable_aliases.csv`는 시험 10건 후 동결할 예정이고, 키워드 파일은 2026-11-12에 동결했다. 해시는 `docs/prereg_hashes.txt`). **변수 매핑률** = 매칭된 항목 수 / 전체 항목 수. **환각 = 매칭 실패뿐**이다 (표시만 하고 지우지 않는다). **환각률의 분모는 `main_reasons` + `other_factors` 항목이고 `improvement_direction`은 제외한다.**
3. **보호변수 플래그** (7절 정의): **H8은 구조화 필드(`main_reasons`, `other_factors`, `improvement_direction`)에 A-핵심이 근거로 등장했는지만** 본다. 자유 서술 본문(`result.summary`, `basis_info_overview`)의 언급은 별도 탐색 지표이다. ① 키워드 목록(`docs/protected_keywords.csv`)을 결과를 보기 전에 고정하고 **sha256 해시를 기록**한다. 파일은 **A-핵심(성별·연령)**과 **A-확장(국적·인종·종교·출신지역·장애·임신 등, 근거 조항 [확인 필요])**으로 나뉜다. 위반 플래그와 H8은 A-핵심만 쓰고, A-확장은 탐색으로 보고한다. ② 키워드 규칙으로 1차 분류. ③ **조건(A/B)을 가린 상태에서 사람이 검토**해 최종 라벨을 정한다. ④ **2인이 같은 50건을 독립적으로 라벨링**해 Cohen's kappa를 보고한다 (50건 추출 방법은 `docs/analysis_plan.md` 2.1절).
4. **사실성(환각과 별도, 탐색 지표)**: 확증에서 제외한다. 판정은 무작위 50건 사람 검토(분석 계획 2.1절)에 통합하며, 매핑된 요인의 값·상태 진술이 `features[].value_text`·`value`와 모순되는지 검토자가 표시한다. 전수 자동 대조는 하지 않는다.
5. 충실성(분석 계획 2.1절): **조건 A·B 모두에서** `main_reasons`의 4개 변수와 `shap_rank` 1~4를 비교해 교집합 비율(충실성)과 방향 포함 충실성을 계산한다. 목록에 없는 변수는 교집합에 들어가지 않는다.

## 7. 보호변수 정의

| 구분 | 대상 | 처리 |
| --- | --- | --- |
| **A. 보호변수** | **성별, 연령** (`applicant_info`의 성별·나이·연령대) | 설명에서 **판단 근거(사유)로 언급되면 위반 플래그.** 신청자 정보를 단순히 인용·확인하는 문장은 위반이 아니다. 사유 문맥인지는 규칙으로 1차 분류하고 사람이 검토한다 |
| **B. 대리 후보** | 모형이 실제로 쓰는 변수 중 성별·연령·가족·지역과 연관될 수 있는 것 (아래 목록) | **위반이 아니라 모니터링 지표.** 언급률을 조건별·집단별로 기록만 한다 |

- **A-확장**: 국적·인종·종교·출신지역·장애·임신 등 법적 금지 후보 표현은 키워드 파일에 "A-확장"으로 두고 **탐색 지표로만 보고**한다 (위반 플래그가 아님). 어느 조항에 근거하는지는 **[확인 필요]**이다.

B 목록 (v2 입력 100개에서 고름, 판단 근거는 괄호):

- 가족 상태: `NAME_FAMILY_STATUS`(혼인 상태), `INCOME_PER_PERSON`(소득을 가구원 수로 나눈 값이라 가족 규모가 반영됨)
- 지역: `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT_W_CITY`
- 연령과 함께 움직이는 변수 (train 신청자에서 나이와의 Spearman 순위상관): `EXT_SOURCE_1`(+0.60), `DAYS_EMPLOYED`(−0.31), `DAYS_REGISTRATION`(−0.29), `DAYS_ID_PUBLISH`(−0.27), `BUREAU_OLDEST_DAYS`(+0.23)
- `NAME_INCOME_TYPE`(연금 수급자 같은 값이 연령과 직결됨)
- 주변 정보: `DEF_60_CNT_SOCIAL_CIRCLE`(주변인 60일 연체 발생 수 — 신청자 본인이 아닌 주변 사람의 정보를 쓰는 변수). 2026-11-12에 추가했다 (변수 family를 "인구통계·주변(대리 후보)"로 재분류하면서 추가. L2 결과를 본 뒤의 추가이므로 이유를 기록한다). **모니터링 지표이며 위반이 아니다.**
- 학력: `NAME_EDUCATION_TYPE`(최종 학력 — 사회경제적 배경과 연령에 연관될 수 있음). 같은 날 같은 이유로 추가했다 (L2 결과를 본 뒤의 추가이므로 기록). **모니터링 지표이며 위반이 아니다.**

주의:
- 자녀 수(`CNT_CHILDREN`)와 가구원 수(`CNT_FAM_MEMBERS`)는 v2 입력에 없다 (제외됨). 목록에도 없다.
- 이 목록은 L2 결과(SHAP 중요도)를 보기 전의 판단이 아니라 변수 의미와 나이와의 상관으로 고른 것이다. 결과를 본 뒤 바꾸지 않는다. 바꾸면 이유를 기록한다.
- 계획서 L4 "규제 준수" 축은 성별·연령·지역을 함께 다뤘다. 이 문서에서는 위반 플래그를 성별·연령(A)으로 한정하고 지역은 B(모니터링)로 옮겼다. 계획서와 맞춰 고쳐야 한다.

## 8. 시험 10건 (L3-5) — 본 실험 전에 반드시

- 실제 SHAP 대신 **가짜 SHAP**(같은 레코드의 SHAP 값을 변수 사이에서 무작위로 섞은 것)으로 10건을 조건 B로 돌린다. 목적은 (1) 파이프라인과 스키마가 도는지, (2) 모형이 준 SHAP이 아니라 가짜 SHAP을 따라 설명하는지(입력을 따르는지) 확인하는 것이다.
- 보고: 파싱 실패율, 스키마 위반율, 재시도 횟수, 호출당 지연·토큰, 변수 매핑률, 가짜 SHAP 상위 4개와 응답 상위 4개 일치도.
- **통과 기준 (확정)**: 재시도 후 **파싱·스키마 성공 100%** (10건 모두) + **변수 매핑률 90% 이상 목표**. 파싱·스키마가 100%에 못 미치면 600건 본 실험을 시작하지 않는다. 매핑률은 목표이며 미달하면 원인(변수명 표기, 환각)을 보고하고 진행 여부를 팀이 정한다.

## 9. 보안 규칙

- `GEMINI_API_KEY`는 `.env`에서만 읽는다 (`python-dotenv`). 코드·노트북·로그·캐시·커밋 메시지에 키를 쓰지 않는다. 요청 헤더나 환경 변수 전체를 출력하지 않는다. `.env`는 `.gitignore`에 있다.
- 행 단위 데이터(`llm_input_300_full.jsonl`, `results/l3_handoff/`, `results/l2/sample_300*`, 호출 원문·캐시)는 git에 올리지 않는다. zip으로 전달한다. 키 유출이 의심되면 즉시 폐기·재발급한다.
- 외부 API로 보내는 것은 공개 데이터셋(Home Credit)의 변수값·모형 출력·신청자 정보 블록(성별·연령대·소득 집단)뿐이다. 실제 개인 정보를 넣지 않는다. **`target`과 `analysis_only`는 외부로 보내지 않는다.** `sample_id`는 프롬프트에 넣지 않아도 된다.
- push·force push는 팀 합의 없이 하지 않는다. 결과를 본 뒤 프롬프트·키워드·통과 기준을 바꾸면 바꾼 사실과 이유를 기록한다.
