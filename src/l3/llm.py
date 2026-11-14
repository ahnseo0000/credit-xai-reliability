"""L3 설명 생성 (L3_SPEC.md 4절 인터페이스, zip 판).

build_prompt  : record -> 프롬프트 (조건 A/B). features는 record의 order 순서 그대로. analysis_only(target)는 읽지 않는다.
call_llm      : Gemini 호출 + 디스크 캐시 + 재시도 + 기록 (status ok/failed)
parse_output  : JSON 추출·파싱 + 스키마 검사 (decision 불일치도 위반)
postprocess   : L3 단계 점검용 지표 (변수 매핑, 충실성 F·F_dir, A-핵심 키워드 1차 플래그)
                최종 판정·검정은 L4(src/l4_analysis.py)가 한다. 여기 값은 시험 보고와 디버깅용이다.
run_batch     : 레코드를 한 번씩 실행, 결과 표 저장 (한 행 = 한 호출)

의존: google-genai, python-dotenv, pandas 만. 기존 src/ 모듈은 import 하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from datetime import datetime

import pandas as pd

from . import settings as S

_RATE_LOCK = threading.Lock()
DIR_UP, DIR_DOWN, DIR_NONE = "부도 위험 증가", "부도 위험 감소", "영향 거의 없음"

# =====================================================================================
# 프롬프트 (L3_SPEC 2절)
# =====================================================================================
ALLOWED_FIELDS = ("model_output", "features", "applicant_info")   # build_prompt가 읽는 필드는 이것뿐

SYSTEM = f"""당신은 금융회사의 대출 심사 결과를 고객에게 알리는 통지문 작성자입니다.
이 회사는 자동화된 신용평가 모형으로 대출 신청을 평가하며, 신용정보법 제36조의2에 따라 고객에게 평가 결과, 주요 판단 기준, 평가에 사용된 기초정보의 개요를 설명해야 합니다.

규칙:
1. 아래에 제공된 정보만 근거로 씁니다. 제공되지 않은 사실을 지어내지 않습니다.
2. result.decision은 제공된 평가 결과(승인 또는 거절)를 그대로 씁니다.
3. main_reasons는 이 고객의 평가에 영향이 큰 순서로 정확히 {S.N_REASONS}개를 씁니다.
   - rank: 1부터 {S.N_REASONS}까지
   - variable: "평가에 사용된 변수" 목록에 적힌 변수 이름을 글자 그대로 씁니다. 목록에 없는 요인은 여기에 쓰지 않습니다.
   - direction: "{DIR_UP}" 또는 "{DIR_DOWN}"
   - reason_text: 고객에게 보이는 한두 문장. 이 고객의 실제 값을 넣습니다.
4. 목록에 없는 요인이 평가에 영향을 주었다고 판단하면 other_factors에 씁니다. 없으면 빈 목록입니다.
5. improvement_direction에는 고객이 다음 신청 때 평가를 개선하기 위해 할 수 있는 일을 변수별로 1~3개 씁니다.
6. basis_info_overview에는 평가에 어떤 종류의 정보를 사용했는지 개요를 씁니다.
7. 고객이 읽을 수 있는 쉬운 한국어로, 정중하게 씁니다.
8. 지정된 JSON 하나만 출력합니다."""

GENDER_KO = {"M": "남성", "F": "여성"}
INCOME_KO = {"하": "하위 1/3", "중": "중위 1/3", "상": "상위 1/3"}


def _applicant_block(ai: dict) -> str:
    # 두 조건에서 문구·위치가 같다
    return "\n".join([
        "## 신청자 정보",
        f"- 성별: {GENDER_KO.get(ai['group_gender'], ai['group_gender'])}",
        f"- 연령대: {ai['group_age']} (만 {ai['age_years']}세)",
        f"- 소득 구간: {INCOME_KO.get(ai['group_income'], ai['group_income'])}",
    ])


def _result_block(mo: dict) -> str:
    return "\n".join([
        "## 평가 결과",
        f"- 추정 부도확률: {mo['pd'] * 100:.1f}%",
        f"- 승인 기준: 부도확률 {mo['cutoff_pd'] * 100:.1f}% 미만이면 승인",
        f"- 평가 결과: {mo['decision']}",
    ])


def _features_block(features: list[dict], condition: str, base_value: float) -> str:
    feats = sorted(features, key=lambda f: f["order"])   # 고정 order 순서 (SHAP과 무관). shap·shap_rank로 정렬하지 않는다
    assert [f["order"] for f in feats] == list(range(1, len(feats) + 1))
    lines = ["## 평가에 사용된 변수 (모든 고객에게 같은 고정 순서이며, 중요도 순서가 아닙니다)"]
    if condition == "B":
        lines += ["각 변수 옆의 SHAP 값은 그 변수가 이 고객의 부도 위험 점수(log-odds)를 기준값보다 얼마나 올렸는지(+) 또는 내렸는지(-)를 나타냅니다. 절댓값이 클수록 영향이 큽니다.",
                  f"기준값은 {base_value:+.4f}이며, 기준값에 모든 SHAP 값을 더하면 이 고객의 점수가 됩니다."]
    fam = None
    for f in feats:
        if f["family"] != fam:
            fam = f["family"]
            lines.append(f"\n### {fam}")
        s = f"- {f['korean_name']}: {f['value_text']}"
        if condition == "B":
            s += f" | SHAP {f['shap']:+.4f} ({f['direction']})"
        lines.append(s)
    return "\n".join(lines)


OUTPUT_SPEC = f"""## 출력 형식
아래 키를 가진 JSON 하나만 출력합니다.
- result: {{"decision": "거절" 또는 "승인", "summary": "평가 결과 한 문장"}}
- main_reasons: 정확히 {S.N_REASONS}개. 각 {{"rank": 정수, "variable": "목록의 변수 이름", "direction": "{DIR_UP}" 또는 "{DIR_DOWN}", "reason_text": "..."}}
- basis_info_overview: 문자열
- improvement_direction: 1~3개. 각 {{"variable": "...", "suggestion": "..."}}
- other_factors: 0개 이상. 각 {{"name": "목록 밖 요인", "reason_text": "..."}}"""


def build_prompt(record: dict, condition: str) -> str:
    """condition = 'A'(SHAP 미제공) | 'B'(SHAP 제공). 허용 필드(model_output, features, applicant_info)만 읽는다."""
    assert condition in ("A", "B"), condition
    r = {k: record[k] for k in ALLOWED_FIELDS}     # analysis_only / target 은 여기서 떨어진다
    mo = r["model_output"]
    parts = [_result_block(mo), _applicant_block(r["applicant_info"]),
             _features_block(r["features"], condition, mo["base_value_logodds"]), OUTPUT_SPEC,
             "위 정보를 바탕으로 이 고객에게 보낼 통지문을 JSON으로 작성하세요."]
    return "\n\n".join(parts)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================================
# 호출 (L3_SPEC 5절)
# =====================================================================================
def _response_schema():
    from typing import Literal
    from pydantic import BaseModel   # google-genai 의존성으로 함께 설치된다

    class Result(BaseModel):
        decision: Literal["거절", "승인"]
        summary: str

    class Reason(BaseModel):
        rank: int
        variable: str               # enum으로 강제하지 않는다 (환각 측정)
        direction: Literal["부도 위험 증가", "부도 위험 감소"]
        reason_text: str

    class Improve(BaseModel):
        variable: str
        suggestion: str

    class Other(BaseModel):
        name: str
        reason_text: str

    class Notice(BaseModel):
        result: Result
        main_reasons: list[Reason]
        basis_info_overview: str
        improvement_direction: list[Improve]
        other_factors: list[Other]

    return Notice


def gen_settings(model: str) -> dict:
    return {"model": model, "temperature": S.TEMPERATURE, "top_p": S.TOP_P, "max_output_tokens": S.MAX_OUTPUT_TOKENS,
            "seed": S.SEED, "response_mime_type": "application/json", "response_schema": "Notice(L3_SPEC 3절)",
            "prompt_version": S.PROMPT_VERSION}


def cache_key(settings: dict, system: str, prompt: str) -> str:
    """sha256(모델·설정, 프롬프트 전문). 반복 번호 없음 (L3_SPEC 5절, 신청자 x 조건당 1회)."""
    blob = json.dumps(settings, sort_keys=True, ensure_ascii=False) + "\n<sys>\n" + system + "\n<user>\n" + prompt
    return sha256_text(blob)


class Gemini:
    _last_call = 0.0

    def __init__(self, model: str):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv(S.ROOT / ".env")
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit(".env에 GEMINI_API_KEY가 없습니다")
        from google.genai import types
        # SDK 자체 재시도는 끄고(attempts=1) HTTP 타임아웃을 건다 -> 재시도는 call_llm의 SPEC 5절 규칙(최대 3회)만 적용
        try:
            ho = types.HttpOptions(timeout=S.HTTP_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1))
        except Exception:  # noqa: BLE001  (구버전 SDK: retry_options 없음)
            ho = types.HttpOptions(timeout=S.HTTP_TIMEOUT_MS)
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=ho)   # 키는 출력·기록하지 않는다
        self.model = model
        self.settings = gen_settings(model)

    def _config(self):
        from google.genai import types
        kw = dict(system_instruction=SYSTEM, temperature=S.TEMPERATURE, max_output_tokens=S.MAX_OUTPUT_TOKENS, seed=S.SEED,
                  response_mime_type="application/json", response_schema=_response_schema())
        if S.TOP_P is not None:
            kw["top_p"] = S.TOP_P
        return types.GenerateContentConfig(**kw)

    def raw_call(self):
        cfg = self._config()
        return lambda prompt: self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)


def _err_kind(e: Exception) -> tuple[str, bool]:
    """(실패 사유, 재시도 대상 여부)"""
    try:
        from google.genai import errors
        if isinstance(e, errors.APIError):
            if e.code == 429:
                return "429 한도 초과", True
            if (e.code or 0) >= 500 or isinstance(e, errors.ServerError):
                return f"{e.code} 서버 오류", True
            return f"{e.code} 요청 오류", False
    except ImportError:
        pass
    if "timeout" in type(e).__name__.lower():
        return "타임아웃", True
    if isinstance(e, (ConnectionError, TimeoutError, OSError)) or type(e).__module__.startswith(("httpx", "httpcore")):
        return "네트워크 오류", True
    return type(e).__name__, False


def call_llm(prompt: str, *, use_cache: bool = True, client: Gemini | None = None) -> dict:
    """반환: {'text','cached','latency_s','attempts','tokens_in','tokens_out','tokens_thought','status','error',
              'model_version','finish_reason','key','called_at'}"""
    assert client is not None, "client=Gemini(model) 를 넘긴다"
    key = cache_key(client.settings, SYSTEM, prompt)
    path = S.CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        out = json.loads(path.read_text(encoding="utf-8"))
        out["cached"] = True
        return out
    call = client.raw_call()
    out = {"key": key, "cached": False, "attempts": 0, "status": "failed", "error": None, "text": None, "latency_s": None,
           "tokens_in": None, "tokens_out": None, "tokens_thought": None, "model_version": None, "finish_reason": None}
    for attempt in range(S.MAX_API_RETRIES + 1):
        with _RATE_LOCK:   # 동시 호출이어도 호출 시작 간격은 MIN_INTERVAL_SEC 이상 (스레드 공유)
            wait = S.MIN_INTERVAL_SEC - (time.time() - Gemini._last_call)
            if wait > 0:
                time.sleep(wait)
            Gemini._last_call = time.time()
        out["attempts"] = attempt + 1
        t0 = time.time()
        try:
            r = call(prompt)
            out["latency_s"] = round(time.time() - t0, 3)
            u = r.usage_metadata
            out.update(text=r.text, status="ok", error=None, model_version=getattr(r, "model_version", None),
                       finish_reason=str(r.candidates[0].finish_reason) if r.candidates else None,
                       tokens_in=getattr(u, "prompt_token_count", None), tokens_out=getattr(u, "candidates_token_count", None),
                       tokens_thought=getattr(u, "thoughts_token_count", None))
            break
        except Exception as e:  # noqa: BLE001
            out["latency_s"] = round(time.time() - t0, 3)
            kind, retry = _err_kind(e)
            out["error"] = f"{kind}: {str(e)[:600]}"
            if not retry or attempt == S.MAX_API_RETRIES:
                break
            time.sleep(S.BACKOFF_BASE_SEC * 2 ** attempt + random.uniform(0, 1))
    out["called_at"] = datetime.now().isoformat(timespec="seconds")
    if out["status"] == "ok":
        S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# =====================================================================================
# 파싱·스키마 (L3_SPEC 3절)
# =====================================================================================
REQUIRED = ("result", "main_reasons", "basis_info_overview", "improvement_direction", "other_factors")


def parse_output(text: str | None, expected_decision: str | None = None) -> tuple[dict | None, list[str]]:
    """반환: (파싱 결과 또는 None, 위반 목록). 위반이 비어 있어야 통과. decision 불일치도 위반이다."""
    if not text:
        return None, ["빈 응답"]
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None, ["파싱 실패"]
    v = [f"필수 키 없음: {k}" for k in REQUIRED if k not in obj]
    res = obj.get("result") or {}
    if res.get("decision") not in ("거절", "승인"):
        v.append("result.decision 값 오류")
    elif expected_decision and res.get("decision") != expected_decision:
        v.append(f"decision 불일치 (입력 {expected_decision}, 응답 {res.get('decision')})")
    mr = obj.get("main_reasons") or []
    if len(mr) != S.N_REASONS:
        v.append(f"main_reasons {len(mr)}개")
    if sorted(x.get("rank") for x in mr if isinstance(x, dict)) != list(range(1, len(mr) + 1)):
        v.append("rank가 1..n이 아님")
    if any(x.get("direction") not in (DIR_UP, DIR_DOWN) for x in mr):
        v.append("direction 값 오류")
    if any(not str(x.get("variable", "")).strip() for x in mr):
        v.append("variable 비어 있음")
    return obj, v


# =====================================================================================
# 후처리 (L3 점검용. 확정 지표는 L4)
# =====================================================================================
_strip = re.compile(r"[\s\W_]+", re.UNICODE)   # 공백·구두점 제거


def _norm(s) -> str:
    return _strip.sub("", str(s or "")).lower()


class Matcher:
    """분석 계획 2.1절 매핑 절차: 정규화 후 ① 변수 id → ② 한국어 이름·별칭 완전 일치 → ③ 별칭 부분 일치(3자 이상, 긴 것 우선).
    둘 이상의 변수에 걸리면 실패. 실패 = 환각 후보. 별칭 파일(docs/variable_aliases.csv)이 없으면 한국어 이름만 쓴다."""

    def __init__(self, features: list[dict], aliases_csv=None):
        aliases_csv = aliases_csv or S.ALIASES_CSV
        self.ids = {_norm(f["variable"]): f["variable"] for f in features}
        names: dict[str, set] = {}
        for f in features:
            names.setdefault(_norm(f["korean_name"]), set()).add(f["variable"])
        self.alias_used = False
        if os.path.exists(aliases_csv):
            al = pd.read_csv(aliases_csv, encoding="utf-8-sig")
            model_vars = {f["variable"] for f in features}
            for r in al.itertuples():
                if r.variable in model_vars:
                    names.setdefault(_norm(r.alias), set()).add(r.variable)
            self.alias_used = True
        self.names = names
        self.by_len = sorted((n for n in names if len(n) >= 3), key=len, reverse=True)

    def match(self, text) -> dict:
        n = _norm(text)
        if not n:
            return {"matched": None, "match_step": "none"}
        if n in self.ids:
            return {"matched": self.ids[n], "match_step": "1_id"}
        if n in self.names:
            vs = self.names[n]
            return {"matched": next(iter(vs)), "match_step": "2_exact"} if len(vs) == 1 else {"matched": None, "match_step": "ambiguous"}
        for a in self.by_len:                       # 가장 긴 별칭부터
            if a in n:
                same_len = {v for b in self.by_len if len(b) == len(a) and b in n for v in self.names[b]}
                return {"matched": next(iter(same_len)), "match_step": "3_partial"} if len(same_len) == 1 else {"matched": None, "match_step": "ambiguous"}
        return {"matched": None, "match_step": "none"}


def _load_keywords():
    if not os.path.exists(S.KEYWORDS_CSV):
        return None
    kw = pd.read_csv(S.KEYWORDS_CSV, encoding="utf-8-sig")
    core = kw[kw["group"] == "A-핵심"]
    pats = [k if t == "regex" else re.escape(k) for k, t in zip(core["keyword"], core["match_type"])]
    return re.compile("|".join(pats)) if pats else None


_KW = None


def postprocess(parsed: dict, record: dict, var_dict: pd.DataFrame | None = None) -> dict:
    """반환: {'summary': 호출 단위 dict, 'items': 요인 단위 list}. var_dict는 인터페이스 호환용(미사용, record의 features가 사전)."""
    global _KW
    feats = {f["variable"]: f for f in record["features"]}
    m = Matcher(record["features"])
    items = []
    for src, key in (("main_reasons", "variable"), ("other_factors", "name")):
        for x in parsed.get(src) or []:
            mm = m.match(x.get(key))
            f = feats.get(mm["matched"])
            items.append({"field": src, "rank": x.get("rank"), "said": x.get(key), "direction_said": x.get("direction"), **mm,
                          "shap_rank": f["shap_rank"] if f else None, "shap_direction": f["direction"] if f else None})
    reasons = [i for i in items if i["field"] == "main_reasons"]
    top4 = {v for v, f in feats.items() if f["shap_rank"] <= S.N_REASONS}
    said4 = {i["matched"] for i in reasons if i["matched"]}
    inter = said4 & top4
    dir_ok = sum(1 for i in reasons if i["matched"] in inter and i["shap_direction"] != DIR_NONE and i["direction_said"] == i["shap_direction"])
    n_map = sum(1 for i in items if i["matched"])
    summ = {"n_items": len(items), "n_mapped": n_map, "mapping_rate": n_map / len(items) if items else None,
            "n_unmapped": len(items) - n_map, "unmapped_said": " | ".join(str(i["said"]) for i in items if not i["matched"]),
            "n_other_factors": len(parsed.get("other_factors") or []),
            "F": len(inter) / S.N_REASONS, "F_dir": dir_ok / S.N_REASONS}
    if _KW is None:
        _KW = _load_keywords() or False
    if _KW:
        struct = json.dumps({k: parsed.get(k) for k in ("main_reasons", "other_factors", "improvement_direction")}, ensure_ascii=False)
        free = json.dumps({"summary": (parsed.get("result") or {}).get("summary"), "basis": parsed.get("basis_info_overview")}, ensure_ascii=False)
        summ["A_core_flag_struct"] = bool(_KW.search(struct))   # 1차 규칙 플래그. 사유 문맥 여부는 사람이 검토 (L4)
        summ["A_core_flag_free"] = bool(_KW.search(free))
    return {"summary": summ, "items": items}


# =====================================================================================
# 배치
# =====================================================================================
def load_records(path=None) -> list[dict]:
    p = path or S.INPUT_JSONL
    if not os.path.exists(p):
        raise SystemExit(f"입력 없음: {p}\nzip을 repo 밖 폴더에 풀고 L3_INPUT_DIR 로 지정한다 (기본: repo 옆 l3_handoff_package/)")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log(msg: str):
    S.L3_DIR.mkdir(parents=True, exist_ok=True)
    line = f"- `{datetime.now():%Y-%m-%d %H:%M:%S}` {msg}"
    print(line, flush=True)
    with open(S.RUNLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(rec: dict, condition: str, *, client: Gemini, tag: str, input_hash: str) -> tuple[dict, dict]:
    """신청자 x 조건 1건 실행 (파싱·스키마 재호출 포함). 반환: (표 한 행, 원문 포함 기록 한 줄). 파일은 쓰지 않는다 (스레드 안전)."""
    prompt = build_prompt(rec, condition)
    p_hash = sha256_text(SYSTEM + "\n" + prompt)
    parse_tries, parsed, viol, out = 0, None, [], None
    for pt in range(S.MAX_PARSE_RETRIES + 1):
        out = call_llm(prompt, use_cache=(pt == 0), client=client)
        parse_tries = pt + 1
        if out["status"] != "ok":
            parsed, viol = None, [out["error"].split(":")[0]]   # 실패 사유 (429 한도 초과 / 503 서버 오류 / 타임아웃 …)
            break
        parsed, viol = parse_output(out["text"], rec["model_output"]["decision"])
        if parsed is not None and not viol:
            if pt > 0:   # 재호출로 통과한 응답을 캐시에 남긴다 (같은 키는 다시 부르지 않게)
                (S.CACHE_DIR / f"{out['key']}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            break
    ok = out["status"] == "ok" and parsed is not None and not viol
    base = {"tag": tag, "sample_id": rec["sample_id"], "cell_id": rec["cell"]["cell_id"], "pd_bin": rec["cell"]["pd_bin"],
            "condition": condition, "model": client.model, "model_version": out.get("model_version"),
            "prompt_version": S.PROMPT_VERSION, "input_sha256": input_hash, "prompt_sha256": p_hash,
            "called_at": out.get("called_at"), "cached": out["cached"], "api_attempts": out["attempts"], "parse_attempts": parse_tries,
            "latency_s": out["latency_s"], "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"],
            "tokens_thought": out["tokens_thought"], "finish_reason": out["finish_reason"],
            "status": "ok" if ok else "failed", "fail_reason": "; ".join(viol), "parse_ok": parsed is not None}
    post = postprocess(parsed, rec) if parsed is not None else None
    if post:
        base.update(post["summary"])
    line = {**base, "response_text": out["text"], "error": out["error"], "items": post["items"] if post else None}
    return base, line


def run_batch(records: list[dict], condition: str, *, client: Gemini, tag: str, input_hash: str | None = None) -> pd.DataFrame:
    """레코드를 순서대로 한 번씩 실행 (시험용). 한 행 = 한 호출. 원문 포함 기록은 results/l3/raw/calls_<tag>.jsonl, 표는 results/l3/raw/l3_<tag>.csv"""
    input_hash = input_hash or sha256_file(S.INPUT_JSONL)
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(S.RAW_DIR / f"calls_{tag}.jsonl", "a", encoding="utf-8") as logf:
        for i, rec in enumerate(records, 1):
            base, line = run_one(rec, condition, client=client, tag=tag, input_hash=input_hash)
            rows.append(base)
            logf.write(json.dumps(line, ensure_ascii=False) + "\n")
            logf.flush()   # 한 건마다 바로 기록 (중단돼도 남는다)
            print(f"  [{tag}] {i}/{len(records)} {base['status']} ({base['latency_s']}s, cached={base['cached']})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(S.RAW_DIR / f"l3_{tag}.csv", index=False, encoding="utf-8-sig")
    return df
