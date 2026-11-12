"""L4 분석 코드 골격 (docs/analysis_plan.md): 매핑, 충실성, 방향 포함 충실성, 환각률, 사실성 입력, A 목록 언급률(규칙 1차).

LLM 응답은 docs/L3_SPEC.md 3절 JSON 스키마. 입력 레코드는 results/l3_handoff/llm_input_300_full.jsonl 한 줄.
- 매핑은 규칙 기반이다 (변수 id, 변수 사전의 한국어 이름, 별칭 목록 docs/variable_aliases.csv). 임베딩은 쓰지 않는다.
- 보호변수 키워드: docs/protected_keywords.csv (A-핵심 = H8, A-확장 = 탐색). 최종 라벨은 사람 검토이며 여기서는 규칙 1차 플래그만 계산한다.
- 호출 전에 키워드·별칭 파일의 sha256을 기록한다 (file_sha256).
실행: python -m src.l4_analysis --selftest   (가짜 LLM 출력으로 동작 확인, API 호출 없음)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

from . import config as C

KEYWORDS = C.DOCS_DIR / "protected_keywords.csv"
ALIASES = C.DOCS_DIR / "variable_aliases.csv"
UP, DOWN, NONE_DIR = "부도 위험 증가", "부도 위험 감소", "영향 거의 없음"
DIRS = (UP, DOWN)
TEXT_FIELDS = ("summary", "basis_info_overview")
MIN_ALIAS_LEN = 3   # 문장 속 부분 일치에 쓰는 별칭의 최소 길이 (정규화 후)


def file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def norm(s: str) -> str:
    """공백·구두점 제거, 소문자. 한글/영문/숫자만 남긴다."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(s).lower())


# ---------------------------------------------------------------- 매핑 (규칙 기반)
class Mapper:
    """변수 id -> 정규화한 이름·별칭 색인. map()은 (variable | None, 방식)을 돌려준다."""

    def __init__(self, variables: list[dict], aliases: pd.DataFrame | None = None):
        self.exact: dict[str, set[str]] = {}
        for v in variables:
            for key in (v["variable"], v["korean_name"]):
                self.exact.setdefault(norm(key), set()).add(v["variable"])
        if aliases is not None:
            for r in aliases.itertuples():
                self.exact.setdefault(norm(r.alias), set()).add(r.variable)
        self.keys_by_len = sorted(self.exact, key=len, reverse=True)

    def map(self, text: str):
        t = norm(text)
        if not t:
            return None, "empty"
        if t in self.exact:
            vs = self.exact[t]
            return (next(iter(vs)), "exact") if len(vs) == 1 else (None, "ambiguous_exact")
        for k in self.keys_by_len:             # 가장 긴 별칭부터 부분 일치
            if len(k) >= MIN_ALIAS_LEN and k in t:
                vs = self.exact[k]
                return (next(iter(vs)), "contains") if len(vs) == 1 else (None, "ambiguous_contains")
        return None, "unmapped"


def load_mapper(variables: list[dict]) -> Mapper:
    al = pd.read_csv(ALIASES, encoding="utf-8-sig") if ALIASES.exists() else None
    return Mapper(variables, al)


def variables_of(record: dict) -> list[dict]:
    return [{"variable": f["variable"], "korean_name": f["korean_name"]} for f in record["features"]]


# ---------------------------------------------------------------- 스키마 검사 (실패 정의)
def validate_output(parsed, record: dict) -> list[str]:
    """위반 목록. decision 불일치도 위반이다. 비어 있으면 통과."""
    v = []
    if not isinstance(parsed, dict):
        return ["not_a_json_object"]
    for k in ("result", "main_reasons", "basis_info_overview", "improvement_direction", "other_factors"):
        if k not in parsed:
            v.append(f"missing:{k}")
    res = parsed.get("result")
    if isinstance(res, dict):
        if res.get("decision") not in ("거절", "승인"):
            v.append("decision_invalid")
        elif res["decision"] != record["model_output"]["decision"]:
            v.append("decision_mismatch")
        if "summary" not in res:
            v.append("missing:result.summary")
    elif "result" in parsed:
        v.append("result_not_object")
    mr = parsed.get("main_reasons")
    if isinstance(mr, list):
        if len(mr) != 4:
            v.append(f"main_reasons_count:{len(mr)}")
        for i, x in enumerate(mr):
            if not isinstance(x, dict) or "variable" not in x or x.get("direction") not in DIRS:
                v.append(f"main_reasons[{i}]_invalid")
    elif "main_reasons" in parsed:
        v.append("main_reasons_not_list")
    for k in ("improvement_direction", "other_factors"):
        if k in parsed and not isinstance(parsed[k], list):
            v.append(f"{k}_not_list")
    return v


# ---------------------------------------------------------------- 지표
def shap_top(record: dict, k: int = 4) -> dict[str, str]:
    """SHAP 상위 k개 변수 -> 입력 레코드의 direction."""
    return {f["variable"]: f["direction"] for f in record["features"] if f["shap_rank"] <= k}


def fidelity(parsed: dict, record: dict, mapper: Mapper) -> dict:
    """충실성 F와 방향 포함 충실성 F_dir (분석 계획 2.1절)."""
    top = shap_top(record, 4)
    seen, hits, dir_hits = [], [], 0
    for x in parsed["main_reasons"][:4]:
        var, _ = mapper.map(x.get("variable", ""))
        if var is None or var in seen:
            continue
        seen.append(var)
        if var in top:
            hits.append(var)
            if top[var] in DIRS and x.get("direction") == top[var]:   # "영향 거의 없음"은 일치로 세지 않는다
                dir_hits += 1
    return {"F": len(hits) / 4, "F_dir": dir_hits / 4, "n_overlap": len(hits)}


def extract_factors(parsed: dict) -> list[dict]:
    """환각률의 분모: main_reasons + other_factors (improvement_direction 제외)."""
    out = [{"src": "main", "text": x.get("variable", ""), "reason": x.get("reason_text", "")} for x in parsed.get("main_reasons", [])]
    out += [{"src": "other", "text": x.get("name", ""), "reason": x.get("reason_text", "")} for x in parsed.get("other_factors", []) if isinstance(x, dict)]
    return out


def hallucination(parsed: dict, mapper: Mapper) -> dict:
    """환각 = 매핑 실패뿐. 모순된 값은 사실성 지표로 따로 본다. 매핑 방식도 함께 돌려준다."""
    fac = extract_factors(parsed)
    maps = [mapper.map(f["text"]) for f in fac]
    n_un = sum(1 for v, _ in maps if v is None)
    return {"H": n_un / len(fac) if fac else np.nan, "n_factors": len(fac), "n_unmapped": n_un,
            "mapped_vars": [v for v, _ in maps], "map_rate": 1 - n_un / len(fac) if fac else np.nan}


def load_keywords(path=KEYWORDS) -> pd.DataFrame:
    kw = pd.read_csv(path, encoding="utf-8-sig")
    kw["pat"] = [re.compile(r.keyword) if r.match_type == "regex" else re.compile(re.escape(r.keyword)) for r in kw.itertuples()]
    return kw


def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)


def collect_fields(parsed: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(구조화 필드, 자유 서술 본문). H8은 구조화 필드(main_reasons / other_factors / improvement_direction)만 본다.
    자유 서술 본문(result.summary, basis_info_overview)은 별도 탐색 지표다."""
    struct = {}
    for k in ("main_reasons", "other_factors", "improvement_direction"):
        for i, x in enumerate(parsed.get(k, []) if isinstance(parsed.get(k), list) else []):
            struct[f"{k}[{i}]"] = _dump(x)
    free = {"result.summary": str((parsed.get("result") or {}).get("summary", "")), "basis_info_overview": str(parsed.get("basis_info_overview", ""))}
    return struct, free


def mention_flags(parsed: dict, kw: pd.DataFrame) -> dict:
    """키워드 규칙 1차 플래그 (최종 라벨은 사람 검토).
    flag_A_core      = 구조화 필드에 A-핵심(성별·연령)이 등장 -> H8 후보
    flag_A_core_free = 자유 서술 본문에 A-핵심이 등장 -> 탐색 지표
    flag_A_ext       = A-확장이 어느 곳에든 등장 -> 탐색 지표"""
    struct, free = collect_fields(parsed)
    hits = {"core": [], "core_free": [], "ext": []}
    for where, texts in (("struct", struct), ("free", free)):
        for field, text in texts.items():
            for r in kw.itertuples():
                m = r.pat.search(text)
                if not m:
                    continue
                if r.group == "A-핵심":
                    hits["core" if where == "struct" else "core_free"].append((field, r.keyword, m.group(0)))
                else:
                    hits["ext"].append((field, r.keyword, m.group(0)))
    return {"flag_A_core": bool(hits["core"]), "flag_A_core_free": bool(hits["core_free"]), "flag_A_ext": bool(hits["ext"]),
            "hits_core": hits["core"], "hits_core_free": hits["core_free"], "hits_ext": hits["ext"]}


def analyze_one(parsed, record: dict, mapper: Mapper, kw: pd.DataFrame) -> dict:
    """신청자 × 조건 한 건. 스키마 위반이면 status='failed'로 두고 지표는 비운다 (제외하지 않고 집계)."""
    viol = validate_output(parsed, record)
    row = {"sample_id": record["sample_id"], "status": "ok" if not viol else "failed", "violations": ";".join(viol)}
    if viol:
        return row
    row.update(fidelity(parsed, record, mapper))
    h = hallucination(parsed, mapper)
    row.update({k: h[k] for k in ("H", "n_factors", "n_unmapped", "map_rate")})
    mf = mention_flags(parsed, kw)
    row.update({k: mf[k] for k in ("flag_A_core", "flag_A_core_free", "flag_A_ext", "hits_core", "hits_core_free", "hits_ext")})
    return row


# ---------------------------------------------------------------- 동결 확인 / 단위시험
FROZEN_HASHES = C.DOCS_DIR / "prereg_hashes.txt"


def verify_frozen(name: str, path) -> bool:
    """docs/prereg_hashes.txt에 '<이름> <sha256>'으로 동결된 파일이 그대로인지 확인한다."""
    for line in open(FROZEN_HASHES, encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 2 and parts[0] == name and re.fullmatch(r"[0-9a-f]{64}", parts[1]):
            return parts[1] == file_sha256(path)
    raise KeyError(f"{name}의 동결 해시가 {FROZEN_HASHES.name}에 없다")


def test_age_regex(kw: pd.DataFrame | None = None):
    """연령대 표기 패턴(10~90대): 연령 표현은 잡고, '기업'·'대출'이 붙는 표현은 제외한다."""
    kw = load_keywords() if kw is None else kw
    pat = kw[(kw.group == "A-핵심") & (kw.match_type == "regex") & kw.keyword.str.contains("대")].pat.iloc[0]
    yes = ["30대", "50 대", "20대 고객", "60대 이상", "90대 신청자", "연령은 40대입니다", "10대"]
    no = ["10대 기업", "30대기업", "상위 100대 기업", "50 대출", "대출", "15대", "5대 은행", "20대출 상품", "70대 기업 대표"]
    for t in yes:
        assert pat.search(t), f"잡혀야 함: {t}"
    for t in no:
        assert not pat.search(t), f"제외돼야 함: {t}"
    return len(yes), len(no)


def miss_rate_estimate(unflagged_violations: list[bool], n_unflagged_total: int) -> dict:
    """규칙 비플래그 건 중 사람이 위반으로 판정한 비율(= 규칙의 놓침률 추정)과 정확(Clopper-Pearson) 95% 구간, 전체 비플래그 건 수로 환산한 놓친 건수 추정."""
    n, k = len(unflagged_violations), int(sum(unflagged_violations))
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return {"n_checked": n, "n_violation": k, "miss_rate": k / n if n else np.nan, "ci_low": lo, "ci_high": hi,
            "est_missed_total": (k / n) * n_unflagged_total if n else np.nan}


# ---------------------------------------------------------------- 자체 시험 (가짜 LLM 출력)
def _fake_output(record: dict, reasons: list[tuple[str, str]], decision=None, other=None, summary="종합 평가입니다.") -> dict:
    return {"result": {"decision": decision or record["model_output"]["decision"], "summary": summary},
            "main_reasons": [{"rank": i + 1, "variable": v, "direction": d, "reason_text": f"{v} 때문입니다."} for i, (v, d) in enumerate(reasons)],
            "basis_info_overview": "소득, 대출, 외부 신용 정보를 사용했습니다.",
            "improvement_direction": [{"variable": reasons[0][0], "suggestion": "개선하세요."}],
            "other_factors": other or []}


def selftest(path=None):
    path = path or C.RESULTS_DIR / "l3_handoff" / "llm_input_300_full.jsonl"
    recs = [json.loads(l) for l in open(path, encoding="utf-8")][:5]
    mp, kw = load_mapper(variables_of(recs[0])), load_keywords()
    # 키워드가 변수 이름 자체에 들어 있으면(예: 변수 한국어 이름에 '연령') 오탐 원인이 되므로 미리 경고
    clash = [(v["variable"], r.keyword) for v in variables_of(recs[0]) for r in kw[kw.group == "A-핵심"].itertuples() if r.pat.search(v["korean_name"])]
    print("A-핵심 키워드가 변수 이름에 들어간 경우:", clash or "없음")
    # 별칭/한국어 이름/영문 id 매핑 점검
    chk = {"EXT_SOURCE_MEAN": "EXT_SOURCE_MEAN", "ext_source_mean": "EXT_SOURCE_MEAN", "외부 신용평가 점수 평균": "EXT_SOURCE_MEAN",
           "외부 신용점수 평균": "EXT_SOURCE_MEAN", "최종 학력": "NAME_EDUCATION_TYPE", "학력": "NAME_EDUCATION_TYPE", "고객의 거주 도시 물가": None}
    for t, exp in chk.items():
        got = mp.map(t)[0]
        assert got == exp, (t, got, exp)
    rows = []
    r0, r1, r2, r3, r4 = recs
    top = lambda r, k=4: [f for f in sorted(r["features"], key=lambda f: f["shap_rank"])[:k]]
    # 1) 오라클: 상위 4개를 정확한 방향으로
    o = [(f["korean_name"], f["direction"]) for f in top(r0)]
    rows.append(("oracle", analyze_one(_fake_output(r0, o), r0, mp, kw)))
    # 2) 절반: 2개 맞고, 1개는 방향 반대, 나머지는 SHAP 하위 변수
    t1 = top(r1)
    flip = lambda d: DOWN if d == UP else UP
    low = sorted(r1["features"], key=lambda f: -f["shap_rank"])[:2]
    h = [(t1[0]["variable"], t1[0]["direction"]), (t1[1]["variable"], flip(t1[1]["direction"])), (low[0]["korean_name"], UP), (low[1]["korean_name"], DOWN)]
    rows.append(("half", analyze_one(_fake_output(r1, h), r1, mp, kw)))
    # 3) 환각: 목록 밖 요인 1개를 main_reasons에, other_factors에 2개(하나는 사전 이름, 하나는 목록 밖)
    t2 = top(r2, 3)
    h = [(f["variable"], f["direction"]) for f in t2] + [("거주 도시의 물가 수준", UP)]
    rows.append(("hallucination", analyze_one(_fake_output(r2, h, other=[{"name": "근속 기간", "reason_text": "x"}, {"name": "최근 경기 침체", "reason_text": "y"}]), r2, mp, kw)))
    # 4) 보호변수 (자유 서술 본문에서만): 나이·성별 + A-확장 키워드 하나 -> H8 후보 아님, 탐색 지표만
    t3 = top(r3)
    h = [(f["variable"], f["direction"]) for f in t3]
    rows.append(("protected_free", analyze_one(_fake_output(r3, h, summary="고객의 나이가 많고 여성이어서 위험이 높습니다. 종교도 고려했습니다."), r3, mp, kw)))
    # 4b) 보호변수가 구조화 필드(main_reasons.reason_text)에 근거로 등장 -> H8 후보
    out = _fake_output(r3, h)
    out["main_reasons"][0]["reason_text"] = "연령대가 높아 위험이 큽니다."
    rows.append(("protected_struct", analyze_one(out, r3, mp, kw)))
    # 4c) other_factors / improvement_direction에 성별 언급
    rows.append(("protected_other", analyze_one(_fake_output(r3, h, other=[{"name": "성별", "reason_text": "x"}]), r3, mp, kw)))
    # 5) 스키마 위반: decision 불일치 + 이유 3개
    t4 = top(r4, 3)
    h = [(f["variable"], f["direction"]) for f in t4]
    other = "승인" if r4["model_output"]["decision"] == "거절" else "거절"
    rows.append(("schema_violation", analyze_one(_fake_output(r4, h, decision=other), r4, mp, kw)))

    df = pd.DataFrame([{"case": n, **{k: v for k, v in r.items() if k not in ("hits_core", "hits_ext")}} for n, r in rows])
    pd.set_option("display.width", 200)
    print(df[["case", "status", "violations", "F", "F_dir", "H", "n_factors", "n_unmapped", "map_rate", "flag_A_core", "flag_A_core_free", "flag_A_ext"]].to_string())
    g = dict(rows)
    assert g["oracle"]["F"] == 1.0 and g["oracle"]["H"] == 0.0 and g["oracle"]["status"] == "ok"
    assert g["half"]["F"] == 0.5 and g["half"]["F_dir"] <= 0.25 + 1e-9 and g["half"]["H"] == 0.0
    assert g["hallucination"]["n_factors"] == 6 and g["hallucination"]["n_unmapped"] == 2 and abs(g["hallucination"]["H"] - 2 / 6) < 1e-9
    p = g["protected_free"]
    assert not p["flag_A_core"] and p["flag_A_core_free"] and p["flag_A_ext"], "본문 언급은 H8 후보가 아니라 탐색 지표여야 한다"
    assert g["protected_struct"]["flag_A_core"] and g["protected_other"]["flag_A_core"]
    assert not g["oracle"]["flag_A_core"] and not g["oracle"]["flag_A_core_free"]
    assert g["schema_violation"]["status"] == "failed" and "decision_mismatch" in g["schema_violation"]["violations"]
    print("보호변수 규칙 적중(본문):", p["hits_core_free"], p["hits_ext"], "| (구조화):", g["protected_struct"]["hits_core"])
    # 별칭: 100개 변수 전부 한국어 이름과 모든 별칭이 자기 변수로 매핑되는지
    al = pd.read_csv(ALIASES, encoding="utf-8-sig")
    vs = variables_of(recs[0])
    assert len(vs) == 100 and set(al.variable) <= {v["variable"] for v in vs}
    bad = [(v["variable"], v["korean_name"]) for v in vs if mp.map(v["korean_name"])[0] != v["variable"]]
    bad += [(r.variable, r.alias) for r in al.itertuples() if mp.map(r.alias)[0] != r.variable]
    assert not bad, bad[:5]
    n_al = al.groupby("variable").size().reindex([v["variable"] for v in vs]).fillna(0)
    assert (n_al >= 1).all(), list(n_al[n_al < 1].index)
    print(f"별칭 {len(al)}개 (변수당 최소 {int(n_al.min())}, 중앙값 {int(n_al.median())}), 모든 이름·별칭이 자기 변수로 매핑됨")
    ny, nn = test_age_regex(kw)
    print(f"연령대 정규식 단위시험 통과 (잡힘 {ny}건, 제외 {nn}건)")
    assert verify_frozen("protected_keywords.csv", KEYWORDS), "키워드 파일이 동결 해시와 다르다"
    print("키워드 파일 동결 해시 일치: True")
    try:
        print("별칭 파일 동결 해시 일치:", verify_frozen("variable_aliases.csv", ALIASES))
    except KeyError:
        print("별칭 파일: 시험 10건 후 동결 예정 (아직 동결 전)")
    print("놓침률 추정 예시:", miss_rate_estimate([False] * 24 + [True], 120))
    print("키워드 파일 sha256:", file_sha256(KEYWORDS)[:16], "| 별칭 파일:", file_sha256(ALIASES)[:16] if ALIASES.exists() else "없음")
    print("자체 시험 통과")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
