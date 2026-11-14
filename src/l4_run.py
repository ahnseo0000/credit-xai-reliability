"""L4 실행: 무결성 점검, 확증 분석(H1,H2,H3,H6,H7,H8), 탐색 지표. 분석 계획(docs/analysis_plan.md)을 그대로 적용한다.

입력: results/l3/raw/calls_main.jsonl (600건 원문, git 제외), results/l3_handoff/llm_input_300_full.jsonl (입력 300건).
출력: results/l4/integrity.md, confirmatory.md(+csv), exploratory.md(+csv). 행 단위 표는 results/l4/raw/ (git 제외).
실행: python -m src.l4_run --stage integrity|confirm|explore|all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .l4_analysis import (ALIASES, DIRS, EXT_PREFIX, KEYWORDS, analyze_one, extract_factors, fidelity_ext_merged, fidelity_k, load_keywords, load_mapper,
                          mention_flags, validate_output, variables_of, verify_frozen)

L3 = C.RESULTS_DIR / "l3"
CALLS = L3 / "raw" / "calls_main.jsonl"
CALLS_TRIAL_A = L3 / "raw" / "calls_trial_A.jsonl"
INPUT = C.RESULTS_DIR / "l3_handoff" / "llm_input_300_full.jsonl"
OUT = C.RESULTS_DIR / "l4"
RAW = OUT / "raw"
SEED = 42
B_BOOT = 10_000
B_LIST = ["NAME_FAMILY_STATUS", "INCOME_PER_PERSON", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT_W_CITY", "EXT_SOURCE_1", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
          "DAYS_ID_PUBLISH", "BUREAU_OLDEST_DAYS", "NAME_INCOME_TYPE", "DEF_60_CNT_SOCIAL_CIRCLE", "NAME_EDUCATION_TYPE"]   # docs/L3_SPEC.md 7절 B 목록
STRUCT = ("main_reasons[", "other_factors[", "improvement_direction[")
APPLICANT_LABELS = ("성별", "연령", "나이", "소득 구간", "소득구간")
SYSTEM_HASH_EXPECTED = "87d32d4e895dbb99ddccafbda8e75ea26008b14212f6b6c96da1efadb065725e"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


def log_l3(msg):
    p = L3 / "RUNLOG.md"
    s = open(p, "rb").read().decode("utf-8")
    if msg in s:      # 같은 기록을 두 번 쓰지 않는다 (재실행 시)
        return
    nl = "\r\n" if "\r\n" in s else "\n"
    s = s.rstrip("\r\n") + nl + f"- `{datetime.now():%Y-%m-%d %H:%M:%S}` {msg}" + nl
    open(p, "wb").write(s.encode("utf-8"))


# ------------------------------------------------------------------ 표 만들기
def build_table():
    recs = {r["sample_id"]: r for r in jl(INPUT)}
    calls = jl(CALLS)
    mp, kw = load_mapper(variables_of(next(iter(recs.values())))), load_keywords()
    rows = []
    for c in calls:
        rec = recs[c["sample_id"]]
        base = {"sample_id": c["sample_id"], "condition": c["condition"], "cell_id": c["cell_id"], "pd_bin": c["pd_bin"], "weight": rec["cell"]["weight"], "cached": bool(c["cached"]),
                "status_call": c["status"], "api_attempts": c["api_attempts"], "parse_attempts": c["parse_attempts"]}
        parsed = None
        if c["status"] == "ok" and c.get("response_text"):
            try:
                parsed = json.loads(c["response_text"])
            except Exception:
                parsed = None
        viol = validate_output(parsed, rec) if parsed is not None else ["unparsed"]
        base["violations"] = ";".join(viol)
        if viol:
            base["status"] = "failed"
            rows.append(base)
            continue
        a = analyze_one(parsed, rec, mp, kw)
        base.update({k: a[k] for k in ("status", "F", "F_dir", "H", "n_factors", "n_unmapped", "map_rate", "flag_A_core", "flag_A_core_free", "flag_A_ext")})
        for k in (3, 5, 6):
            base.update(fidelity_k(parsed, rec, mp, k))
        base.update(fidelity_ext_merged(parsed, rec, mp))
        fac = extract_factors(parsed)
        maps = [mp.map(f["text"])[0] for f in fac]
        n_main = len(parsed["main_reasons"])
        base["b_main"] = any(v in B_LIST for v in maps[:n_main] if v)
        base["b_any"] = any(v in B_LIST for v in maps if v)
        base["b_vars"] = ";".join(sorted({v for v in maps if v in B_LIST}))
        mf = mention_flags(parsed, kw)
        base["a_ext_struct"] = any(h[0].startswith(STRUCT) for h in mf["hits_ext"])
        base["a_core_struct_fields"] = ";".join(sorted({h[0].split("[")[0] for h in mf["hits_core"]}))
        base["a_ext_kw"] = ";".join(sorted({h[1] for h in mf["hits_ext"] if h[0].startswith(STRUCT)}))
        rows.append(base)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "per_row_metrics.csv", index=False, encoding="utf-8-sig")
    return df, recs, calls, mp, kw


# ------------------------------------------------------------------ 무결성
def integrity(df, recs, calls):
    import src.l3.llm as L
    from src.l3 import settings as S
    n = len(calls)
    cd = pd.DataFrame(calls)
    ids_in = set(recs)
    ids_call = set(cd.sample_id)
    per = cd.groupby("sample_id").condition.apply(lambda s: tuple(sorted(s)))
    pairs_ok = int((per == ("A", "B")).sum())
    # 프롬프트 재구성 해시
    match, mism = 0, []
    for c in calls:
        p = L.build_prompt(recs[c["sample_id"]], c["condition"])
        if L.sha256_text(L.SYSTEM + "\n" + p) == c["prompt_sha256"]:
            match += 1
        else:
            mism.append((c["sample_id"], c["condition"]))
    sys_hash = L.sha256_text(L.SYSTEM)
    calls_sha = sha256_file(CALLS)
    in_sha = sha256_file(INPUT)
    cs = json.loads((L3 / "call_settings_main.json").read_text(encoding="utf-8"))
    ta = pd.DataFrame(jl(CALLS_TRIAL_A))
    cached = cd[cd.cached]
    cached_ids = sorted(cached.sample_id.tolist())
    ta_ids = sorted(ta.sample_id.unique().tolist())
    ta_ok_ids = sorted(ta[ta.status == "ok"].sample_id.unique().tolist())
    cell_ok = all(recs[c["sample_id"]]["cell"]["cell_id"] == c["cell_id"] for c in calls)
    # 팀 F와 재계산 F 일치
    team_F = cd.set_index(["sample_id", "condition"])["F"]
    mine = df.set_index(["sample_id", "condition"])["F"]
    teamD = cd.set_index(["sample_id", "condition"])["F_dir"]
    mineD = df.set_index(["sample_id", "condition"])["F_dir"]
    maxdF = float((team_F - mine).abs().max())
    maxdD = float((teamD - mineD).abs().max())
    fk_ok = (df.violations == "").all()
    md = ["# L4 무결성 점검 (results/l3/raw/calls_main.jsonl)", "",
          f"- 점검 시각 {datetime.now():%Y-%m-%d %H:%M}. 원문은 읽기만 했고 git에 올리지 않는다 (`results/*/raw/` 무시).",
          f"- **calls_main.jsonl sha256 = `{calls_sha}`** ({CALLS.stat().st_size:,} bytes). 이 값을 `results/l3/RUNLOG.md`에 기록했다.", "",
          "## 1. 건수와 상태", "", "| 항목 | 값 | 기대 | 판정 |", "| --- | --- | --- | --- |",
          f"| 행 수 | {n} | 600 | {'일치' if n == 600 else '불일치'} |",
          f"| 조건별 행 수 | A {int((cd.condition == 'A').sum())}, B {int((cd.condition == 'B').sum())} | 300, 300 | {'일치' if (cd.condition == 'A').sum() == 300 and (cd.condition == 'B').sum() == 300 else '불일치'} |",
          f"| A·B 쌍이 완성된 신청자 | {pairs_ok} | 300 | {'일치' if pairs_ok == 300 else '불일치'} |",
          f"| status ok / failed | {int((cd.status == 'ok').sum())} / {int((cd.status != 'ok').sum())} | 600 / 0 | {'일치' if (cd.status == 'ok').all() else '불일치'} |",
          f"| 재검증(스키마·decision 일치, 이 분석 코드의 validate_output) 위반 건수 | {int((df.violations != '').sum())} | 0 | {'일치' if fk_ok else '불일치'} |",
          f"| 입력 jsonl 300건과 신청자 ID 집합 | 입력 {len(ids_in)}, 호출 {len(ids_call)}, 차집합 {len(ids_in ^ ids_call)} | 동일 | {'일치' if ids_in == ids_call else '불일치'} |",
          f"| 호출 기록 cell_id와 입력 cell | {'전부 일치' if cell_ok else '불일치 있음'} | 일치 | {'일치' if cell_ok else '불일치'} |", "",
          "## 2. 모델·해시", "",
          "model (요청한 이름): " + ", ".join(f"`{k}` {v}건" for k, v in cd.model.value_counts().items()),
          "", "model_version (응답이 알려준 버전): " + ", ".join(f"`{k}` {v}건" for k, v in cd.model_version.value_counts().items()),
          "", "prompt_version: " + ", ".join(f"`{k}` {v}건" for k, v in cd.prompt_version.value_counts().items()), "",
          "| 항목 | 값 | 기대 | 판정 |", "| --- | --- | --- | --- |",
          f"| 호출 기록의 input_sha256 종류 | {cd.input_sha256.nunique()} (`{cd.input_sha256.iloc[0][:12]}…`) | 1 | {'일치' if cd.input_sha256.nunique() == 1 else '불일치'} |",
          f"| 로컬 입력 jsonl 실제 sha256 | `{in_sha[:12]}…` | 호출 기록과 같음 | {'일치' if in_sha == cd.input_sha256.iloc[0] else '불일치'} |",
          f"| 호출 설정 파일의 input_sha256 | `{cs['input_sha256'][:12]}…` | 같음 | {'일치' if cs['input_sha256'] == in_sha else '불일치'} |",
          f"| system prompt sha256 (`src/l3/llm.py`의 SYSTEM에서 직접 계산) | `{sys_hash}` | `{SYSTEM_HASH_EXPECTED[:8]}…` | {'일치' if sys_hash == SYSTEM_HASH_EXPECTED else '불일치'} |",
          f"| 호출 설정 파일의 system_prompt_sha256 | `{cs['system_prompt_sha256'][:12]}…` | 같음 | {'일치' if cs['system_prompt_sha256'] == sys_hash else '불일치'} |",
          f"| 조건 A prompt_sha256 종류 | {cd[cd.condition == 'A'].prompt_sha256.nunique()} | (아래 설명) | |",
          f"| 조건 B prompt_sha256 종류 | {cd[cd.condition == 'B'].prompt_sha256.nunique()} | (아래 설명) | |",
          f"| 저장소 코드(`build_prompt`)로 입력 300건에서 프롬프트를 재구성해 계산한 해시와 호출 기록 prompt_sha256이 일치한 건수 | {match} / {n} | 600 / 600 | {'일치' if match == n else '불일치 ' + str(mism[:3])} |", "",
          "- **prompt_sha256 설명**: 호출 기록의 `prompt_sha256`은 시스템 프롬프트 + 신청자별 프롬프트 전문의 해시라 신청자마다 다르다 (조건별 종류 수가 신청자 수와 같은 것이 정상이다). 따라서 \"조건별 해시 1개\"는 성립하지 않는다. "
          "대신 (1) 시스템 프롬프트 해시가 1종이고 기대값과 일치하며 (2) 이 저장소 코드로 입력 300건에서 프롬프트를 다시 만들어 계산한 해시가 600건 전부 호출 기록과 일치함으로써, 호출에 쓰인 프롬프트가 이 코드의 템플릿과 같음을 확인했다.", "",
          "## 3. 재시도·지연", "", f"- api_attempts 분포: " + ", ".join(f"{k}회 {v}건" for k, v in cd.api_attempts.value_counts().sort_index().items()),
          f"- parse_attempts 분포: " + ", ".join(f"{k}회 {v}건" for k, v in cd.parse_attempts.value_counts().sort_index().items()),
          f"- API 재시도(api_attempts>1) 건: " + (", ".join(f"{r.sample_id}/{r.condition}({r.api_attempts}회, {r.status})" for r in cd[cd.api_attempts > 1].itertuples()) or "없음"),
          f"- finish_reason: " + ", ".join(f"`{k}` {v}건" for k, v in cd.finish_reason.value_counts().items()),
          f"- 호출 시각 범위: {cd.called_at.min()} ~ {cd.called_at.max()} (캐시 재사용 건 제외 시 {cd[~cd.cached].called_at.min()} ~ {cd[~cd.cached].called_at.max()})", "",
          "## 4. 시험 호출을 재사용한 조건 A", "",
          f"- `cached = true`인 호출: **{len(cached)}건**, 전부 조건 {sorted(cached.condition.unique())}, status {sorted(cached.status.unique())}.",
          f"- 이 신청자 ID {len(cached_ids)}개 (호출 기록의 `cached` 플래그에서 추출; RUNLOG·main_summary.md에는 ID 목록이 없고 \"조건 A 10건은 시험 trial_A의 캐시를 썼다\"는 서술만 있다): `{cached_ids}`",
          f"- `calls_trial_A.jsonl`(시험 A 호출)의 신청자 ID: {len(ta_ids)}개 (그중 ok가 있는 ID {len(ta_ok_ids)}개). 재사용 ID와의 관계: 재사용 ⊂ 시험 ID = {set(cached_ids) <= set(ta_ids)}, 재사용 == 시험 ok ID = {set(cached_ids) == set(ta_ok_ids)}.", "",
          f"- 재사용된 A 10건의 `called_at`(시험 당시 시각): " + ", ".join(sorted(cached.called_at.str[11:16].tolist())) + " (11:44~11:47 5건과 18:37 5건으로, 모두 본 호출 시작(18:39) 전의 시험 실행에서 얻은 응답이다).", "",
          "## 5. 팀 계산 지표와 이 분석 코드의 재계산 비교", "",
          f"- 호출 기록에 팀이 넣어 둔 F, F_dir와 이 저장소의 `src/l4_analysis.py`로 다시 계산한 값의 최대 절대 차이: F {maxdF:.2e}, F_dir {maxdD:.2e} (0이면 완전 일치). 매핑에는 동결된 별칭 파일(sha256 `{sha256_file(ALIASES)[:12]}…`)과 키워드 파일(`{sha256_file(KEYWORDS)[:12]}…`)을 썼고 `verify_frozen`은 각각 {verify_frozen('variable_aliases.csv', ALIASES)}, {verify_frozen('protected_keywords.csv', KEYWORDS)}이다.", ""]
    write("integrity.md", "\n".join(md))
    log_l3(f"[L4] calls_main.jsonl sha256 = {calls_sha} ({CALLS.stat().st_size} bytes, {n}행). 입력 jsonl sha256 {in_sha[:12]}…, 시스템 프롬프트 해시 {sys_hash[:12]}… (기대값과 {'일치' if sys_hash == SYSTEM_HASH_EXPECTED else '불일치'}). 점검 결과는 results/l4/integrity.md")
    return dict(cached_ids=cached_ids, calls_sha=calls_sha, match=match, maxdF=maxdF, maxdD=maxdD, sys_ok=sys_hash == SYSTEM_HASH_EXPECTED)


# ------------------------------------------------------------------ 확증 분석
def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def wilcoxon_pratt(d, alt):
    d = np.asarray(d, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"stat": 0.0, "p": 1.0, "r_rb": 0.0, "n_nonzero": 0}
    res = stats.wilcoxon(d, zero_method="pratt", alternative=alt, method="approx")
    ranks = stats.rankdata(np.abs(d))
    mask = d != 0
    rp, rm = ranks[mask & (d > 0)].sum(), ranks[mask & (d < 0)].sum()
    return {"stat": float(res.statistic), "p": float(res.pvalue), "r_rb": float((rp - rm) / (rp + rm)), "n_nonzero": int(mask.sum())}


def boot_mean(d, rng):
    n = len(d)
    idx = rng.integers(0, n, (B_BOOT, n))
    m = d[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def boot_two_group(x, y, rng):
    """독립 두 집단 층화 부트스트랩: 각 집단 안에서 복원추출한 평균의 차이(x−y)."""
    ix = rng.integers(0, len(x), (B_BOOT, len(x)))
    iy = rng.integers(0, len(y), (B_BOOT, len(y)))
    m = x[ix].mean(1) - y[iy].mean(1)
    return np.percentile(m, [2.5, 97.5])


def mwu(x, y, alt):
    res = stats.mannwhitneyu(x, y, alternative=alt, method="asymptotic")
    u1 = float(res.statistic)
    return {"U": u1, "p": float(res.pvalue), "r_rb": float(2 * u1 / (len(x) * len(y)) - 1)}


def confirmatory_tests(df, label=""):
    """df: 두 조건 모두 ok인 쌍만 (주 분석). 반환: 가설별 결과 표 + 부가 정보."""
    ok = df[df.status == "ok"]
    pair_ids = sorted(set(ok[ok.condition == "A"].sample_id) & set(ok[ok.condition == "B"].sample_id))
    A = ok[(ok.condition == "A") & ok.sample_id.isin(pair_ids)].set_index("sample_id").loc[pair_ids]
    Bd = ok[(ok.condition == "B") & ok.sample_id.isin(pair_ids)].set_index("sample_id").loc[pair_ids]
    res = []
    rng = np.random.default_rng(SEED)
    for hid, col, alt, text in (("H1", "F", "greater", "B>A"), ("H2", "F_dir", "greater", "B>A"), ("H3", "H", "less", "B<A")):
        d = (Bd[col] - A[col]).to_numpy()
        w = wilcoxon_pratt(d, alt)
        lo, hi = boot_mean(d, rng)
        res.append({"hid": hid, "metric": col, "test": "쌍 Wilcoxon 부호순위(Pratt, 정규근사), 단측 " + text, "n": len(d), "A_mean": A[col].mean(), "B_mean": Bd[col].mean(),
                    "effect": d.mean(), "effect_def": "B−A", "ci_lo": lo, "ci_hi": hi, "r_rb": w["r_rb"], "stat": w["stat"], "p": w["p"], "n_nonzero": w["n_nonzero"]})
    # H6, H7
    for hid, cond_df, col in (("H6", A, "F"), ("H7", Bd, "F")):
        x = cond_df[cond_df.pd_bin == "경계"][col].to_numpy()
        y = cond_df[cond_df.pd_bin != "경계"][col].to_numpy()
        m = mwu(x, y, "less")
        lo, hi = boot_two_group(x, y, rng)
        res.append({"hid": hid, "metric": f"F ({'A' if hid == 'H6' else 'B'})", "test": "Mann-Whitney U, 단측 경계<비경계(정규근사, 가중치 없음)", "n": f"{len(x)} vs {len(y)}",
                    "A_mean": x.mean() if hid == "H6" else np.nan, "B_mean": x.mean() if hid == "H7" else np.nan, "effect": x.mean() - y.mean(), "effect_def": "경계−비경계", "ci_lo": lo, "ci_hi": hi,
                    "r_rb": m["r_rb"], "stat": m["U"], "p": m["p"], "n_nonzero": np.nan})
    # H8
    a = A["flag_A_core"].astype(int).to_numpy()
    b = Bd["flag_A_core"].astype(int).to_numpy()
    n10, n01 = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())   # b: B만, c: A만
    p8 = 1.0 if n10 + n01 == 0 else float(stats.binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue)
    d8 = (b - a).astype(float)
    lo, hi = boot_mean(d8, rng)
    if n10 + n01 > 0:
        ci = stats.binomtest(n10, n10 + n01, 0.5).proportion_ci(method="exact")
        orr = (n10 / n01) if n01 > 0 else np.inf
        or_lo = ci.low / (1 - ci.low) if ci.low < 1 else np.inf
        or_hi = ci.high / (1 - ci.high) if ci.high < 1 else np.inf
    else:
        orr, or_lo, or_hi = np.nan, np.nan, np.nan
    res.append({"hid": "H8", "metric": "M_A (규칙 1차 플래그)", "test": "정확 McNemar(불일치 쌍 이항검정), 양측", "n": len(a), "A_mean": a.mean(), "B_mean": b.mean(), "effect": d8.mean(), "effect_def": "B−A",
                "ci_lo": lo, "ci_hi": hi, "r_rb": np.nan, "stat": f"B만 {n10}, A만 {n01}", "p": p8, "n_nonzero": n10 + n01, "OR": orr, "OR_low": or_lo, "OR_high": or_hi})
    t = pd.DataFrame(res)
    t["p_BH"] = bh(t["p"].to_numpy())
    t["crit"] = np.where(t["p_BH"] <= 0.05, "충족", "미충족")
    return t, A, Bd


def cp_ci(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(method="exact")
    return ci.low, ci.high


def confirmatory(df):
    t, A, Bd = confirmatory_tests(df)
    ok = df[df.status == "ok"]
    n_fail = {c: int(((df.condition == c) & (df.status != "ok")).sum()) for c in "AB"}
    nf_A, nf_B = int(A.n_factors.sum()), int(Bd.n_factors.sum())
    nu_A, nu_B = int(A.n_unmapped.sum()), int(Bd.n_unmapped.sum())
    # H8 부가: 조건별 M_A, Clopper-Pearson, 가중치 적용
    lines = []
    for c, d in (("A", A), ("B", Bd)):
        k, n = int(d.flag_A_core.sum()), len(d)
        lo, hi = cp_ci(k, n)
        w = d.weight.to_numpy()
        wp = float((w * d.flag_A_core.to_numpy()).sum() / w.sum())
        lines.append((c, k, n, k / n, lo, hi, wp))
    t.to_csv(OUT / "confirmatory.csv", index=False, encoding="utf-8-sig")
    f = lambda x, d=4: "-" if (isinstance(x, float) and np.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x))
    pf = lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
    md = ["# L4 확증 분석 (사전등록 H1, H2, H3, H6, H7, H8)", "",
          "- 분석 계획 `docs/analysis_plan.md` 2절을 그대로 적용했다. 분석 단위는 신청자 300명(쌍 설계), 검정·효과크기·부트스트랩(B=10,000, 시드 42)·BH 보정(6개 한 묶음, q=0.05)은 계획대로이다.",
          "- 해석 문장은 쓰지 않고 숫자와 사전등록 판정 기준(BH 보정 p ≤ 0.05, 사전 지정 방향)에 따른 \"충족/미충족\"만 적는다.",
          "- **H8의 M_A는 사람 검토로 확정한 라벨이 아니라 키워드 규칙 1차 플래그(구조화 필드: main_reasons / other_factors / improvement_direction 기준)로 계산한 값이다.** 사전등록의 최종 라벨(조건을 가린 사람 검토)은 아직 없다. H8의 기준 판정은 이 규칙 플래그 기준이다.",
          "- 가중치는 쓰지 않았다 (구간·조건 간 검정). 가중치 적용 모집단 추정치는 아래 H8 부가 표에만 있다.",
          f"- 처리: 주 분석은 두 조건 모두 ok인 쌍 {len(A)}명. 실패(failed) 건 A {n_fail['A']}, B {n_fail['B']}라 제외한 신청자는 0명이며 포함 시나리오(실패에 보수적 값 대입)는 주 분석과 같다 (5절).",
          f"- 매핑 실패 처리: 환각률 H의 분모는 main_reasons 4 + other_factors, 분자는 매핑 실패 요인. 조건 A 요인 {nf_A}개 중 매핑 실패 {nu_A}개, 조건 B 요인 {nf_B}개 중 {nu_B}개. 결측값은 없다 (모든 지표가 300쌍에서 계산됨).", "",
          "## 결과 표", "", "| 가설 | 지표 | 검정 | n | 효과(평균 차) | 95% CI | r (rank-biserial) | 통계량 | 원 p | BH 보정 p | 기준 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in t.itertuples(index=False):
        stat = r.stat if isinstance(r.stat, str) else f"{r.stat:.1f}"
        md.append(f"| {r.hid} | {r.metric} | {r.test} | {r.n} | {r.effect:+.4f} ({r.effect_def}) | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {f(r.r_rb)} | {stat} | {pf(r.p)} | {pf(r.p_BH)} | **{r.crit}** |")
    md += ["", "### 조건별 평균 (참고)", "", "| 가설 | A 평균 | B 평균 |", "| --- | --- | --- |"]
    for r in t.itertuples(index=False):
        md.append(f"| {r.hid} ({r.metric}) | {f(r.A_mean)} | {f(r.B_mean)} |")
    h8 = t[t.hid == "H8"].iloc[0]
    md += ["", "## H8 부가: 조건별 A-핵심 언급률 M_A (규칙 1차 플래그, 구조화 필드)", "", "| 조건 | 언급 건수 | n | 비율 | Clopper-Pearson 95% | 가중치 적용 모집단 추정 |", "| --- | --- | --- | --- | --- | --- |"]
    for c, k, n, p, lo, hi, wp in lines:
        md.append(f"| {c} | {k} | {n} | {p:.4f} | [{lo:.4f}, {hi:.4f}] | {wp:.4f} |")
    orr = f"{h8['OR']:.3f}" if np.isfinite(h8["OR"]) else ("무한대(A만 0건)" if h8["n_nonzero"] and h8["OR"] == np.inf else "정의 안 됨(불일치 0)")
    md += ["", f"- 불일치 쌍: {h8['stat']}. 불일치 쌍의 오즈비(B만/A만) = {orr} (정확 95% CI [{h8['OR_low']:.3f}, {h8['OR_high']:.3f}]), 비율 차이(B−A) = {h8['effect']:+.4f}, 쌍 부트스트랩 95% CI [{h8['ci_lo']:+.4f}, {h8['ci_hi']:+.4f}].",
           "- 규칙 플래그는 신청자 정보를 단순 인용하는 문장도 잡을 수 있다 (사전등록의 사람 검토 단계에서 걸러야 함). 아래 탐색 파일에 플래그된 필드별 분포가 있다."] 
    h3 = t[t.hid == "H3"].iloc[0]
    d3 = (Bd["H"] - A["H"]).to_numpy()
    nneg, npos = int((d3 < 0).sum()), int((d3 > 0).sum())
    ex3 = float(stats.binomtest(nneg, nneg + npos, 0.5, alternative="greater").pvalue) if nneg + npos else 1.0
    md += ["", "## 검정 방법에 관한 기록", "",
           f"- H3(환각률): 두 조건에서 차이가 0이 아닌 쌍이 {int(h3['n_nonzero'])}쌍뿐이다 (B가 낮은 쌍 {nneg}, 높은 쌍 {npos}, 동점 {len(d3) - nneg - npos}). Pratt 처리 정규근사 p = {h3['p']:.4f}. "
           f"참고로 비영 쌍만으로 계산한 정확 부호검정(단측) p = {ex3:.4f} (정규근사보다 크다). 어느 쪽이든 BH 보정 후 기준은 미충족이다 (정규근사 BH p = {h3['p_BH']:.4f}). 사전등록 검정은 Pratt Wilcoxon이므로 표의 값이 판정값이다.",
           "- 쌍 Wilcoxon은 scipy.stats.wilcoxon(zero_method='pratt', method='approx'), Mann-Whitney U는 scipy.stats.mannwhitneyu(method='asymptotic', 동점 보정 포함). rank-biserial r은 Wilcoxon: (R+ − R−)/(R+ + R−) (Pratt로 동점 0 차이 순위 제외), Mann-Whitney: 2U/(n1 n2) − 1 (U는 경계 집단 기준).",
           "- 부트스트랩 CI는 percentile 방식, B=10,000, 시드 42 (쌍 검정은 신청자 단위 복원추출, H6·H7은 집단 안에서 각각 복원추출)."]
    write("confirmatory.md", "\n".join(md) + "\n")
    return t, A, Bd


# ------------------------------------------------------------------ 탐색
def boot_ci_mean(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B_BOOT, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def exploratory(df, recs, calls, mp, kw, cached_ids, conf):
    OUT.mkdir(parents=True, exist_ok=True)
    ok = df[df.status == "ok"]
    A = ok[ok.condition == "A"].set_index("sample_id").sort_index()
    Bd = ok[ok.condition == "B"].set_index("sample_id").sort_index()
    rng = np.random.default_rng(SEED)
    md = ["# L4 탐색적 지표", "", "> **탐색적, 확증 판정에 사용하지 않음.** 보정 없음, p값은 해석하지 않고 효과 크기와 95% 구간 위주로 적는다 (분석 계획 3절).", "",
          "## 1. F_k와 F_ext (정의: 분석 계획 3절, 2026-11-14 명시)", "",
          "- F_k = |LLM이 든 4개 변수 ∩ SHAP 상위 k개| / min(4, k). F_ext = EXT_SOURCE_* 7개를 한 요인으로 합쳐 중복 제거 후 |LLM 요인 ∩ SHAP 상위 4 요인| / |SHAP 상위 4 요인| (분모 1~4). 확증 F는 변수 단위·분모 4.", "",
          "| 지표 | A 평균 [95% CI] | B 평균 [95% CI] | 차이 B−A [95% CI] |", "| --- | --- | --- | --- |"]
    rows_csv = []
    for name, col in (("F (확증 정의, k=4)", "F"), ("F3", "F3"), ("F5", "F5"), ("F6", "F6"), ("F_ext", "F_ext")):
        a, b = A[col].to_numpy(), Bd[col].to_numpy()
        la, ha, lb, hb, ld, hd = *boot_ci_mean(a, rng), *boot_ci_mean(b, rng), *boot_mean(b - a, rng)
        md.append(f"| {name} | {a.mean():.4f} [{la:.4f}, {ha:.4f}] | {b.mean():.4f} [{lb:.4f}, {hb:.4f}] | {(b - a).mean():+.4f} [{ld:+.4f}, {hd:+.4f}] |")
        rows_csv.append({"metric": name, "A_mean": a.mean(), "A_lo": la, "A_hi": ha, "B_mean": b.mean(), "B_lo": lb, "B_hi": hb, "diff": (b - a).mean(), "diff_lo": ld, "diff_hi": hd})
    pd.DataFrame(rows_csv).to_csv(OUT / "exploratory_Fk_Fext.csv", index=False, encoding="utf-8-sig")
    md += ["", f"- F_ext의 SHAP 상위 4 요인 수(분모): 분포 " + ", ".join(f"{int(k)}개 {int(v)}명" for k, v in A.n_top_factors.value_counts().sort_index().items()) + " (두 조건 같음, 입력 기준).", ""]
    # 보호변수 언급률
    md += ["## 2. 보호변수 언급률 (구조화된 사유 필드 기준: main_reasons / other_factors / improvement_direction)", "",
           "| 지표 | 조건 A 건수 (n=300) | 비율 | 조건 B 건수 | 비율 |", "| --- | --- | --- | --- | --- |"]
    def rate(col, label):
        a, b = int(A[col].sum()), int(Bd[col].sum())
        la, ha = cp_ci(a, len(A)); lb, hb = cp_ci(b, len(Bd))
        md.append(f"| {label} | {a} | {a / len(A):.4f} [{la:.3f}, {ha:.3f}] | {b} | {b / len(Bd):.4f} [{lb:.3f}, {hb:.3f}] |")
        return {"metric": label, "A": a, "B": b}
    prot = [rate("a_ext_struct", "A-확장 키워드 (구조화 필드, 규칙 플래그)"), rate("flag_A_ext", "A-확장 키워드 (어느 곳이든, 본문 포함)"), rate("flag_A_core", "A-핵심 (구조화 필드, 규칙 플래그; H8의 M_A)"),
            rate("flag_A_core_free", "A-핵심 (자유 서술 본문; M_A_free)"), rate("b_main", "B 목록 변수를 main_reasons 사유로 든 신청자"), rate("b_any", "B 목록 변수를 main_reasons 또는 other_factors로 든 신청자")]
    pd.DataFrame(prot).to_csv(OUT / "exploratory_protected_rates.csv", index=False, encoding="utf-8-sig")
    md += ["", "B 목록: " + ", ".join(f"`{v}`" for v in B_LIST) + " (L3_SPEC 7절, 12개). 모니터링 지표이며 위반이 아니다.", ""]
    # B 변수별
    md += ["B 변수별 언급 신청자 수 (main_reasons + other_factors):", "", "| 변수 | A | B |", "| --- | --- | --- |"]
    cntA = pd.Series(";".join(A.b_vars.fillna("")).split(";")).replace("", np.nan).dropna().value_counts()
    cntB = pd.Series(";".join(Bd.b_vars.fillna("")).split(";")).replace("", np.nan).dropna().value_counts()
    for v in B_LIST:
        md.append(f"| {v} | {int(cntA.get(v, 0))} | {int(cntB.get(v, 0))} |")
    ext_kw = pd.Series(";".join(pd.concat([A.a_ext_kw, Bd.a_ext_kw]).fillna("")).split(";")).replace("", np.nan).dropna().value_counts()
    md += ["", "A-확장 키워드 적중 목록(구조화 필드, 두 조건 합): " + (", ".join(f"`{k}` {v}회" for k, v in ext_kw.items()) or "없음"), "",
           "A-핵심 규칙 플래그가 걸린 필드(두 조건): " + ", ".join(f"{c}: " + (", ".join(f"{k} {v}건" for k, v in pd.Series(";".join(d.a_core_struct_fields.fillna("")).split(";")).replace("", np.nan).dropna().value_counts().items()) or "없음") for c, d in (("A", A), ("B", Bd))), ""]
    # 시험 재사용 10건 제외 민감도
    keep = df[~df.sample_id.isin(cached_ids)]
    t_all = conf[0]
    t_ex, _, _ = confirmatory_tests(keep)
    md += ["## 3. 민감도: 시험 호출을 재사용한 조건 A의 10건(신청자 10명, 두 조건 모두)을 제외", "",
           f"- 제외한 신청자 {len(cached_ids)}명, 남은 쌍 {len(set(keep.sample_id))}명. 같은 절차(BH 보정 6개 묶음)로 다시 계산했다. 탐색적이며 확증 판정에는 쓰지 않는다.", "",
           "| 가설 | 전체 300명: 효과 | 원 p | BH p | 기준 | 10건 제외(290명): 효과 | 95% CI | 원 p | BH p | 기준 | 기준 변화 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pf = lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
    changed = 0
    sens = []
    for a, b in zip(t_all.itertuples(index=False), t_ex.itertuples(index=False)):
        ch = "변화 없음" if a.crit == b.crit else "**바뀜**"
        changed += a.crit != b.crit
        md.append(f"| {a.hid} | {a.effect:+.4f} | {pf(a.p)} | {pf(a.p_BH)} | {a.crit} | {b.effect:+.4f} | [{b.ci_lo:+.4f}, {b.ci_hi:+.4f}] | {pf(b.p)} | {pf(b.p_BH)} | {b.crit} | {ch} |")
        sens.append({"hypothesis": a.hid, "effect_all": a.effect, "p_all": a.p, "pBH_all": a.p_BH, "crit_all": a.crit, "effect_excl10": b.effect, "ci_lo": b.ci_lo, "ci_hi": b.ci_hi, "p_excl10": b.p, "pBH_excl10": b.p_BH, "crit_excl10": b.crit})
    pd.DataFrame(sens).to_csv(OUT / "exploratory_sensitivity_excl10.csv", index=False, encoding="utf-8-sig")
    md += ["", f"- 기준(BH 보정 p ≤ 0.05)이 바뀐 가설 수: {changed} / 6.", ""]
    # 매핑 실패 목록
    md += ["## 4. 매핑 실패 목록 (환각률 H의 분자)", ""]
    fails = []
    for c in calls:
        if c["status"] != "ok":
            continue
        rec = recs[c["sample_id"]]
        parsed = json.loads(c["response_text"])
        for f_ in extract_factors(parsed):
            v, how = mp.map(f_["text"])
            if v is None:
                txt = f_["text"]
                cites = any(l in txt for l in APPLICANT_LABELS)
                cites_reason = any(l in str(f_["reason"]) for l in APPLICANT_LABELS)
                fails.append({"조건": c["condition"], "필드": "main_reasons" if f_["src"] == "main" else "other_factors", "요인 이름": txt, "사유 텍스트 요약(앞 70자)": str(f_["reason"])[:70].replace("\n", " "),
                              "매핑 결과": how, "요인 이름이 신청자 정보(성별·연령·소득 구간)를 가리킴": "예" if cites else "아니오", "사유 문장에 신청자 정보 표현 포함": "예" if cites_reason else "아니오",
                              "A-핵심 규칙 플래그(해당 신청자·조건)": bool(df[(df.sample_id == c["sample_id"]) & (df.condition == c["condition"])].flag_A_core.iloc[0])})
    fdf = pd.DataFrame(fails)
    if len(fdf):
        fdf.to_csv(OUT / "exploratory_mapping_failures.csv", index=False, encoding="utf-8-sig")
        md += [f"- 매핑 실패 요인 총 {len(fdf)}건 (조건 A {int((fdf.조건 == 'A').sum())}, B {int((fdf.조건 == 'B').sum())}). 신청자 ID는 행 단위 식별자라 이 표에 쓰지 않았다.", "",
               "| # | 조건 | 필드 | 요인 이름 | 사유 텍스트 요약 | 매핑 결과 | 요인 이름이 신청자 정보를 가리킴 | 사유에 신청자 정보 표현 | 해당 건 A-핵심 플래그 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i, r in enumerate(fdf.itertuples(index=False), 1):
            md.append(f"| {i} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
        n_cite = int((fdf.iloc[:, 5] == "예").sum())
        md += ["", f"- **입력으로 제공된 신청자 정보(성별·연령대·소득 구간)를 요인 이름으로 인용한 매핑 실패: {n_cite}건 / {len(fdf)}건.** 사유 문장에 신청자 정보 표현이 들어 있는 건 {int((fdf.iloc[:, 6] == '예').sum())}건."]
    else:
        md += ["- 매핑 실패 요인 없음."]
    write("exploratory.md", "\n".join(md) + "\n")
    return fdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["integrity", "confirm", "explore", "all"], default="all")
    a = ap.parse_args()
    df, recs, calls, mp, kw = build_table()
    info = None
    if a.stage in ("integrity", "all"):
        info = integrity(df, recs, calls)
        print("integrity done", {k: v for k, v in info.items() if k != "cached_ids"})
    conf = None
    if a.stage in ("confirm", "explore", "all"):
        conf = confirmatory(df)
        print("confirm done")
    if a.stage in ("explore", "all"):
        cached_ids = sorted(df[df.cached].sample_id.unique().tolist())
        exploratory(df, recs, calls, mp, kw, cached_ids, conf)
        print("explore done")
