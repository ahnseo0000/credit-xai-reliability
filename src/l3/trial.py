"""L3-5 시험 10건 (L3_SPEC 8절): 가짜 SHAP으로 조건 B.

가짜 SHAP = 같은 레코드의 SHAP 값을 변수 사이에서 무작위로 섞은 것 (합이 같아 pd·decision은 그대로).
목적: (1) 파이프라인·스키마가 도는지, (2) 응답이 모형의 진짜 SHAP이 아니라 입력으로 준 가짜 SHAP을 따르는지.
통과 기준: 재시도 후 파싱·스키마 성공 10/10, 변수 매핑률 90% 이상(목표). 600건 본 호출은 이 파일이 하지 않는다.

실행:
  python -m src.l3.trial --list-models                 # 키로 쓸 수 있는 모델 이름 확인 (preview/exp 아닌 것 고르기)
  python -m src.l3.trial --model <모델 이름> --dry-run   # 프롬프트만 저장 (API 호출 없음)
  python -m src.l3.trial --model <모델 이름>             # 10건 호출 + 보고서 results/l3/trial_L3-5.md
  (선택) --price-in 0.30 --price-out 2.50              # 유료 단가(USD / 1M 토큰)를 주면 비용 환산. 무료 티어면 실제 청구 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime

import pandas as pd

from . import llm
from . import settings as S

N_TRIAL = 10
TRIAL_SEED = 42


def fake_shap(rec: dict, seed: int) -> dict:
    r = copy.deepcopy(rec)
    vals = [f["shap"] for f in r["features"]]
    random.Random(seed).shuffle(vals)
    order = sorted(range(len(vals)), key=lambda j: -abs(vals[j]))
    rank = {j: k for k, j in enumerate(order, 1)}
    for j, f in enumerate(r["features"]):
        f["real_shap_rank"] = f["shap_rank"]
        f["shap"], f["shap_rank"] = vals[j], rank[j]
        f["direction"] = llm.DIR_UP if vals[j] > 1e-4 else (llm.DIR_DOWN if vals[j] < -1e-4 else llm.DIR_NONE)
    return r


def pick(records: list[dict]) -> list[dict]:
    ids = sorted(r["sample_id"] for r in records)
    chosen = set(random.Random(TRIAL_SEED).sample(ids, N_TRIAL))
    return [r for r in sorted(records, key=lambda r: r["sample_id"]) if r["sample_id"] in chosen]


def list_models():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(S.ROOT / ".env")
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for m in c.models.list():
        acts = getattr(m, "supported_actions", None) or []
        if "generateContent" in acts and "gemini" in m.name:
            tag = "  (preview/exp)" if any(t in m.name for t in ("preview", "exp")) else ""
            print(f"{m.name.removeprefix('models/')}{tag}  — {getattr(m, 'display_name', '')} / version {getattr(m, 'version', '')}")


def report(df: pd.DataFrame, fakes: list[dict], model: str, args) -> str:
    calls = df[~df.cached] if (~df.cached).any() else df
    lines = [json.loads(l) for l in open(S.RAW_DIR / f"calls_{args.tag}.jsonl", encoding="utf-8")]
    real_overlap = []
    for rec in fakes:
        last = [x for x in lines if x["sample_id"] == rec["sample_id"]][-1]
        said = {i["matched"] for i in (last["items"] or []) if i["field"] == "main_reasons" and i["matched"]}
        real_top = {f["variable"] for f in rec["features"] if f["real_shap_rank"] <= S.N_REASONS}
        real_overlap.append(len(said & real_top) / S.N_REASONS)
    tin, tout, tth = (calls[c].fillna(0) for c in ("tokens_in", "tokens_out", "tokens_thought"))
    cost = None
    if args.price_in is not None and args.price_out is not None:
        cost = (tin * args.price_in + (tout + tth) * args.price_out) / 1e6
    ok = int((df.status == "ok").sum())
    n_items, n_map = df.get("n_items", pd.Series(dtype=float)).sum(), df.get("n_mapped", pd.Series(dtype=float)).sum()
    unm = [u for s in df.get("unmapped_said", pd.Series(dtype=str)).dropna() for u in str(s).split(" | ") if u]
    alias = "있음 (docs/variable_aliases.csv)" if os.path.exists(S.ALIASES_CSV) else "없음 → 한국어 이름만으로 매핑"
    md = [f"# L3-5 시험 10건 (가짜 SHAP, 조건 B) — {datetime.now():%Y-%m-%d %H:%M}", "",
          f"- 모델: `{model}` (응답 model_version: {', '.join(sorted(set(map(str, df.model_version.dropna())))) or '-'})",
          f"- 설정: temperature {S.TEMPERATURE}, top_p {S.TOP_P or '기본값'}, max_output_tokens {S.MAX_OUTPUT_TOKENS}, seed {S.SEED}, 프롬프트 {S.PROMPT_VERSION}",
          f"- 호출 최소 간격: {S.MIN_INTERVAL_SEC}초. 재시도: API 최대 {S.MAX_API_RETRIES}회(지수 백오프), 파싱·스키마 최대 {S.MAX_PARSE_RETRIES}회",
          f"- 입력 sha256: `{df.input_sha256.iloc[0]}`",
          f"- 표본: 300명 중 시드 {TRIAL_SEED}로 10명, SHAP은 레코드 안에서 섞음 (시드 = sample_id). 별칭 파일: {alias}", "",
          "## 결과", "", "| 항목 | 값 |", "| --- | --- |",
          f"| 파싱 성공 (재시도 후) | {int(df.parse_ok.sum())}/{len(df)} |",
          f"| 파싱·스키마 성공 = status ok (재시도 후) | {ok}/{len(df)} |",
          f"| **통과 기준 (10/10)** | **{'통과' if ok == len(df) == N_TRIAL else '미달'}** |",
          f"| API 재시도 | {int((df.api_attempts > 1).sum())}건 (총 시도 {int(df.api_attempts.sum())}회) |",
          f"| 파싱·스키마 재호출 | {int((df.parse_attempts > 1).sum())}건 |",
          f"| 호출당 지연 (새 호출) | 평균 {calls.latency_s.mean():.1f}초, 최대 {calls.latency_s.max():.1f}초 |",
          f"| 호출당 토큰 | 입력 {tin.mean():,.0f} / 출력 {tout.mean():,.0f} / 생각 {tth.mean():,.0f} |",
          f"| 호출당 비용 | {'무료 티어: 청구 0 (유료 단가 미입력)' if cost is None else f'유료 단가 기준 ${cost.mean():.5f}, 600건 환산 ${cost.mean() * 600:.2f}'} |",
          f"| 변수 매핑률 (main_reasons + other_factors) | {n_map:.0f}/{n_items:.0f} = {n_map / max(n_items, 1):.1%} (목표 90%) |",
          f"| 가짜 SHAP 상위 4 ∩ 응답 4개 (F) | 평균 {df.get('F', pd.Series(dtype=float)).mean():.2f} |",
          f"| 방향 포함 (F_dir, 가짜 SHAP 기준) | 평균 {df.get('F_dir', pd.Series(dtype=float)).mean():.2f} |",
          f"| (대조) 진짜 SHAP 상위 4 ∩ 응답 4개 | 평균 {pd.Series(real_overlap).mean():.2f} |", ""]
    if "A_core_flag_struct" in df:
        md += [f"- A-핵심 키워드 1차 플래그 (구조화 필드 / 자유 서술): {int(df.A_core_flag_struct.sum())} / {int(df.A_core_flag_free.sum())}건 (판정 아님, L4에서 검토)", ""]
    bad = df[df.status != "ok"]
    if len(bad):
        md += ["## 실패 건", ""] + [f"- {r.sample_id}: {r.fail_reason}" for r in bad.itertuples()] + [""]
    if unm:
        md += ["## 매핑 실패 표기 (별칭 파일 조정 후보, 분석 계획 6절)", ""] + [f"- {u}" for u in sorted(set(unm))] + [""]
    md += [f"행 단위 원문: results/l3/raw/calls_{args.tag}.jsonl (git 제외)."]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="trial")
    ap.add_argument("--price-in", type=float)
    ap.add_argument("--price-out", type=float)
    a = ap.parse_args()
    if a.list_models:
        return list_models()
    assert a.model, "--model 을 준다 (먼저 --list-models 로 확인)"
    fakes = [fake_shap(r, seed=r["sample_id"]) for r in pick(llm.load_records())]
    if a.dry_run:
        d = S.RAW_DIR / "prompt_examples"
        d.mkdir(parents=True, exist_ok=True)
        (d / "system.txt").write_text(llm.SYSTEM, encoding="utf-8")
        for c in ("A", "B"):
            (d / f"{fakes[0]['sample_id']}_{c}.txt").write_text(llm.build_prompt(fakes[0], c), encoding="utf-8")
        print(f"프롬프트 예시 저장: {d}")
        return
    in_hash = llm.sha256_file(S.INPUT_JSONL)
    (S.L3_DIR / "call_settings.json").write_text(json.dumps({
        **llm.gen_settings(a.model), "min_interval_sec": S.MIN_INTERVAL_SEC, "max_api_retries": S.MAX_API_RETRIES,
        "max_parse_retries": S.MAX_PARSE_RETRIES, "input_sha256": in_hash, "spec_sha256": llm.sha256_file(S.SPEC_MD),
        "system_prompt_sha256": llm.sha256_text(llm.SYSTEM)}, ensure_ascii=False, indent=1), encoding="utf-8")
    client = llm.Gemini(a.model)
    df = llm.run_batch(fakes, "B", client=client, tag=a.tag, input_hash=in_hash)
    md = report(df, fakes, a.model, a)
    (S.L3_DIR / "trial_L3-5.md").write_text(md, encoding="utf-8")
    print(md)
    llm.log(f"L3-5 시험 10건 (가짜 SHAP, B): 모델 {a.model}, 간격 {S.MIN_INTERVAL_SEC}s, status ok {int((df.status == 'ok').sum())}/10, "
            f"매핑률 {df.get('n_mapped', pd.Series([0])).sum() / max(df.get('n_items', pd.Series([1])).sum(), 1):.1%} → results/l3/trial_L3-5.md")


if __name__ == "__main__":
    main()
