"""L3-6 본 호출: 300명 x 조건 A/B = 600건, 신청자 x 조건당 1회 (L3_SPEC 2절·5절).

- 진짜 SHAP을 쓴다 (가짜 SHAP은 trial.py에서만).
- 신청자마다 A -> B 순서로 부른다 (중간에 멈춰도 완성된 쌍이 최대한 남도록).
- 이어 하기: results/l3/raw/calls_main.jsonl에 이미 기록된 (sample_id, 조건)은 건너뛴다.
  단, API 오류(429·5xx·타임아웃·네트워크)로 failed가 STOP_AFTER_API_FAILS번 연속이면 그 건들은 기록하지 않고 멈춘다
  (하루 한도 소진 등). 다시 실행하면 그 지점부터 이어서 부른다. 파싱·스키마 실패는 SPEC대로 failed로 남는다.
- 끝나면 results/l3/raw/l3_main.csv(한 행 = 한 호출)를 기록 파일에서 다시 만든다.

실행: python -m src.l3.run_main --model gemini-3.5-flash
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import llm
from . import settings as S

TAG = "main"
API_ERR = ("429", "5", "타임아웃", "네트워크")   # fail_reason 앞부분으로 API 오류 판별 (5xx 서버 오류 포함)


def _is_api_fail(row: dict) -> bool:
    return row["status"] == "failed" and not row["parse_ok"] and str(row["fail_reason"]).startswith(API_ERR)


def done_keys(path) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(d["sample_id"], d["condition"]) for d in map(json.loads, f) if d.get("status")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, help="(점검용) 앞에서 n명만")
    a = ap.parse_args()

    recs = sorted(llm.load_records(), key=lambda r: r["sample_id"])[: a.limit]
    in_hash = llm.sha256_file(S.INPUT_JSONL)
    log_path = S.RAW_DIR / f"calls_{TAG}.jsonl"
    done = done_keys(log_path)
    todo = [(r, c) for r in recs for c in ("A", "B") if (r["sample_id"], c) not in done]
    print(f"전체 {len(recs) * 2}건 중 완료 {len(recs) * 2 - len(todo)}건, 남은 {len(todo)}건")
    if not todo:
        return rebuild()

    S.L3_DIR.mkdir(parents=True, exist_ok=True)
    (S.L3_DIR / "call_settings_main.json").write_text(json.dumps({
        **llm.gen_settings(a.model), "min_interval_sec": S.MIN_INTERVAL_SEC, "max_api_retries": S.MAX_API_RETRIES,
        "max_parse_retries": S.MAX_PARSE_RETRIES, "http_timeout_ms": S.HTTP_TIMEOUT_MS, "sdk_retry": "off",
        "input_sha256": in_hash, "spec_sha256": llm.sha256_file(S.SPEC_MD),
        "system_prompt_sha256": llm.sha256_text(llm.SYSTEM)}, ensure_ascii=False, indent=1), encoding="utf-8")
    llm.log(f"L3-6 본 호출 시작/재개: 모델 {a.model}, 남은 {len(todo)}건, 입력 sha256 {in_hash[:12]}")

    client = llm.Gemini(a.model)
    tmp = S.RAW_DIR / f"calls_{TAG}_tmp.jsonl"
    streak, pending, n_new = 0, [], 0
    for i, (rec, cond) in enumerate(todo, 1):
        if tmp.exists():
            tmp.unlink()
        df = llm.run_batch([rec], cond, client=client, tag=f"{TAG}_tmp", input_hash=in_hash)
        line = tmp.read_text(encoding="utf-8")
        row = df.iloc[0].to_dict()
        if _is_api_fail(row):
            streak += 1
            pending.append(line)
            if streak >= S.STOP_AFTER_API_FAILS:
                llm.log(f"L3-6 중단: API 오류 연속 {streak}건 ({row['fail_reason']}). 해당 {streak}건은 기록하지 않음. 다시 실행하면 이어서 부른다")
                break
            continue
        with open(log_path, "a", encoding="utf-8") as f:   # 연속 오류가 끊기면 보류분(진짜 failed)도 기록
            f.writelines(pending + [line])
        pending, streak = [], 0
        n_new += 1
        print(f"[{i}/{len(todo)}] {rec['sample_id']} {cond} {row['status']} ({row['latency_s']}s)", flush=True)
    else:
        if pending:   # 마지막이 연속 오류로 끝났지만 기준 미만이면 failed로 기록
            with open(log_path, "a", encoding="utf-8") as f:
                f.writelines(pending)
    for p in (tmp, S.RAW_DIR / f"l3_{TAG}_tmp.csv"):
        if p.exists():
            p.unlink()
    rebuild()


def rebuild():
    log_path = S.RAW_DIR / f"calls_{TAG}.jsonl"
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    df = pd.DataFrame([{k: v for k, v in d.items() if k not in ("response_text", "error", "items")} for d in rows])
    df = df.drop_duplicates(["sample_id", "condition"], keep="first")
    df.to_csv(S.RAW_DIR / f"l3_{TAG}.csv", index=False, encoding="utf-8-sig")
    tab = df.groupby("condition").status.value_counts().unstack(fill_value=0)
    pairs = df[df.status == "ok"].groupby("sample_id").condition.nunique().eq(2).sum()
    msg = f"L3-6 현황: 기록 {len(df)}/600건, 조건별 상태 {tab.to_dict('index')}, 두 조건 모두 ok인 쌍 {pairs}명 → results/l3/raw/l3_{TAG}.csv"
    llm.log(msg)


if __name__ == "__main__":
    main()
