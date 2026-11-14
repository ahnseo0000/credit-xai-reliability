"""L3-6 본 호출: 300명 x 조건 A/B = 600건, 신청자 x 조건당 1회 (L3_SPEC 2절·5절).

- 진짜 SHAP을 쓴다 (가짜 SHAP은 trial.py에서만).
- 신청자 순서는 무작위 (settings.ORDER_SEED). 한 신청자의 A와 B를 연달아 부른다.
- 동시 호출 --workers개 (최대 settings.MAX_WORKERS). 호출 시작 간격은 MIN_INTERVAL_SEC 이상 (llm.call_llm 공유 잠금).
- 실패: 429·503·타임아웃·네트워크는 지수 백오프로 최대 3회 재시도 후 failed로 기록(fail_reason, error에 사유).
  파싱·스키마 위반은 같은 프롬프트로 최대 2회 재호출 후 failed (SPEC 3·5절).
- 이어 하기: results/l3/raw/calls_main.jsonl을 읽어 (sample_id, 조건)의 마지막 기록이
  ok 이거나 파싱·스키마 failed면 건너뛰고, API 오류 failed면 다시 부른다 (재호출 기록이 뒤에 덧붙는다. 표는 마지막 기록 기준).
- API 오류 failed가 연속 STOP_AFTER_API_FAILS건이면 새 호출을 멈춘다 (하루 한도 소진 등). 다시 실행하면 이어서 부른다.
- 끝나면 results/l3/raw/l3_main.csv (한 행 = 신청자 x 조건, 마지막 기록)를 다시 만든다.

실행: python -m src.l3.run_main --model gemini-3.5-flash --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import llm
from . import settings as S

TAG = "main"
API_ERR = ("429", "5", "타임아웃", "네트워크")   # fail_reason 앞부분으로 API 오류 판별 (5xx 서버 오류 포함)
LOG = S.RAW_DIR / f"calls_{TAG}.jsonl"


def is_api_fail(row: dict) -> bool:
    return row["status"] == "failed" and not row["parse_ok"] and str(row["fail_reason"]).startswith(API_ERR)


def last_records(path=None) -> dict:
    path = path or LOG
    last = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for d in map(json.loads, f):
                last[(d["sample_id"], d["condition"])] = d
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=S.MAX_WORKERS)
    ap.add_argument("--limit", type=int, help="(점검용) 무작위 순서 앞에서 n명만")
    a = ap.parse_args()
    workers = max(1, min(a.workers, S.MAX_WORKERS))

    recs = llm.load_records()
    random.Random(S.ORDER_SEED).shuffle(recs := sorted(recs, key=lambda r: r["sample_id"]))
    recs = recs[: a.limit]
    in_hash = llm.sha256_file(S.INPUT_JSONL)
    last = last_records()

    def need(rec, c):
        d = last.get((rec["sample_id"], c))
        return d is None or is_api_fail(d)

    jobs = [(r, [c for c in ("A", "B") if need(r, c)]) for r in recs]
    jobs = [(r, cs) for r, cs in jobs if cs]
    n_todo = sum(len(cs) for _, cs in jobs)
    print(f"전체 {len(recs) * 2}건 중 남은 {n_todo}건 (신청자 {len(jobs)}명), 동시 {workers}개, 시작 간격 {S.MIN_INTERVAL_SEC}초 이상")
    if not jobs:
        return rebuild()

    S.L3_DIR.mkdir(parents=True, exist_ok=True)
    (S.L3_DIR / "call_settings_main.json").write_text(json.dumps({
        **llm.gen_settings(a.model), "min_interval_sec": S.MIN_INTERVAL_SEC, "max_api_retries": S.MAX_API_RETRIES,
        "max_parse_retries": S.MAX_PARSE_RETRIES, "http_timeout_ms": S.HTTP_TIMEOUT_MS, "sdk_retry": "off",
        "workers": workers, "order_seed": S.ORDER_SEED, "order": "신청자 무작위, 신청자마다 A 다음 B",
        "input_sha256": in_hash, "spec_sha256": llm.sha256_file(S.SPEC_MD),
        "system_prompt_sha256": llm.sha256_text(llm.SYSTEM)}, ensure_ascii=False, indent=1), encoding="utf-8")
    llm.log(f"L3-6 본 호출 시작/재개: 모델 {a.model}, 남은 {n_todo}건, 동시 {workers}, 입력 sha256 {in_hash[:12]}")

    client = llm.Gemini(a.model)
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    st = {"streak": 0, "done": 0}
    logf = open(LOG, "a", encoding="utf-8")

    def work(job):
        rec, conds = job
        for c in conds:
            if stop.is_set():
                return
            base, line = llm.run_one(rec, c, client=client, tag=TAG, input_hash=in_hash)
            with lock:
                logf.write(json.dumps(line, ensure_ascii=False) + "\n")
                logf.flush()
                st["done"] += 1
                st["streak"] = st["streak"] + 1 if is_api_fail(base) else 0
                extra = f" [{base['fail_reason']}]" if base["status"] != "ok" else ""
                print(f"[{st['done']}/{n_todo}] {rec['sample_id']} {c} {base['status']} ({base['latency_s']}s, v={base['model_version']}){extra}", flush=True)
                if st["streak"] >= S.STOP_AFTER_API_FAILS and not stop.is_set():
                    stop.set()
                    llm.log(f"L3-6 중단: API 오류 failed 연속 {st['streak']}건 ({base['fail_reason']}). 진행 중인 호출만 마치고 멈춘다. 다시 실행하면 그 건들부터 이어서 부른다")

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, jobs))
    finally:
        logf.close()
    rebuild()


def rebuild():
    last = last_records()
    if not last:
        return
    df = pd.DataFrame([{k: v for k, v in d.items() if k not in ("response_text", "error", "items")} for d in last.values()])
    df = df.sort_values(["sample_id", "condition"])
    df.to_csv(S.RAW_DIR / f"l3_{TAG}.csv", index=False, encoding="utf-8-sig")
    tab = df.groupby("condition").status.value_counts().unstack(fill_value=0).to_dict("index")
    reasons = df[df.status != "ok"].fail_reason.value_counts().to_dict()
    pairs = int(df[df.status == "ok"].groupby("sample_id").condition.nunique().eq(2).sum())
    vers = df.model_version.value_counts(dropna=False).to_dict()
    llm.log(f"L3-6 현황: 기록 {len(df)}/600건, 조건별 {tab}, 실패 사유 {reasons}, 두 조건 모두 ok 쌍 {pairs}명, "
            f"model_version {vers} → results/l3/raw/l3_{TAG}.csv")


if __name__ == "__main__":
    main()
