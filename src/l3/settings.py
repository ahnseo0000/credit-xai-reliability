"""L3 호출·실험 설정 (docs/L3_SPEC.md 5절). 결과를 본 뒤 바꾸면 results/l3/RUNLOG.md에 이유를 적는다.

모델 이름은 코드에 박지 않는다. 실행할 때 --model 로 주고, 쓴 값은 호출 기록과 results/l3/call_settings.json에 남는다.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L3_DIR = ROOT / "results" / "l3"
CACHE_DIR = L3_DIR / "cache"          # .gitignore: results/l3/cache/
RAW_DIR = L3_DIR / "raw"              # .gitignore: results/*/raw/  (호출 원문, 행 단위 표)
RUNLOG = L3_DIR / "RUNLOG.md"

# zip을 푼 폴더 (repo 밖). 기본값: repo 옆 폴더 l3_handoff_package/ (없으면 l3_handoff/)
_default_in = ROOT.parent / "l3_handoff_package"
if not _default_in.exists():
    _default_in = ROOT.parent / "l3_handoff"
INPUT_DIR = Path(os.environ.get("L3_INPUT_DIR", _default_in))
INPUT_JSONL = INPUT_DIR / "llm_input_300_full.jsonl"
SPEC_MD = INPUT_DIR / "L3_SPEC.md"

# 선택: repo에 올라오면 쓰는 파일 (없으면 해당 지표만 건너뛴다). 최종 판정은 L4(src/l4_analysis.py)가 한다.
ALIASES_CSV = ROOT / "docs" / "variable_aliases.csv"
KEYWORDS_CSV = ROOT / "docs" / "protected_keywords.csv"

# ---- 생성 설정 (전부 호출 기록에 남는다)
TEMPERATURE = 0.0
TOP_P = None                 # None = 모델 기본값
MAX_OUTPUT_TOKENS = 8192
SEED = 42
N_REASONS = 4

# ---- 호출 정책 (L3_SPEC 5절)
MIN_INTERVAL_SEC = float(os.environ.get("GEMINI_MIN_INTERVAL", "6.5"))  # 호출 시작 간 최소 간격
MAX_API_RETRIES = 3          # 429·5xx·네트워크 오류: 지수 백오프 최대 3회 재시도 후 failed
MAX_PARSE_RETRIES = 2        # 파싱·스키마 실패: 같은 프롬프트로 최대 2회 다시 부름
BACKOFF_BASE_SEC = 5.0
HTTP_TIMEOUT_MS = 120_000   # 호출 1회 HTTP 타임아웃 (넘으면 "타임아웃"으로 재시도 규칙 적용)
STOP_AFTER_API_FAILS = 6   # 본 호출: API 오류(429·5xx·타임아웃)로 failed가 연속 이만큼이면 새 호출을 멈춘다 (한도 소진 대비)
MAX_WORKERS = 4            # 본 호출 동시 호출 수 상한 (시작 간격은 MIN_INTERVAL_SEC 이상 유지)
ORDER_SEED = 42            # 본 호출 신청자 순서 무작위화 시드

PROMPT_VERSION = "p2"   # p2: 방향 표기를 입력과 같은 "부도 위험 증가/감소"로 (L3_SPEC 3절)
