"""프로젝트 공통 경로와 상수."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"  # git에 올리지 않는 중간 산출물
MODELS_DIR = RESULTS_DIR / "models"

RAW_DIR = DATA_DIR / "raw"
RAW_TRAIN = RAW_DIR / "application_train.csv"
VAR_DICT = DOCS_DIR / "variable_dict.csv"

SEED = 42
N_JOBS = 4  # XGBoost 스레드 수 고정 (재현성). 스레드 수가 바뀌면 부동소수점 합산 순서가 달라질 수 있다.
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# DAYS_EMPLOYED의 무직·연금 수령 표시값
DAYS_EMPLOYED_ANOMALY = 365243

def set_korean_font():
    """설치된 한글 폰트로 matplotlib 설정 (macOS AppleGothic, Windows 맑은 고딕, Linux 나눔고딕)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# 7:1.5:1.5 분할
VALID_SIZE = 0.15
TEST_SIZE = 0.15
