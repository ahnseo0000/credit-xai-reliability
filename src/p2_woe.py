"""Phase 2 공통: train/valid 변수 읽기, WoE 구간화(분위수 + 단조 병합), IV, WoE 변환.

규칙: test는 읽지 않는다. 데이터는 L2 캐시(results/l2/raw)의 X_train/X_valid, y_train/y_valid만 읽는다 (읽기 전용, 수정 안 함).
캐시의 X는 v2 모형 입력(신청서 변수 전처리 후 + L0 변수)이다. 범주형은 원-핫 열에서 범주 이름을 복원한다. DAYS_EMPLOYED의 365243은 이미 결측이다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .models import feature_to_variable

L2RAW = C.RESULTS_DIR / "l2" / "raw"
FROZEN = C.RESULTS_DIR / "v2_frozen"
P2 = C.RESULTS_DIR / "phase2"
MISSING = "__MISSING__"
EPS = 0.5          # WoE 평활화 (구간별 good/bad 수에 더함)


class Data:
    """train 또는 valid 한 분할. test는 만들지 않는다."""

    def __init__(self, split: str):
        assert split in ("train", "valid"), "test는 읽지 않는다"
        self.split = split
        self.X = np.load(L2RAW / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(L2RAW / f"y_{split}.npy")
        self.ids = np.load(L2RAW / f"ids_{split}.npy")
        feat = json.loads((FROZEN / "features.json").read_text(encoding="utf-8"))
        self.variables = feat["application_features"] + feat["l0_features"]
        self.col_names = json.loads((L2RAW / "col_names.json").read_text(encoding="utf-8"))
        f2v = feature_to_variable(self.col_names, self.variables)
        self.cols = {v: [i for i, c in enumerate(self.col_names) if f2v[c] == v] for v in self.variables}
        self.is_cat = {v: not (len(c) == 1 and self.col_names[c[0]] == v) for v, c in self.cols.items()}

    def var(self, v: str):
        """수치형은 float 배열(결측 NaN), 범주형은 문자열 배열(결측은 MISSING)."""
        idx = self.cols[v]
        if not self.is_cat[v]:
            return np.asarray(self.X[:, idx[0]], dtype=np.float64)
        block = np.asarray(self.X[:, idx])
        names = np.array([self.col_names[i][len(v) + 1:] for i in idx], dtype=object)
        lab = names[block.argmax(axis=1)]
        lab[lab == "Missing"] = MISSING
        return lab


# ---------------------------------------------------------------- WoE/IV
def woe_table(n_bad: np.ndarray, n_good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """구간별 WoE = ln(good 비중 / bad 비중) (평활화)와 IV 기여."""
    B, G = n_bad.sum(), n_good.sum()
    pb = (n_bad + EPS) / (B + EPS * len(n_bad))
    pg = (n_good + EPS) / (G + EPS * len(n_good))
    woe = np.log(pg / pb)
    return woe, (pg - pb) * woe


def fine_cuts(x: np.ndarray, n_q: int = 10, min_share: float = 0.02) -> np.ndarray:
    """수치형: 분위수 절단점(중복 제거). 비중이 min_share 미만인 구간은 이웃 구간과 합쳐 너무 작은 구간이 IV를 부풀리지 않게 한다."""
    nm = x[~np.isnan(x)]
    if len(nm) == 0:
        return np.array([])
    cuts = list(np.unique(np.quantile(nm, np.linspace(0, 1, n_q + 1)[1:-1])))
    while cuts:
        share = np.bincount(np.digitize(nm, cuts), minlength=len(cuts) + 1) / len(nm)
        i = int(np.argmin(share))
        if share[i] >= min_share:
            break
        if i == 0:
            j = 0
        elif i == len(cuts):
            j = len(cuts) - 1
        else:
            j = i - 1 if share[i - 1] <= share[i + 1] else i
        cuts.pop(j)
    return np.array(cuts)


def assign_numeric(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """구간 번호 0..len(cuts), 결측은 -1."""
    b = np.digitize(np.where(np.isnan(x), 0.0, x), cuts) if len(cuts) else np.zeros(len(x), int)
    b = b.astype(int)
    b[np.isnan(x)] = -1
    return b


def iv_of(bins: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    d = pd.DataFrame({"b": bins, "y": y}).groupby("b")["y"].agg(["size", "sum"])
    n_bad, n_good = d["sum"].to_numpy(float), (d["size"] - d["sum"]).to_numpy(float)
    woe, iv = woe_table(n_bad, n_good)
    t = pd.DataFrame({"bin": d.index, "n": d["size"].to_numpy(), "bad": n_bad, "woe": woe, "iv": iv})
    return float(iv.sum()), t


def categorical_bins(lab: np.ndarray, y: np.ndarray, min_share: float = 0.02, max_bins: int = 8):
    """범주형: 부도율 순으로 정렬한 범주를 (희소 범주 병합 후) 인접 병합해 max_bins 이하로. 반환: 범주 -> 구간 번호 딕셔너리, 결측은 별도."""
    d = pd.DataFrame({"c": lab, "y": y})
    nm = d[d.c != MISSING]
    g = nm.groupby("c")["y"].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    small = g["size"] / g["size"].sum() < min_share
    groups = {c: [c] for c in g.index[~small]}
    if small.any():   # 희소 범주는 하나로 묶는다
        groups["__RARE__"] = list(g.index[small])
    stats = {k: (g.loc[v, "size"].sum(), g.loc[v, "sum"].sum()) for k, v in groups.items()}
    order = sorted(stats, key=lambda k: stats[k][1] / stats[k][0])
    blocks = [[k] for k in order]
    while len(blocks) > max_bins:
        rates = [sum(stats[k][1] for k in b) / sum(stats[k][0] for k in b) for b in blocks]
        j = int(np.argmin(np.diff(rates)))
        blocks[j:j + 2] = [blocks[j] + blocks[j + 1]]
    mapping = {}
    for bi, b in enumerate(blocks):
        for k in b:
            for c in groups[k]:
                mapping[c] = bi
    mapping["__RARE__"] = next((bi for bi, b in enumerate(blocks) if "__RARE__" in b), None)
    return mapping


def assign_categorical(lab: np.ndarray, mapping: dict) -> np.ndarray:
    rare = mapping.get("__RARE__")
    out = np.array([-1 if c == MISSING else mapping.get(c, rare if rare is not None else -2) for c in lab], dtype=int)
    return out


def monotone_cuts(x: np.ndarray, y: np.ndarray, max_bins: int = 6, min_share: float = 0.05, n_q: int = 20):
    """수치형 단조 구간화: 분위수 20구간 -> 아이소토닉(PAVA)으로 단조 병합 -> 최소 비중·최대 구간 수 맞춤.
    방향(증가/감소)은 구간별 부도율과 값 순서의 상관 부호로 정한다. 반환: (절단점, 방향 +1/-1)"""
    nm = ~np.isnan(x)
    cuts = fine_cuts(x, n_q=n_q)
    if len(cuts) == 0:
        return np.array([]), 1
    b = np.digitize(x[nm], cuts)
    d = pd.DataFrame({"b": b, "y": y[nm]}).groupby("b")["y"].agg(["size", "sum"])
    rate = (d["sum"] / d["size"]).to_numpy()
    direction = 1 if np.corrcoef(np.arange(len(rate)), rate)[0, 1] >= 0 else -1
    iso = IsotonicRegression(increasing=(direction == 1)).fit(np.arange(len(rate)), rate, sample_weight=d["size"].to_numpy())
    fitted = iso.predict(np.arange(len(rate)))
    blocks = [[0]]
    for i in range(1, len(rate)):
        if np.isclose(fitted[i], fitted[i - 1]):
            blocks[-1].append(i)
        else:
            blocks.append([i])
    n = d["size"].to_numpy(float)
    bad = d["sum"].to_numpy(float)

    def stat(bl):
        return n[bl].sum(), bad[bl].sum()

    total = n.sum()
    while True:   # 최소 비중 미만 구간은 부도율이 가장 가까운 이웃에 병합, 그다음 최대 구간 수까지 병합
        shares = [stat(b)[0] / total for b in blocks]
        rates = [stat(b)[1] / stat(b)[0] for b in blocks]
        if len(blocks) == 1:
            break
        small = [i for i, s in enumerate(shares) if s < min_share]
        if small:
            i = small[0]
            cand = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
            j = min(cand, key=lambda j: abs(rates[j] - rates[i]))
        elif len(blocks) > max_bins:
            diffs = np.abs(np.diff(rates))
            i = int(np.argmin(diffs)); j = i + 1
        else:
            break
        lo, hi = min(i, j), max(i, j)
        blocks[lo:hi + 1] = [blocks[lo] + blocks[hi]]
    new_cuts = np.array([cuts[bl[0] - 1] for bl in blocks[1:]])   # 각 블록 첫 구간의 아래 경계
    return new_cuts, direction
