"""Phase 2-4a: 스코어카드 22개 변수의 family 간 상관과 VIF 점검 (train 기준, test 미사용). 제안만 하고 모형은 바꾸지 않는다.

스코어카드가 실제로 보는 값(WoE 변환값)에서 상관·VIF를 계산한다. 구간화는 p2_scorecard와 같은 함수·같은 파라미터로 train에서 다시 만든다 (결정적이라 같은 결과).
실행: python -m src.p2_vif
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .l2_common import family_map
from .p2_scorecard import MAX_BINS_CAT, MAX_BINS_NUM, MIN_SHARE_CAT, MIN_SHARE_NUM
from .p2_woe import P2, Data, assign_categorical, assign_numeric, categorical_bins, iv_of, monotone_cuts

CORR_FLAG = 0.7
VIF_WARN, VIF_BAD = 5.0, 10.0


def log(msg: str, decision: bool = False):
    line = f"- `{datetime.now():%H:%M:%S}` " + (f"**[결정]** {msg}" if decision else msg)
    with open(P2 / "RUNLOG.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        pass


def woe_train(tr: Data, variables: list[str]) -> pd.DataFrame:
    W = {}
    for v in variables:
        x = tr.var(v)
        if tr.is_cat[v]:
            b = assign_categorical(x, categorical_bins(x, tr.y, min_share=MIN_SHARE_CAT, max_bins=MAX_BINS_CAT))
        else:
            cuts, _ = monotone_cuts(x, tr.y, max_bins=MAX_BINS_NUM, min_share=MIN_SHARE_NUM)
            b = assign_numeric(x, cuts)
        _, t = iv_of(b, tr.y)
        woe = dict(zip(t["bin"], t["woe"]))
        W[v] = np.array([woe.get(k, 0.0) for k in b])
    return pd.DataFrame(W)


def main():
    sc = json.loads((P2 / "scorecard_model.json").read_text(encoding="utf-8"))
    vs, coef = sc["variables"], pd.Series(sc["coef"])
    tr = Data("train")
    fam = family_map(tr.variables)
    W = woe_train(tr, vs)
    R = W.corr()
    iv = pd.read_csv(P2 / "iv_all.csv", encoding="utf-8-sig").set_index("variable")["iv_train"]
    log(f"4a 시작: 스코어카드 {len(vs)}개 변수, train {len(W):,}건의 WoE 변환값에서 상관·VIF 계산 (test 미사용)")
    pairs = []
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            r = R.loc[a, b]
            if abs(r) > CORR_FLAG:
                pairs.append({"a": a, "b": b, "fam_a": fam[a], "fam_b": fam[b], "same_family": fam[a] == fam[b], "corr": r})
    pairs = pd.DataFrame(pairs)
    vif = pd.Series(np.diag(np.linalg.inv(R.to_numpy())), index=vs, name="VIF")
    # 해당 변수가 가장 크게 상관된 다른 변수
    mx = {v: R.loc[v].drop(v).abs().sort_values(ascending=False).head(1) for v in vs}
    tab = pd.DataFrame({"family": [fam[v] for v in vs], "IV(train)": iv.reindex(vs).to_numpy(), "스코어카드 계수": coef.reindex(vs).to_numpy(), "VIF": vif.to_numpy(),
                        "최대 |상관| 상대": [mx[v].index[0] for v in vs], "최대 |상관|": [mx[v].iloc[0] for v in vs]}, index=vs).sort_values("VIF", ascending=False)

    # ---- 제안 (적용하지 않음): VIF 큰 변수부터, 그 변수와 가장 강하게 얽힌 쌍에서 IV 낮은 쪽을 제거 후보로
    sug, cur = [], list(vs)
    while True:
        Rc = W[cur].corr().to_numpy()
        vf = pd.Series(np.diag(np.linalg.inv(Rc)), index=cur)
        if vf.max() < VIF_WARN:
            break
        worst = vf.idxmax()
        partner = W[cur].corr()[worst].drop(worst).abs().idxmax()
        drop = worst if iv[worst] <= iv[partner] else partner
        sug.append({"drop": drop, "worst_vif_var": worst, "vif": float(vf.max()), "partner": partner, "corr": float(W[cur].corr().loc[worst, partner]),
                    "iv_drop": float(iv[drop]), "iv_keep": float(iv[partner if drop == worst else worst])})
        cur.remove(drop)
    md = ["# Phase 2 점검: 스코어카드 22개 변수의 상관과 VIF (train)", "",
          f"- 계산 대상: 스코어카드가 실제로 쓰는 WoE 변환값 {len(vs)}개 열, train {len(W):,}건. **test는 읽지 않았다.** 구간화는 p2_scorecard와 같은 규칙으로 train에서 다시 만들었다.",
          f"- 기준: 쌍 |상관| > {CORR_FLAG}, VIF ≥ {VIF_WARN:g} 주의, ≥ {VIF_BAD:g} 문제 (VIF = 상관행렬 역행렬의 대각, 표준화 WoE 기준). **이 문서는 점검과 제안만 한다. 모형은 바꾸지 않았다.**",
          f"- 이전 선택 단계에서는 같은 family 안의 상관만 정리했다(|상관|>0.7). 이번에는 family 간 상관을 처음 본다.", "",
          f"## 1. |상관| > {CORR_FLAG} 쌍", ""]
    if pairs.empty:
        md.append("**없음.** 22개 변수 중 |상관| > 0.7인 쌍은 없다.")
    else:
        md += ["| 변수 A | 변수 B | family A | family B | 같은 family | 상관 |", "| --- | --- | --- | --- | --- | --- |"]
        for r in pairs.sort_values("corr", key=lambda s: -s.abs()).itertuples():
            md.append(f"| {r.a} | {r.b} | {r.fam_a} | {r.fam_b} | {'예' if r.same_family else '아니오 (family 간)'} | {r.corr:+.3f} |")
    top = []
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            top.append((abs(R.loc[a, b]), a, b, R.loc[a, b]))
    top.sort(reverse=True)
    md += ["", "참고: 기준 아래라도 상관이 큰 상위 10쌍.", "", "| 변수 A | 변수 B | family A | family B | 상관 |", "| --- | --- | --- | --- | --- |"]
    for _, a, b, r in top[:10]:
        md.append(f"| {a} | {b} | {fam[a]} | {fam[b]} | {r:+.3f} |")
    md += ["", "## 2. 변수별 VIF (큰 순)", "", "| 변수 | family | IV | 계수 | VIF | 가장 상관 큰 상대 | |상관| |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for v, r in tab.iterrows():
        flag = " **(문제)**" if r["VIF"] >= VIF_BAD else (" (주의)" if r["VIF"] >= VIF_WARN else "")
        md.append(f"| {v} | {r['family']} | {r['IV(train)']:.3f} | {r['스코어카드 계수']:+.3f} | {r['VIF']:.2f}{flag} | {r['최대 |상관| 상대']} | {r['최대 |상관|']:.3f} |")
    md += ["", f"VIF 최대 {vif.max():.2f}, 중앙값 {vif.median():.2f}, 주의(≥{VIF_WARN:g}) {int((vif >= VIF_WARN).sum())}개, 문제(≥{VIF_BAD:g}) {int((vif >= VIF_BAD).sum())}개.", "", "## 3. 제안 (적용하지 않음)", ""]
    if not sug:
        md.append(f"VIF가 {VIF_WARN:g}를 넘는 변수가 없고 쌍 상관 {'도 기준 이하' if pairs.empty else '기준 초과 쌍은 위 표뿐'}이라 **제거를 제안할 변수가 없다.**" if pairs.empty else
                  f"VIF가 {VIF_WARN:g}를 넘는 변수는 없다. 위 쌍 상관은 해석상 중복 신호이므로 필요하면 IV가 낮은 쪽을 후보로 삼을 수 있다.")
        if not pairs.empty:
            for r in pairs.itertuples():
                lo = r.a if iv[r.a] <= iv[r.b] else r.b
                md.append(f"- {r.a} ↔ {r.b} (상관 {r.corr:+.2f}): IV가 낮은 {lo} (IV {iv[lo]:.3f})가 후보.")
    else:
        md += ["VIF가 큰 변수부터, 그 변수와 가장 상관이 큰 상대와 비교해 IV가 낮은 쪽을 제거 후보로 두고 VIF가 5 미만이 될 때까지 반복한 결과이다.", "",
               "| 순서 | 제거 후보 | 근거(VIF 최대 변수) | VIF | 상관 상대 | 상관 | IV(후보) | IV(남길 쪽) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i, s in enumerate(sug, 1):
            md.append(f"| {i} | **{s['drop']}** | {s['worst_vif_var']} | {s['vif']:.2f} | {s['partner']} | {s['corr']:+.2f} | {s['iv_drop']:.3f} | {s['iv_keep']:.3f} |")
        md.append("")
        md.append(f"제거 후 남는 변수 {len(cur)}개. 실제 적용 여부는 사용자가 정한다 (적용 시 스코어카드·GBM·비교표를 다시 만들어야 한다).")
    md += ["", "## 4. 해석상 주의", "",
           "- WoE 변환값의 상관은 구간화 결과에 의존한다. 원 변수 값의 상관과 다를 수 있다.",
           "- VIF는 선형 모형(스코어카드)의 계수 안정성 지표이다. GBM에서는 같은 정보가 중복되어도 예측에는 문제가 없지만 SHAP 기여가 변수 사이로 나뉘어 해석이 흔들릴 수 있다.",
           "- 계수가 모두 음수(부호 정상)라는 점은 이미 확인했다. VIF가 낮으면 이 부호가 다른 변수 때문에 억지로 만들어진 값은 아니라는 근거가 된다.", ""]
    (P2 / "vif_check.md").write_text("\n".join(md), encoding="utf-8")
    pairs.to_csv(P2 / "vif_corr_pairs.csv", index=False, encoding="utf-8-sig")
    tab.to_csv(P2 / "vif_table.csv", encoding="utf-8-sig")
    log(f"4a 결과(vif_check.md): |상관|>{CORR_FLAG} 쌍 {len(pairs)}개 (family 간 {int((~pairs.same_family).sum()) if len(pairs) else 0}개), VIF 최대 {vif.max():.2f} ({vif.idxmax()}), 주의(≥{VIF_WARN:g}) {int((vif >= VIF_WARN).sum())}개, 문제(≥{VIF_BAD:g}) {int((vif >= VIF_BAD).sum())}개. "
        f"제거 제안(적용 안 함): {[s['drop'] for s in sug] or '없음'}", decision=True)


if __name__ == "__main__":
    main()
