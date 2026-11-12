# 결정이 필요한 것 (L0 Tier 2 요약)

최종 누적 모형(10단계, 입력 190개): test AUC **0.7930**. 재학습 기준 4r(0.7774) 대비 +0.0156 [+0.0125, +0.0188], 처음 기준선(0.7695) 대비 +0.0234 [+0.0193, +0.0277]. 누수 중단은 한 번도 없었다 (후보 단독 train AUC 최대 0.596).

1. **증분이 0과 구분되지 않은 두 단계를 남길지.** 8단계 POS_CASH_balance(+0.0008 [−0.0007, +0.0023], 변수 9개)와 10단계 previous_application(+0.0002 [−0.0008, +0.0013], 변수 8개). 규칙대로 선택됐고 지우지 않았다. 나머지 6·7·9단계는 구간이 0을 제외한다.
2. **선택 규칙이 느슨하다.** "IV>=0.02이고 증분 AUC가 train·valid 모두 0보다 큼"이라 효과가 아주 작은 변수도 통과한다 (선택 106개 중 증분 AUC 0.001 미만이 다수). 임계(예: 증분 >= 0.0005)를 둘지, 상관 중복을 테이블 *사이*로 넓힐지 정해야 한다. 지금 중복 제거는 테이블 안에서만 했다.
3. **선택된 변수끼리 정보가 겹친다** (같은 측정값의 3/6/12/24개월 창, 건수·비율 등). 로지스틱·SHAP·반사실 설명 단계 전에 VIF/군집 단위 정리를 할지. family 열(`docs/variable_dict_l0.csv`)은 그 정리의 단위로 쓸 수 있다.
4. **연령 대리 측정**: PD–연령 순위상관 −0.268, PD만으로 연령 3구간 AUC 0.599 (기준선 −0.290 / 0.609). 새 변수가 연령 정보를 늘리지는 않았지만 0.6 수준의 정보가 남아 있다는 한계는 그대로다.
5. **시간 상한 초과.** 4시간 상한을 넘겼다 (이번 실행 총 446분). 10단계 학습 1회가 335분 걸렸는데(11:50→17:27) 같은 크기의 이전 학습은 10~20분이었다. 원인은 확인하지 못했다 (절전·메모리 부족 가능성). 결과(best_iter 1406, valid 0.7908)는 정상 범위라 그대로 썼다. 상한은 테이블 사이에서만 검사해서 학습 도중에는 막지 못했다.
6. **test를 여러 번 봤다.** 횟수와 목적은 `results/comparison_2x2/comparison.md` 한계 7번에 한 줄 추가했다 (이 파일 수정은 이번 지시로 한 것이며 표·숫자는 그대로). 별도 hold-out이 없다.
7. **Tier1에서 넘어온 미결**: EXT_SOURCE 계열 파생 4개의 중복 정리(IV 0.46~0.61)와 bureau_balance 유지 여부. bureau_balance는 Tier2에서 다시 만들지 않았다.
8. **분류 판단**: family 10종 중 몇 건은 해석 여지가 있다 (예: 카드 `UTIL`은 소진율, 할부 `SHORT_*`는 상환비율). 규칙은 `src/l0_family.py`에 있다.

---

# L0 Tier 1 실행 로그

시작 2026-11-10 01:45:40

- `01:45:40` 준비 결정
- `01:45:40` **[결정]** 루트 PROJECT_PLAN.md는 직전 지시(루트 내용을 docs에 반영하고 루트 삭제)로 이미 삭제됨. 복원하지 않고 docs/PROJECT_PLAN.md를 유지한다. docs/PROJECT_PLAN.md의 미커밋 수정(루트 반영분)은 이 실행에서 커밋하지 않는다.
- `01:45:40` **[결정]** v0-baseline 태그는 이미 d9cd7a7에 있음 (HL·PSI 결과가 d9cd7a7에서 처음 들어간 것을 git show --stat으로 확인함).
- `01:45:40` push는 하지 않는다. 커밋 메시지에 Co-Authored-By를 넣지 않는다. v0_baseline, v1_*, comparison_2x2는 수정하지 않는다.
- `01:45:43` 분할 확인: 기존 splits.csv와 일치 (train/valid/test = 215,257/46,127/46,127)
- `01:45:43` 기준 입력 변수 51개 (연령·성별 제외 확인)
- `01:45:45` 1단계(application_derived) 집계 완료: 9개 변수, 키 307,511개 (중복 없음), 1초
- `01:45:48` 2단계(bureau) 집계 완료: 13개 변수, 키 307,511개 (중복 없음), 3초
- `01:45:52` 3단계(previous_application) 집계 완료: 5개 변수, 키 307,511개 (중복 없음), 4초
- `01:46:01` 4단계(bureau_balance) 집계 완료: 6개 변수, 키 307,511개 (중복 없음), 9초
- `01:46:02` 0단계 기준선: AUC 0.7695, KS 0.4099, 입력 51개 (기존 v1_noage_noweight 예측 재사용)
- `01:48:30` 1단계(application_derived) 학습 2.3분: 입력 60개(+9), best_iter 894, AUC 0.7702 (직전 대비 +0.0006 [-0.0008, +0.0020]), KS 0.4083, 보정 전 ECE 0.0029 → 후 0.0030, PD-연령 순위상관 -0.291, 연령구간 AUC 0.609
- `01:48:31` 커밋: L0 1단계(application_derived): 변수 9개 추가, test AUC 0.7702 (5a8deb5)
- `01:51:50` 2단계(bureau) 학습 3.0분: 입력 73개(+13), best_iter 1037, AUC 0.7742 (직전 대비 +0.0041 [+0.0019, +0.0062]), KS 0.4072, 보정 전 ECE 0.0026 → 후 0.0028, PD-연령 순위상관 -0.286, 연령구간 AUC 0.607
- `01:51:50` 커밋: L0 2단계(bureau): 변수 13개 추가, test AUC 0.7742 (233c66c)
- `01:56:45` 3단계(previous_application) 학습 4.6분: 입력 78개(+5), best_iter 1587, AUC 0.7780 (직전 대비 +0.0038 [+0.0019, +0.0059]), KS 0.4230, 보정 전 ECE 0.0030 → 후 0.0028, PD-연령 순위상관 -0.272, 연령구간 AUC 0.602
- `01:56:45` 커밋: L0 3단계(previous_application): 변수 5개 추가, test AUC 0.7780 (7a72317)
- `02:00:24` 4단계(bureau_balance) 학습 3.3분: 입력 84개(+6), best_iter 1016, AUC 0.7769 (직전 대비 -0.0011 [-0.0024, +0.0000]), KS 0.4227, 보정 전 ECE 0.0017 → 후 0.0019, PD-연령 순위상관 -0.281, 연령구간 AUC 0.605
- `02:00:25` 커밋: L0 4단계(bureau_balance): 변수 6개 추가, test AUC 0.7769 (3ceaf2e)
- `02:00:27` IV 표 저장 (train 215,257행만 사용, 선별 안 함): 변수 84개, IV>0.5 1개 ['EXT_SOURCE_MEAN']
- `02:00:27` IV 상위 5: EXT_SOURCE_MEAN 0.611, EXT_SOURCE_MIN 0.462, EXT_SOURCE_MAX 0.450, EXT_SOURCE_PROD 0.379, EXT_SOURCE_3 0.327
- `02:00:27` 커밋: L0: IV 표 (train만, 선별 없음), IV>0.5 변수 1개 (1e01866)
- `02:00:27` 전체 소요 14.8분, 완료 단계 [1, 2, 3, 4], 중단 여부 False

## git log (최근 8개, 이 로그 커밋 직전 시점)

```
1e01866 L0: IV 표 (train만, 선별 없음), IV>0.5 변수 1개
3ceaf2e L0 4단계(bureau_balance): 변수 6개 추가, test AUC 0.7769
7a72317 L0 3단계(previous_application): 변수 5개 추가, test AUC 0.7780
233c66c L0 2단계(bureau): 변수 13개 추가, test AUC 0.7742
5a8deb5 L0 1단계(application_derived): 변수 9개 추가, test AUC 0.7702
ead87ba L0 Tier 1: 보조 테이블 집계 피처 모듈과 단계별 ablation 실행기 추가
0afbb54 결과: v1 조합 3종과 2x2 비교 (v1 기준 = 연령 제외, 가중치 없음)
230b6ed train에 --version/--exclude/--spw 옵션과 2x2 비교 스크립트 추가
```
- `02:00:28` 커밋: L0: RUNLOG 마무리 (4004b2f)


# L0 Tier 2 실행 로그 (2026-11-10 09:22)

테이블: installments_payments

- `09:22:01` push 안 함, Co-Authored-By 없음, v0_baseline/v1_*/comparison_2x2 결과 수정 안 함. 라벨 계산은 train/valid만, test는 선택이 끝난 묶음의 보고용.
- `09:22:01` **[결정]** 순서: installments_payments(6) → credit_card_balance(7) → POS_CASH_balance(8) → bureau(9) → previous_application(10). bureau_balance는 Tier1(4단계)에 이미 있어 Tier2에서 제외한다. bureau/previous_application은 Tier1에 없던 시간창·상태분리 슬라이스를 추가하는 후보 생성이다.
- `09:22:01` **[결정]** 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `09:22:10` 분할 확인 (train/valid/test = 215,257/46,127/46,127), 기준 입력 51개
- `09:31:07` **[결정]** 조립 방식 점검: Tier1 입력을 새 방식으로 재학습한 test AUC 0.7774 (기존 4단계 0.7769, 차이 +0.0005). 6단계부터 이 4r과 짝 비교한다.
- `09:31:56` installments_payments: 타깃 EDA 저장 (슬라이스 9개 중 유지 9개: ALL, L3M, L6M, L12M, L24M, LATE, LATE_L12M, SHORT, SHORT_L12M)
- `09:33:33` installments_payments: 후보 63개 생성, 키 307,511개 (중복 없음), 2.4분
- `09:33:34` 커밋: L0 Tier2 6단계(installments_payments): 타깃 EDA, 후보 63개 생성 (86224aa)
- `09:33:47` installments_payments: 후보 63개 → 상관 중복 제거 14개 → 증분 AUC 대상 49개 (IV는 train)
- `09:37:04` installments_payments: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 172열) train 3-fold 0.7693, valid 0.7744
- `09:44:37` installments_payments: 선택 42개 / 후보 63개 (11.1분). 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `09:44:37` installments_payments: IV 상위 5 INS_L24M_LATE_DAYS_MAX 0.094, INS_L12M_LATE_DAYS_MAX 0.081, INS_L24M_LATE_FLAG_MEAN 0.079, INS_LATE_L12M_LATE_DAYS_MAX 0.077, INS_LATE_L12M_LATE_DAYS_MEAN 0.073. IV>=0.02 56개, 단독 train AUC 최대 0.578
- `09:44:38` 커밋: L0 Tier2 6단계(installments_payments): 상관 중복 제거·증분 AUC·변수 선택 42개 (test 보기 전) (344c7ee)
- `10:00:34` 6단계(+installments_payments) 학습 14.7분: 입력 126개(+42), best_iter 1409, valid AUC 0.7840, **test AUC 0.7854** (직전 대비 +0.0080 [+0.0054, +0.0107]), KS 0.4312, 보정 전 ECE 0.0031 → 후 0.0029, PD-연령 순위상관 -0.267, 연령구간 AUC 0.599
- `10:00:35` 커밋: L0 Tier2 6단계(installments_payments): 선택 42개 누적, test AUC 0.7854 (36c7168)
- `10:00:35` 소요 39분, 테이블 ['installments_payments'], 누수 중단 False

## git log (최근 8개, 이 로그 커밋 직전 시점)

```
36c7168 L0 Tier2 6단계(installments_payments): 선택 42개 누적, test AUC 0.7854
344c7ee L0 Tier2 6단계(installments_payments): 상관 중복 제거·증분 AUC·변수 선택 42개 (test 보기 전)
86224aa L0 Tier2 6단계(installments_payments): 타깃 EDA, 후보 63개 생성
78e46ea L0: RUNLOG 마지막 줄 반영
4004b2f L0: RUNLOG 마무리
1e01866 L0: IV 표 (train만, 선별 없음), IV>0.5 변수 1개
3ceaf2e L0 4단계(bureau_balance): 변수 6개 추가, test AUC 0.7769
7a72317 L0 3단계(previous_application): 변수 5개 추가, test AUC 0.7780
```
- `10:00:35` 커밋: L0 Tier2: RUNLOG 갱신 (f0a844f)


# L0 Tier 2 실행 로그 (2026-11-10 10:01)

테이블: credit_card_balance, POS_CASH_balance, bureau, previous_application

- `10:01:26` push 안 함, Co-Authored-By 없음, v0_baseline/v1_*/comparison_2x2 결과 수정 안 함. 라벨 계산은 train/valid만, test는 선택이 끝난 묶음의 보고용.
- `10:01:26` **[결정]** 순서: installments_payments(6) → credit_card_balance(7) → POS_CASH_balance(8) → bureau(9) → previous_application(10). bureau_balance는 Tier1(4단계)에 이미 있어 Tier2에서 제외한다. bureau/previous_application은 Tier1에 없던 시간창·상태분리 슬라이스를 추가하는 후보 생성이다.
- `10:01:26` **[결정]** 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `10:01:36` 분할 확인 (train/valid/test = 215,257/46,127/46,127), 기준 입력 51개
- `10:01:38` 앞선 테이블 installments_payments: 선택된 변수 42개를 불러옴
- `10:02:15` credit_card_balance: 타깃 EDA 저장 (슬라이스 10개 중 유지 8개: ALL, L3M, L6M, L12M, L24M, COMPLETED, SIGNED, DPD_L12M)
- `10:02:15` credit_card_balance: EDA에서 신호가 약해 후보를 만들지 않은 슬라이스: ACTIVE, DPD
- `10:03:08` credit_card_balance: 후보 70개 생성, 키 307,511개 (중복 없음), 1.3분
- `10:03:09` 커밋: L0 Tier2 7단계(credit_card_balance): 타깃 EDA, 후보 70개 생성 (60b9e2e)
- `10:03:14` credit_card_balance: 후보 70개 → 상관 중복 제거 22개 → 증분 AUC 대상 48개 (IV는 train)
- `10:06:23` credit_card_balance: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 214열) train 3-fold 0.7778, valid 0.7806
- `10:13:29` credit_card_balance: 선택 28개 / 후보 70개 (10.3분). 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `10:13:29` credit_card_balance: IV 상위 5 CCB_L12M_UTIL_MAX 0.088, CCB_L6M_UTIL_MAX 0.085, CCB_L3M_UTIL_MAX 0.084, CCB_L24M_UTIL_MAX 0.082, CCB_L3M_UTIL_MEAN 0.075. IV>=0.02 35개, 단독 train AUC 최대 0.552
- `10:13:30` 커밋: L0 Tier2 7단계(credit_card_balance): 상관 중복 제거·증분 AUC·변수 선택 28개 (test 보기 전) (8fb8478)
- `10:36:22` 7단계(+credit_card_balance) 학습 21.2분: 입력 154개(+28), best_iter 1649, valid AUC 0.7855, **test AUC 0.7887** (직전 대비 +0.0033 [+0.0016, +0.0050]), KS 0.4361, 보정 전 ECE 0.0030 → 후 0.0027, PD-연령 순위상관 -0.268, 연령구간 AUC 0.600
- `10:36:23` 커밋: L0 Tier2 7단계(credit_card_balance): 선택 28개 누적, test AUC 0.7887 (756cf50)
- `10:37:10` POS_CASH_balance: 타깃 EDA 저장 (슬라이스 10개 중 유지 10개: ALL, L3M, L6M, L12M, L24M, ACTIVE, COMPLETED, RETURNED, DPD, DPD_L12M)
- `10:38:06` POS_CASH_balance: 후보 60개 생성, 키 307,511개 (중복 없음), 1.7분
- `10:38:08` 커밋: L0 Tier2 8단계(POS_CASH_balance): 타깃 EDA, 후보 60개 생성 (ade5020)
- `10:38:18` POS_CASH_balance: 후보 60개 → 상관 중복 제거 27개 → 증분 AUC 대상 33개 (IV는 train)
- `10:42:05` POS_CASH_balance: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 242열) train 3-fold 0.7788, valid 0.7821
- `10:47:20` POS_CASH_balance: 선택 9개 / 후보 60개 (9.2분). 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `10:47:20` POS_CASH_balance: IV 상위 5 POS_L24M_DPD_FLAG_SUM 0.034, POS_L24M_DPD_MEAN 0.033, POS_L24M_DPD_MAX 0.033, POS_L24M_DPD_FLAG_MEAN 0.033, POS_L24M_DPD_DEF_MAX 0.029. IV>=0.02 18개, 단독 train AUC 최대 0.533
- `10:47:21` 커밋: L0 Tier2 8단계(POS_CASH_balance): 상관 중복 제거·증분 AUC·변수 선택 9개 (test 보기 전) (fdf06b6)
- `10:58:16` 8단계(+POS_CASH_balance) 학습 10.1분: 입력 163개(+9), best_iter 957, valid AUC 0.7878, **test AUC 0.7895** (직전 대비 +0.0008 [-0.0007, +0.0023]), KS 0.4416, 보정 전 ECE 0.0027 → 후 0.0024, PD-연령 순위상관 -0.274, 연령구간 AUC 0.602
- `10:58:17` 커밋: L0 Tier2 8단계(POS_CASH_balance): 선택 9개 누적, test AUC 0.7895 (034008f)
- `10:58:32` bureau: 타깃 EDA 저장 (슬라이스 14개 중 유지 13개: ALL, L3M, L6M, L12M, L24M, ACTIVE, CLOSED, CONSUMER, CARD, CAR, MORTGAGE, MICRO, ACTIVE_L12M)
- `10:58:32` bureau: EDA에서 신호가 약해 후보를 만들지 않은 슬라이스: BADDEBT
- `10:59:05` bureau: 후보 94개 생성, 키 307,511개 (중복 없음), 0.8분
- `10:59:06` 커밋: L0 Tier2 9단계(bureau): 타깃 EDA, 후보 94개 생성 (e7dd303)
- `10:59:23` bureau: 후보 94개 → 상관 중복 제거 36개 → 증분 AUC 대상 58개 (IV는 train)
- `11:03:32` bureau: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 251열) train 3-fold 0.7813, valid 0.7854
- `11:10:18` bureau: 선택 19개 / 후보 94개 (11.2분). 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `11:10:19` bureau: IV 상위 5 BUR_ALL_DEBT_RATIO_MAX 0.148, BUR_ALL_DEBT_RATIO_MEAN 0.138, BUR_ALL_AGE_DAYS_MEAN 0.121, BUR_CARD_DEBT_RATIO_MEAN 0.101, BUR_ACTIVE_DEBT_RATIO_MEAN 0.093. IV>=0.02 53개, 단독 train AUC 최대 0.596
- `11:10:20` 커밋: L0 Tier2 9단계(bureau): 상관 중복 제거·증분 AUC·변수 선택 19개 (test 보기 전) (e21437a)
- `11:27:42` 9단계(+bureau) 학습 16.7분: 입력 182개(+19), best_iter 1414, valid AUC 0.7912, **test AUC 0.7927** (직전 대비 +0.0032 [+0.0014, +0.0050]), KS 0.4428, 보정 전 ECE 0.0029 → 후 0.0028, PD-연령 순위상관 -0.268, 연령구간 AUC 0.599
- `11:27:42` 커밋: L0 Tier2 9단계(bureau): 선택 19개 누적, test AUC 0.7927 (eb62e53)
- `11:27:58` previous_application: 타깃 EDA 저장 (슬라이스 26개 중 유지 22개: ALL, L3M, L6M, L12M, L24M, REFUSED, CASH, CONSUMER, REVOLVING, NEWCLI, REPEAT, RJ_HC, RJ_LIMIT, RJ_SCO, RJ_SCOFR, RJ_VERIF, PF_POS, PF_CASHP, PF_CARDS, REFUSED_L6M, REFUSED_L12M, REFUSED_L24M)
- `11:27:58` previous_application: EDA에서 신호가 약해 후보를 만들지 않은 슬라이스: APPROVED, CANCELED, UNUSED, APPROVED_L12M
- `11:28:27` previous_application: 후보 124개 생성, 키 307,511개 (중복 없음), 0.7분
- `11:28:28` 커밋: L0 Tier2 10단계(previous_application): 타깃 EDA, 후보 124개 생성 (6d13733)
- `11:28:50` previous_application: 후보 124개 → 상관 중복 제거 17개 → 증분 AUC 대상 107개 (IV는 train)
- `11:32:50` previous_application: 증분 AUC 기준(신청서+Tier1+앞서 선택된 변수 270열) train 3-fold 0.7834, valid 0.7877
- `11:50:16` previous_application: 선택 8개 / 후보 124개 (21.8분). 선택 규칙(사전 고정): 상관 0.95 초과 중복이 아니고(테이블 안, IV 높은 쪽을 남김) AND train IV >= 0.02 AND train 3-fold 증분 AUC > 0 AND valid 증분 AUC > 0.
- `11:50:16` previous_application: IV 상위 5 PRV_ALL_CREDIT_APP_RATIO_MEAN 0.071, PRV_ALL_REFUSED_FLAG_MEAN 0.070, PRV_L24M_REFUSED_FLAG_MEAN 0.069, PRV_L24M_REFUSED_FLAG_SUM 0.067, PRV_REFUSED_L24M_CNT 0.066. IV>=0.02 65개, 단독 train AUC 최대 0.572
- `11:50:18` 커밋: L0 Tier2 10단계(previous_application): 상관 중복 제거·증분 AUC·변수 선택 8개 (test 보기 전) (a8188ad)
- `17:27:26` 10단계(+previous_application) 학습 335.4분: 입력 190개(+8), best_iter 1406, valid AUC 0.7908, **test AUC 0.7930** (직전 대비 +0.0002 [-0.0008, +0.0013]), KS 0.4464, 보정 전 ECE 0.0027 → 후 0.0024, PD-연령 순위상관 -0.268, 연령구간 AUC 0.599
- `17:27:26` 커밋: L0 Tier2 10단계(previous_application): 선택 8개 누적, test AUC 0.7930 (5256b62)
- `17:27:27` 소요 446분, 테이블 ['credit_card_balance', 'POS_CASH_balance', 'bureau', 'previous_application'], 누수 중단 False

## git log (최근 8개, 이 로그 커밋 직전 시점)

```
5256b62 L0 Tier2 10단계(previous_application): 선택 8개 누적, test AUC 0.7930
a8188ad L0 Tier2 10단계(previous_application): 상관 중복 제거·증분 AUC·변수 선택 8개 (test 보기 전)
6d13733 L0 Tier2 10단계(previous_application): 타깃 EDA, 후보 124개 생성
eb62e53 L0 Tier2 9단계(bureau): 선택 19개 누적, test AUC 0.7927
e21437a L0 Tier2 9단계(bureau): 상관 중복 제거·증분 AUC·변수 선택 19개 (test 보기 전)
e7dd303 L0 Tier2 9단계(bureau): 타깃 EDA, 후보 94개 생성
034008f L0 Tier2 8단계(POS_CASH_balance): 선택 9개 누적, test AUC 0.7895
fdf06b6 L0 Tier2 8단계(POS_CASH_balance): 상관 중복 제거·증분 AUC·변수 선택 9개 (test 보기 전)
```
- `17:27:27` 커밋: L0 Tier2: RUNLOG 갱신 (4c732bf)
- `16:49:43` 절전 방지 설정 (SetThreadExecutionState)


# L0 v2 정리·동결 (timing, 2026-11-11 16:49)

- `16:50:00` 190개 모형(신청서 51 + L0 139) 정상 속도 점검: 학습 1회, valid만 평가 (test 미사용)
- `17:07:31` 학습 1회 17.4분 (best_iter 1406, 입력 열 278), valid AUC 0.7908 (Tier2 10단계 기록 valid 0.7908과 비교). 모형은 C:\Users\안서영\Desktop\kubig\credit-xai-llm\results\l0_v2\raw\model_190.json에 저장
- `17:08:14` 절전 방지 설정 (SetThreadExecutionState)


# L0 v2 정리·동결 (sweep, 2026-11-11 17:08)

- `17:08:45` **[결정]** K-sweep 시작: 190개 모형 1회 17.4분 → K = [40, 70, 100] + 190(재사용). test는 쓰지 않는다.
- `17:08:46` gain 중요도(train, total_gain 변수 단위): 상위 5 EXT_SOURCE_MEAN 19.1%, EXT_SOURCE_MIN 4.1%, EXT_SOURCE_MAX 3.4%, CREDIT_TERM 3.0%, DAYS_EMPLOYED 2.5%. gain 0인 변수 2개, 상위 40개가 69.9%, 70개 84.5%, 100개 92.4%
- `17:08:50` family 안 상관 중복 제거: L0 139개 중 1개 (family별 {'연체이력': 1})
- `17:08:50` 순위 대상 변수 189개 (신청서 51 + L0 138)
- `17:15:03` K=40: 변수 15+25, 학습 6.0분, valid AUC 0.7840, 보정 전 ECE 0.0020
- `17:21:20` K=70: 변수 20+50, 학습 6.1분, valid AUC 0.7868, 보정 전 ECE 0.0018
- `17:27:41` K=100: 변수 24+76, 학습 6.2분, valid AUC 0.7898, 보정 전 ECE 0.0024
- `17:27:41` **[결정]** K 선택(valid만): 최고 valid AUC 0.7908, 0.002 이내 가장 작은 K = 100
- `17:27:41` v2 후보: K=100, 신청서 24 + L0 76 = 100개. L0 제외 63개는 변수 사전 use 열에 사유 기록(지우지 않음). 제외된 신청서 변수 ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_HOUSING_TYPE', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
- `17:27:42` 커밋: L0 v2: gain 중요도 K-sweep, valid 기준 K=100 후보 확정 (test 미사용) (5231203)
- `18:59:50` 절전 방지 설정 (SetThreadExecutionState)


# L0 v2 정리·동결 (final, 2026-11-11 18:59)

- `19:00:06` v2 후보 K=100: 신청서 24 + L0 76 = 100개 입력
- `19:00:17` 저장된 모형 무결성 확인: valid 예측 일치, valid AUC 0.7898
- `19:00:18` **[결정]** Platt 필요 여부(valid): 보정 전 ECE 0.0024, Platt in-sample 0.0022, cross-fit 0.0026; Brier 0.0657 → 0.0657. 규칙: cross-fit ECE가 0.001 이상 낮을 때만 사용 → **사용 안 함(원 확률 그대로)**
- `19:00:53` 로지스틱 C=1e+06: valid AUC 0.7740
- `19:01:05` 로지스틱 C=1: valid AUC 0.7741
- `19:01:18` 로지스틱 C=0.1: valid AUC 0.7742
- `19:01:29` 로지스틱 C=0.01: valid AUC 0.7743
- `19:01:30` **[결정]** 로지스틱 챔피언: C=0.01, valid AUC 0.7743, 입력 100개 + 결측 표시 54개(패턴 중복 합침)
- `20:03:20` **[결정]** 보고 코드 dry run(valid) 통과. 이제 test를 처음으로 최종 1회 평가한다. 이후 test로 아무 결정도 하지 않는다.
- `20:04:07` **[결정]** **test 최종 1회**: xgb_v2 AUC 0.7900, 로지스틱 AUC 0.7757
- `20:04:07` interface.md 채움: q70 0.082282, q90 0.194800, 18칸 최소 크기 133 (합 46,127)
- `20:04:09` 동결 파일 저장: results/v2_frozen/
- `20:04:10` 커밋: v2 동결: K=100 입력 100개, test 최종 1회 AUC 0.7900 (로지스틱 0.7757), interface.md q70/q90 (a19822b)
- `20:04:10` 태그 v2-frozen: a19822b
