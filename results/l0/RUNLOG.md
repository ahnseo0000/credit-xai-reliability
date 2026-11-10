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
