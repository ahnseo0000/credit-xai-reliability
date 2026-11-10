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
