# v2 동결

- `models/xgb_v2.json` 모형, `models/xgb_preprocessor.joblib` 신청서 변수 전처리(train 적합), `models/logit_champion.joblib` 로지스틱 챔피언
- `features.json` 입력 변수(순서 고정), `variable_dict_v2_*.csv` 변수 사전, `config.json` 시드·하이퍼파라미터·버전
- `test_final/` test 최종 1회 평가 (ablation과 별개). `k_sweep.*` 변수 수 결정 근거
- `raw/`(git 제외) 예측 파일
- L0 변수는 `results/l0/cand_*.parquet`, `features_stage*.parquet` 집계에서 만든다 (git 제외). 재생성: `python -m src.l0_features`/`src.l0_tier2_run --reuse`
