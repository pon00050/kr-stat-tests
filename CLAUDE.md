# kr-stat-tests
> 14 statistical validation scripts for the forensic accounting pipeline — PCA, bootstrap, LASSO, random forest, survival analysis, FDR correction, and more.

## Install & Test
```bash
uv sync
uv run pytest tests/ -v
```

## Architecture

Package root: `kr_stat_tests/`

14 analysis scripts (each runnable standalone):
- `pca_beneish.py` — PCA on Beneish ratio space
- `lasso_beneish.py` — LASSO feature selection for M-Score predictors
- `rf_feature_importance.py` — Random forest feature importance
- `bootstrap_threshold.py` — Bootstrap CI for Beneish threshold calibration
- `bootstrap_centrality.py` — Bootstrap CI for officer network centrality
- `cluster_peers.py` — Peer clustering for comparative analysis
- `cross_screen_analysis.py` — Cross-screen signal correlation
- `fdr_disclosure_leakage.py` — FDR-corrected test for pre-disclosure leakage
- `fdr_timing_anomalies.py` — FDR-corrected timing anomaly significance
- `impute_financials.py` — Financial statement imputation for missing data
- `label_coverage_analysis.py` — Coverage analysis for labeled enforcement cases
- `permutation_repricing_peak.py` — Permutation test for repricing-at-peak signal
- `survival_repricing.py` — Survival analysis of repricing events
- `classify_extreme_outliers.py` — Classifies moneyness outliers (SPLIT_ARTIFACT / GENUINE_ITM / DATA_ERROR)

Orchestrated via `stats_runner.py`. Reads parquets from `kr_forensic_core.paths.data_dir()`.

## Conventions
- Package manager: `uv`
- Build system: hatchling
- Test command: `uv run pytest tests/ -v`
- Each script is independently runnable and importable

## Key Decisions
- `classify_extreme_outliers.py` inlines `parse_amount()` (14 lines) rather than importing from kr-dart-pipeline, to keep this repo free of platform dependencies. Repos in this layer should not import from ETL repos.
- `stats_runner.py` sibling lookup simplified in March 2026 to target `krff-shell` only (removed tuple of alternatives).

## Known Gaps

| Gap | Why | Status |
|-----|-----|--------|
| Only 5 tests for 14 analysis scripts | Scripts require parquet inputs not available in CI | Deferred — need fixture parquets |
| `survival_repricing.py` not validated against known fraud cases | Needs labeled enforcement dataset integration + SEIBRO repricing data (blocked by XB-002) | Blocked — SEIBRO API ETA end of April 2026 |


---

**Working notes** (regulatory analysis, legal compliance research, or anything else not appropriate for this public repo) belong in the gitignored working directory of the coordination hub. Engineering docs (API patterns, test strategies, run logs) stay here.
