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

---

## NEVER commit to this repo

This repository is **public**. Before staging or writing any new file, check the list below. If the content matches any item, route it to the gitignored working directory of the coordination hub instead, NOT to this repo.

**Hard NO list:**

1. **Any API key, token, or credential — even a truncated fingerprint.** This includes Anthropic key fingerprints (sk-ant-...), AWS keys (AKIA...), GitHub tokens (ghp_...), DART/SEIBRO/KFTC API keys, FRED keys. Even partial / display-truncated keys (e.g. "sk-ant-api03-...XXXX") leak the org-to-key linkage and must not be committed.
2. **Payment / billing data of any kind.** Card numbers (full or last-four), invoice IDs, receipt numbers, order numbers, billing-portal URLs, Stripe/Anthropic/PayPal account states, monthly-spend caps, credit balances.
3. **Vendor support correspondence.** Subject lines, body text, ticket IDs, or summaries of correspondence with Anthropic / GitHub / Vercel / DART / any vendor's support team.
4. **Named third-party outreach targets.** Specific company names, hedge-fund names, audit-firm names, regulator-individual names appearing in a planning, pitch, or outreach context. Engineering content discussing Korean financial institutions in a neutral domain context (e.g. "DART is the FSS disclosure system") is fine; planning text naming them as a sales target is not.
5. **Commercial-positioning memos.** Documents discussing buyer segments, monetization models, pricing strategy, competitor analysis, market positioning, or go-to-market plans. Research methodology and technical roadmaps are fine; commercial strategy is not.
6. **Files matching the leak-prevention .gitignore patterns** (*_prep.md, *_billing*, *_outreach*, *_strategy*, *_positioning*, *_pricing*, *_buyer*, *_pitch*, product_direction.md, etc.). If you find yourself wanting to write a file with one of these names, that is a signal that the content belongs in the hub working directory.

**When in doubt:** put the content in the hub working directory (gitignored), not this repo. It is always safe to add later. It is expensive to remove after force-pushing — orphaned commits remain resolvable on GitHub for weeks.

GitHub Push Protection is enabled on this repo and will reject pushes containing well-known credential patterns. That is a backstop, not the primary defense — write-time discipline is.
