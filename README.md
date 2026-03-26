# kr-stat-tests

Statistical validation suite for the Korean forensic accounting methodology.

14 tests: PCA, clustering, FDR, bootstrap, LASSO, random forest, survival analysis,
permutation tests, imputation, outlier classification, cross-screen, label coverage.

Part of the [forensic accounting toolkit](https://github.com/pon00050/forensic-accounting-toolkit) ecosystem.

## Install

```bash
pip install kr-stat-tests
```

## Usage

```python
from kr_stat_tests import get_stats_audit, format_stats_audit

result = get_stats_audit()
print(format_stats_audit(result))
```

## Data Requirements

Set `KRFF_DATA_DIR` to the processed data directory:

```bash
export KRFF_DATA_DIR=/path/to/01_Data/processed
```

## Tests

| # | Script | Input | Output |
|---|--------|-------|--------|
| 1 | `fdr_timing_anomalies.py` | timing_anomalies.csv | fdr_timing_anomalies.csv |
| 2 | `fdr_disclosure_leakage.py` | centrality_report.csv | fdr_disclosure_leakage.csv |
| 3 | `pca_beneish.py` | beneish_scores.parquet | pca_pc3_scores.csv |
| 4 | `cluster_peers.py` | beneish_scores.parquet | peer_clusters.csv |
| 5 | `impute_financials.py` | beneish_scores.parquet | company_financials_imputed.parquet |
| 6 | `bootstrap_centrality.py` | centrality_report.csv | bootstrap_centrality.csv |
| 7 | `bootstrap_threshold.py` | beneish_scores.parquet | bootstrap_threshold.csv |
| 8 | `permutation_repricing_peak.py` | cb_bw_summary.csv | permutation_repricing_peak.csv |
| 9 | `lasso_beneish.py` | beneish_scores.parquet | lasso_feature_importance.csv |
| 10 | `rf_feature_importance.py` | beneish_scores.parquet + labels.csv | rf_feature_importance.csv |
| 11 | `survival_repricing.py` | cb_bw_events.parquet | survival_repricing.csv |
| 12 | `cross_screen_analysis.py` | multiple | cross_screen_analysis.csv |
| 13 | `label_coverage_analysis.py` | beneish_scores.parquet | label_coverage_analysis.csv |
| 14 | `classify_extreme_outliers.py` | beneish_scores.parquet | outlier_classification.csv |
