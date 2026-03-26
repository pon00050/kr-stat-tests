"""kr-stat-tests — Statistical validation suite for Korean forensic accounting.

14 tests validating the forensic screening methodology:
PCA, clustering, FDR, bootstrap, LASSO, random forest, survival analysis,
permutation tests, imputation, outlier classification, cross-screen, label coverage.

Usage:
    from kr_stat_tests.stats_runner import get_stats_audit, format_stats_audit
    result = get_stats_audit()
    print(format_stats_audit(result))
"""

from __future__ import annotations

from kr_stat_tests.stats_runner import get_stats_audit, format_stats_audit  # noqa: F401

__version__ = "1.0.0"
__all__ = ["get_stats_audit", "format_stats_audit"]
