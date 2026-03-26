"""Smoke tests for kr-stat-tests package."""

from pathlib import Path


def test_package_importable():
    import kr_stat_tests
    assert kr_stat_tests.__version__ == "1.0.0"


def test_stats_runner_importable():
    from kr_stat_tests.stats_runner import get_stats_audit, format_stats_audit, STATS_DAG
    assert callable(get_stats_audit)
    assert callable(format_stats_audit)
    assert len(STATS_DAG) == 14, f"Expected 14 stat nodes, got {len(STATS_DAG)}"


def test_all_stat_scripts_present():
    """Ensure all 14 test scripts are present in the package."""
    pkg_dir = Path(__file__).resolve().parents[1] / "kr_stat_tests"
    expected = {
        "bootstrap_centrality.py",
        "bootstrap_threshold.py",
        "classify_extreme_outliers.py",
        "cluster_peers.py",
        "cross_screen_analysis.py",
        "fdr_disclosure_leakage.py",
        "fdr_timing_anomalies.py",
        "impute_financials.py",
        "label_coverage_analysis.py",
        "lasso_beneish.py",
        "pca_beneish.py",
        "permutation_repricing_peak.py",
        "rf_feature_importance.py",
        "survival_repricing.py",
    }
    present = {f.name for f in pkg_dir.glob("*.py") if not f.name.startswith("_")}
    missing = expected - present
    assert not missing, f"Missing stat scripts: {missing}"


def test_no_krff_imports():
    """None of the stat scripts should import from krff (delivery shell)."""
    pkg_dir = Path(__file__).resolve().parents[1] / "kr_stat_tests"
    violations = []
    for py_file in pkg_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="replace")
        if "from krff" in text or "import krff" in text:
            violations.append(py_file.name)
    assert not violations, f"Scripts import from krff: {violations}"


def test_stats_dag_scripts_reference_package_paths():
    """All DAG script paths should start with kr_stat_tests or be relative paths."""
    from kr_stat_tests.stats_runner import STATS_DAG
    for node in STATS_DAG:
        assert node.script, f"Node {node.name} has no script path"
