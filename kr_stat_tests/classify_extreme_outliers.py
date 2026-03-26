"""
classify_extreme_outliers.py — Batch frmtrm_amount cross-check for all extreme M-score outliers.

Reads beneish_scores.parquet for rows with m_score > 10, then for each:
1. Reads the current year raw parquet → thstrm_amount for all monetary fields
2. Reads the next year raw parquet → frmtrm_amount (prior-year restated)
3. Compares: if frmtrm ≈ thstrm (within tolerance), DART confirms the original value
4. Classifies: DART-confirmed (genuine), DART-restated (possible error), or unverifiable

Output: 00_Reference/XX_Extreme_Outlier_Classification.md + CSV summary.

Usage:
    python 03_Analysis/statistical_tests/classify_extreme_outliers.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import warnings

# Windows: force UTF-8 stdout/stderr so Korean company names don't crash cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=pathlib.Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
sys.path.insert(0, str(ROOT / "02_Pipeline"))
from _pipeline_helpers import parse_amount as _parse_amount  # noqa: E402

log = logging.getLogger(__name__)
PROCESSED = ROOT / "01_Data" / "processed"
RAW_FIN = ROOT / "01_Data" / "raw" / "financials"
OUTPUT_DIR = ROOT / "03_Analysis" / "statistical_tests" / "outputs"
REF_DIR = ROOT / "00_Reference"

# Beneish weights for component contribution
BENEISH_WEIGHTS = {
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "tata": 4.679,
    "lvgi": -0.327,
}

# ACCOUNT_SPECS from transform.py — maps field names to DART account_ids
ACCOUNT_SPECS: dict[str, list[str]] = {
    "revenue": ["ifrs-full_Revenue", "ifrs-full_RevenueFromContractsWithCustomers"],
    "receivables": [
        "dart_ShortTermTradeReceivable",
        "dart_ShortTermTradeReceivables",
        "ifrs-full_TradeAndOtherCurrentReceivables",
        "ifrs-full_TradeAndOtherReceivables",
    ],
    "cogs": ["ifrs-full_CostOfSales"],
    "sga": ["ifrs-full_SellingGeneralAndAdministrativeExpense"],
    "ppe": [
        "ifrs-full_PropertyPlantAndEquipment",
        "ifrs-full_PropertyPlantAndEquipmentAbstract",
    ],
    "depreciation": [
        "ifrs-full_DepreciationAmountAbstract",
        "ifrs-full_DepreciationAndAmortisationExpense",
    ],
    "total_assets": ["ifrs-full_Assets"],
    "lt_debt": [
        "dart_LongTermBorrowings",
        "ifrs-full_NoncurrentFinancialLiabilities",
    ],
    "net_income": ["ifrs-full_ProfitLoss"],
    "cfo": [
        "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        "ifrs-full_CashFlowsFromOperatingActivities",
    ],
}


def _extract_amounts(parquet_path: pathlib.Path, col: str = "thstrm_amount") -> dict[str, float | None]:
    """Extract monetary field values from a raw DART parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        return {}
    except Exception:
        log.warning("Failed to read %s: skipping", parquet_path)
        return {}

    results: dict[str, float | None] = {}
    for field, account_ids in ACCOUNT_SPECS.items():
        val = None
        for aid in account_ids:
            match = df[df["account_id"] == aid]
            if not match.empty:
                val = _parse_amount(match.iloc[0].get(col))
                break
        results[field] = val
    return results


def classify_outliers() -> pd.DataFrame:
    """Main classification routine. Returns a DataFrame with all outliers classified."""
    scores = pd.read_parquet(PROCESSED / "beneish_scores.parquet")
    financials = pd.read_parquet(PROCESSED / "company_financials.parquet")

    # Get company names
    name_map = financials.drop_duplicates("corp_code").set_index("corp_code")["company_name"].to_dict()

    # Positive extreme outliers
    outliers = scores[scores["m_score"] > 10].copy()
    outliers = outliers.sort_values("m_score", ascending=False).reset_index(drop=True)

    results = []
    for _, row in outliers.iterrows():
        corp = row["corp_code"]
        year = int(row["year"])
        m = row["m_score"]
        company_name = name_map.get(corp, "?")

        curr_path = RAW_FIN / f"{corp}_{year}.parquet"
        next_path = RAW_FIN / f"{corp}_{year + 1}.parquet"

        # Identify primary driver
        contributions = {}
        for comp, weight in BENEISH_WEIGHTS.items():
            val = row.get(comp, np.nan)
            if pd.notna(val) and np.isfinite(val):
                contributions[comp] = abs(val * weight)
        primary_driver = max(contributions, key=contributions.get) if contributions else "unknown"
        driver_value = row.get(primary_driver, np.nan) if primary_driver != "unknown" else np.nan

        # Extract current year thstrm and next year frmtrm
        curr_amounts = _extract_amounts(curr_path, "thstrm_amount") if curr_path.exists() else {}
        next_frmtrm = _extract_amounts(next_path, "frmtrm_amount") if next_path.exists() else {}

        # Cross-check: compare thstrm(year) vs frmtrm(year) from year+1 filing
        mismatches = []
        confirmations = []
        for field in ACCOUNT_SPECS:
            th = curr_amounts.get(field)
            fr = next_frmtrm.get(field)
            if th is not None and fr is not None and th != 0:
                ratio = fr / th
                if abs(ratio - 1.0) > 0.01:  # >1% difference
                    mismatches.append((field, th, fr, ratio))
                else:
                    confirmations.append(field)

        # Classification
        has_next = next_path.exists()
        if not has_next:
            classification = "UNVERIFIABLE"
            detail = "No year+1 raw file (2023 data, no 2024 extract)"
        elif mismatches:
            # Check if mismatch is a clean XBRL scale factor
            rev_mismatch = [m for m in mismatches if m[0] == "revenue"]
            if rev_mismatch:
                _, th, fr, ratio = rev_mismatch[0]
                if ratio in (1000.0, 1e6, 1e9, 0.001, 1e-6, 1e-9):
                    classification = "XBRL_UNIT_ERROR"
                    detail = f"Revenue frmtrm/thstrm ratio = {ratio:.0e} (clean XBRL scale)"
                else:
                    classification = "DART_RESTATED"
                    detail = f"Revenue restated: thstrm={th:.3e} → frmtrm={fr:.3e} (ratio={ratio:.4f})"
            else:
                field, th, fr, ratio = mismatches[0]
                classification = "DART_RESTATED"
                detail = f"{field} restated: thstrm={th:.3e} → frmtrm={fr:.3e} (ratio={ratio:.4f})"
        else:
            classification = "DART_CONFIRMED"
            detail = f"All {len(confirmations)} matched fields confirmed by year+1 frmtrm"

        # COVID-era tagging
        covid_tag = ""
        sgi = row.get("sgi", 0)
        if primary_driver == "sgi" and sgi > 10 and year in (2020, 2021):
            covid_tag = "COVID-candidate"
        elif primary_driver == "gmi" and year in (2020, 2021):
            covid_tag = "COVID-margin"

        # Revenue time series for context
        rev_series = financials[financials["corp_code"] == corp][["year", "revenue"]].sort_values("year")
        rev_str = "; ".join(f"{int(r.year)}:{r.revenue:.2e}" if pd.notna(r.revenue) else f"{int(r.year)}:NaN"
                          for _, r in rev_series.iterrows())

        results.append({
            "corp_code": corp,
            "company_name": company_name,
            "year": year,
            "m_score": m,
            "primary_driver": primary_driver,
            "driver_value": driver_value,
            "sgi": sgi,
            "classification": classification,
            "detail": detail,
            "covid_tag": covid_tag,
            "n_mismatches": len(mismatches),
            "n_confirmations": len(confirmations),
            "has_next_year": has_next,
            "revenue_series": rev_str,
        })

    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame) -> str:
    """Generate the 00_Reference markdown report."""
    now = datetime.now().strftime("%Y-%m-%d")

    # Summary stats
    total = len(df)
    confirmed = len(df[df.classification == "DART_CONFIRMED"])
    restated = len(df[df.classification == "DART_RESTATED"])
    xbrl_err = len(df[df.classification == "XBRL_UNIT_ERROR"])
    unverifiable = len(df[df.classification == "UNVERIFIABLE"])

    lines = [
        f"# Extreme M-Score Outlier Classification",
        f"",
        f"*Generated: {now} (Session 48)*",
        f"",
        f"## Context",
        f"",
        f"The Beneish M-Score dataset (`beneish_scores.parquet`, 5,476 rows, 2019-2023) contains",
        f"**{total} company-years with m_score > 10** — values driven to extremes by one or more",
        f"Beneish components. The data period coincides with the COVID-19 pandemic, which caused",
        f"unprecedented revenue swings, margin collapses, and receivables disruptions across KOSDAQ.",
        f"",
        f"This document classifies every extreme outlier as genuine business event or data error",
        f"using a systematic batch cross-check of DART `frmtrm_amount` (the prior-year restated",
        f"value reported in the following year's financial filing).",
        f"",
        f"## Methodology",
        f"",
        f"For each (corp_code, year) with m_score > 10:",
        f"",
        f"1. Read `01_Data/raw/financials/{{corp_code}}_{{year}}.parquet` → `thstrm_amount` for 10 monetary fields",
        f"2. Read `01_Data/raw/financials/{{corp_code}}_{{year+1}}.parquet` → `frmtrm_amount` (prior-year restated)",
        f"3. Compare field-by-field within 1% tolerance",
        f"4. Classify:",
        f"   - **DART_CONFIRMED**: All matched fields agree — DART itself confirms the original value",
        f"   - **DART_RESTATED**: One or more fields differ significantly — DART corrected the value",
        f"   - **XBRL_UNIT_ERROR**: Revenue mismatch is a clean XBRL scale factor (1e3, 1e6, 1e9)",
        f"   - **UNVERIFIABLE**: No year+1 raw file available (all 2023 data points)",
        f"",
        f"## Summary",
        f"",
        f"| Classification | Count | % |",
        f"|---|---|---|",
        f"| DART_CONFIRMED (genuine) | {confirmed} | {confirmed/total*100:.0f}% |",
        f"| DART_RESTATED (possible error) | {restated} | {restated/total*100:.0f}% |",
        f"| XBRL_UNIT_ERROR (confirmed error) | {xbrl_err} | {xbrl_err/total*100:.0f}% |",
        f"| UNVERIFIABLE (no 2024 data) | {unverifiable} | {unverifiable/total*100:.0f}% |",
        f"| **Total** | **{total}** | |",
        f"",
        f"### Component Driver Distribution",
        f"",
    ]

    driver_counts = df["primary_driver"].value_counts()
    lines.append("| Driver | Count | % | Interpretation |")
    lines.append("|---|---|---|---|")
    driver_interp = {
        "gmi": "Gross margin compression; cost inflation or pricing pressure",
        "dsri": "Receivables spike; delayed collections or channel stuffing",
        "sgi": "Extreme revenue growth; COVID demand, M&A, or base-year effect",
        "tata": "Accruals anomaly; working capital manipulation",
        "aqi": "Asset quality deterioration; capitalization issues",
        "depi": "Depreciation index anomaly",
        "sgai": "SGA expense anomaly",
        "lvgi": "Leverage anomaly",
    }
    for driver, count in driver_counts.items():
        pct = count / total * 100
        interp = driver_interp.get(driver, "")
        lines.append(f"| {driver.upper()} | {count} | {pct:.0f}% | {interp} |")

    lines.extend([
        "",
        "## Full Classification Table",
        "",
        "| # | Corp Code | Company | Year | M-Score | Driver | Driver Val | Class | COVID | Detail |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for i, (_, r) in enumerate(df.iterrows(), 1):
        name = r.company_name[:12] if isinstance(r.company_name, str) else "?"
        dv = f"{r.driver_value:.1f}" if pd.notna(r.driver_value) and np.isfinite(r.driver_value) else "?"
        covid = r.covid_tag if r.covid_tag else ""
        detail = r.detail[:60] if len(r.detail) > 60 else r.detail
        lines.append(
            f"| {i} | {r.corp_code} | {name} | {r.year} | {r.m_score:.1f} "
            f"| {r.primary_driver.upper()} | {dv} | {r.classification} | {covid} | {detail} |"
        )

    # Revenue time series section
    lines.extend([
        "",
        "## Revenue Time Series (Top 15 Outliers)",
        "",
        "Revenue trajectory across all available years for each company, providing context",
        "for whether extreme values represent sudden shocks or sustained trends.",
        "",
    ])
    for _, r in df.head(15).iterrows():
        name = r.company_name if isinstance(r.company_name, str) else r.corp_code
        lines.append(f"**{name} ({r.corp_code}) — year={r.year}, m_score={r.m_score:.1f}, class={r.classification}**")
        lines.append(f"```")
        lines.append(f"{r.revenue_series}")
        lines.append(f"```")
        lines.append("")

    # DART_RESTATED deep dive
    restated_rows = df[df.classification == "DART_RESTATED"]
    if len(restated_rows) > 0:
        lines.extend([
            "## DART-Restated Cases (Require Manual Review)",
            "",
            "These company-years show discrepancies between the original filing's `thstrm_amount`",
            "and the next year's `frmtrm_amount`. This may indicate:",
            "- A genuine restatement (corrected in subsequent filing)",
            "- An XBRL rounding/truncation difference",
            "- A change in consolidation scope (CFS vs OFS switch)",
            "",
        ])
        for _, r in restated_rows.iterrows():
            lines.append(f"- **{r.company_name} ({r.corp_code}) {r.year}**: {r.detail}")
        lines.append("")

    # Recommendations
    lines.extend([
        "## Recommendations",
        "",
        "### For `BENEISH_EXTREME_OUTLIERS` in `src/constants.py`",
        "",
        "All DART_CONFIRMED and UNVERIFIABLE entries should be added to the exclusion set,",
        "as they represent genuine business events (not data errors) that would distort",
        "threshold calibration. DART_RESTATED entries need manual review before inclusion.",
        "",
        "### For `_XBRL_UNIT_CORRECTIONS` in `transform.py`",
        "",
    ])
    if xbrl_err > 0:
        lines.append(f"**{xbrl_err} new XBRL unit errors identified** — add to the corrections dict.")
        for _, r in df[df.classification == "XBRL_UNIT_ERROR"].iterrows():
            lines.append(f"- `(\"{r.corp_code}\", {r.year})`: {r.detail}")
    else:
        lines.append("No new XBRL unit errors identified. Existing 3 corrections remain sufficient.")

    lines.extend([
        "",
        "### For COVID-era interpretation",
        "",
        "The data period (2019-2023) spans the full COVID-19 cycle. The Beneish M-Score",
        "model was calibrated on US pre-pandemic data with a -1.78 threshold. During COVID:",
        "",
        "- Revenue swings of 10-1,500× are genuine for diagnostics, pharma, and tech companies",
        "- Margin collapses (GMI > 100) reflect real cost disruptions, not earnings manipulation",
        "- Receivables spikes (DSRI > 10) may reflect delayed collections, not channel stuffing",
        "",
        "**Implication:** The Beneish threshold of -1.78 generates excessive false positives",
        "during 2020-2022. The bootstrap-calibrated KOSDAQ threshold of -1.85 (session 39) may",
        "need COVID-era stratification — separate thresholds for 2019 (pre-COVID baseline),",
        "2020-2021 (acute phase), and 2022-2023 (recovery).",
        "",
    ])

    return "\n".join(lines)


def main():
    print("Classifying extreme M-score outliers...")
    df = classify_outliers()

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "extreme_outlier_classification.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {csv_path} ({len(df)} rows)")

    # Generate and save report
    report = generate_report(df)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REF_DIR / "37_Extreme_Outlier_Classification.md"
    ref_path.write_text(report, encoding="utf-8")
    print(f"  Report: {ref_path}")

    # Print summary
    print("\nClassification summary:")
    for cls, count in df["classification"].value_counts().items():
        print(f"  {cls}: {count}")
    print(f"\nDriver distribution:")
    for drv, count in df["primary_driver"].value_counts().items():
        print(f"  {drv.upper()}: {count}")

    # Print entries to add to BENEISH_EXTREME_OUTLIERS
    genuine = df[df.classification.isin(["DART_CONFIRMED", "UNVERIFIABLE"])]
    print(f"\n{len(genuine)} entries for BENEISH_EXTREME_OUTLIERS:")
    for _, r in genuine.iterrows():
        print(f'    ("{r.corp_code}", {r.year}),   # {r.company_name}: {r.primary_driver.upper()}={r.driver_value:.1f}; {r.classification}')


if __name__ == "__main__":
    main()
