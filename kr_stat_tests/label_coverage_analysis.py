"""
label_coverage_analysis.py
--------------------------
Cross-reference confirmed fraud=1 labels against pipeline screens to measure
coverage and identify blind spots.

For each fraud=1 company, answers:
  - in_beneish_screen : has at least one m_score > -1.78
  - in_cb_screen      : appears in cb_bw_summary with flag_count >= 1
  - max_flag_count    : highest flag_count across all CB/BW events
  - in_dual_screen    : appears in double_flagged_companies.csv (PC3 + CB/BW cross-screen)
  - pc3_rank          : best PC3_pct_rank in double_flagged if present, else NaN

Outputs:
  - Console table
  - outputs/label_coverage.csv

Run with:
  uv run python 03_Analysis/statistical_tests/label_coverage_analysis.py
"""

import pathlib
import sys
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows (Korean company names in tables)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

LABELS_CSV = HERE / "labels.csv"
CB_BW_CSV = HERE.parent / "cb_bw_summary.csv"
DOUBLE_CSV = OUTPUTS / "double_flagged_companies.csv"
try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=pathlib.Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[1] / "01_Data" / "processed"
BENEISH_PARQUET = _PROCESSED_DIR / "beneish_scores.parquet"
OUTPUT_CSV = OUTPUTS / "label_coverage.csv"

THRESHOLD = -1.78


def load_beneish():
    df = pd.read_parquet(BENEISH_PARQUET, columns=["corp_code", "year", "m_score"])
    df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)
    df["m_score"] = pd.to_numeric(df["m_score"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def main():
    # --- Load labels (fraud=1 only) ---
    labels = pd.read_csv(LABELS_CSV, dtype={"corp_code": str})
    labels["corp_code"] = labels["corp_code"].str.zfill(8)
    # column is fraud_label in this project
    fraud1 = labels[labels["fraud_label"] == 1].copy()
    print(f"fraud=1 labels: {len(fraud1)}")

    # --- Beneish screen ---
    beneish = load_beneish()
    beneish_flagged = (
        beneish[beneish["m_score"] > THRESHOLD]
        .groupby("corp_code")["m_score"]
        .max()
        .reset_index()
        .rename(columns={"m_score": "peak_m_score"})
    )
    fraud1 = fraud1.merge(beneish_flagged, on="corp_code", how="left")
    fraud1["in_beneish_screen"] = fraud1["peak_m_score"].notna()

    # --- CB/BW screen (flag_count >= 1) ---
    if CB_BW_CSV.exists():
        cb = pd.read_csv(CB_BW_CSV, dtype={"corp_code": str})
        cb_max = (
            cb.groupby("corp_code")["flag_count"]
            .max()
            .reset_index()
            .rename(columns={"flag_count": "max_flag_count"})
        )
        fraud1 = fraud1.merge(cb_max, on="corp_code", how="left")
        fraud1["max_flag_count"] = fraud1["max_flag_count"].fillna(0).astype(int)
        fraud1["in_cb_screen"] = fraud1["max_flag_count"] >= 1
    else:
        fraud1["max_flag_count"] = np.nan
        fraud1["in_cb_screen"] = False
        print(f"WARNING: {CB_BW_CSV} not found")

    # --- Cross-screen (PC3 + CB/BW dual screen) ---
    if DOUBLE_CSV.exists():
        double = pd.read_csv(DOUBLE_CSV, dtype={"corp_code": str})
        # Best PC3 rank per company
        pc3_best = (
            double.groupby("corp_code")["PC3_pct_rank"]
            .max()
            .reset_index()
            .rename(columns={"PC3_pct_rank": "best_pc3_rank"})
        )
        fraud1 = fraud1.merge(pc3_best, on="corp_code", how="left")
        fraud1["in_dual_screen"] = fraud1["corp_code"].isin(double["corp_code"])
    else:
        fraud1["best_pc3_rank"] = np.nan
        fraud1["in_dual_screen"] = False
        print(f"WARNING: {DOUBLE_CSV} not found")

    # --- Summary table ---
    cols = [
        "corp_code", "company_name",
        "in_beneish_screen", "peak_m_score",
        "in_cb_screen", "max_flag_count",
        "in_dual_screen", "best_pc3_rank",
    ]
    result = fraud1[cols].copy()
    result["peak_m_score"] = result["peak_m_score"].round(2)
    result["best_pc3_rank"] = result["best_pc3_rank"].round(4)

    print("\n=== Label Coverage Analysis ===\n")
    print(result.to_string(index=False))

    # --- Coverage summary ---
    n = len(result)
    print(f"\n--- Coverage Summary ({n} fraud=1 labels) ---")
    print(f"  in_beneish_screen : {result['in_beneish_screen'].sum()}/{n} ({result['in_beneish_screen'].mean():.0%})")
    print(f"  in_cb_screen      : {result['in_cb_screen'].sum()}/{n} ({result['in_cb_screen'].mean():.0%})")
    print(f"  in_dual_screen    : {result['in_dual_screen'].sum()}/{n} ({result['in_dual_screen'].mean():.0%})")

    # --- Blind spot breakdown ---
    missed_beneish = result[~result["in_beneish_screen"]]
    missed_cb = result[~result["in_cb_screen"]]
    missed_both = result[~result["in_beneish_screen"] & ~result["in_cb_screen"]]

    if not missed_beneish.empty:
        print(f"\nMissed by Beneish screen ({len(missed_beneish)}):")
        for _, r in missed_beneish.iterrows():
            print(f"  {r['corp_code']} {r['company_name']}")

    if not missed_cb.empty:
        print(f"\nMissed by CB/BW screen ({len(missed_cb)}):")
        for _, r in missed_cb.iterrows():
            print(f"  {r['corp_code']} {r['company_name']}  (max_flag_count={r['max_flag_count']})")

    if not missed_both.empty:
        print(f"\nMissed by BOTH screens ({len(missed_both)}) — pipeline blind spots:")
        for _, r in missed_both.iterrows():
            print(f"  {r['corp_code']} {r['company_name']}")

    # --- Save ---
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
