"""
S9 — Cross-Screen: PC3 Top-Decile × Flagged CB/BW Events
=========================================================
Joins pca_pc3_scores.csv (PC3 percentile rank >= 0.90) against cb_bw_summary.csv
(flag_count >= 1) on corp_code to identify companies doubly flagged by both screens.

Output: outputs/double_flagged_companies.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
OUTPUT = ROOT / "03_Analysis" / "statistical_tests" / "outputs"
ANALYSIS = ROOT / "03_Analysis"

TIER1 = {"01051092", "01207761", "00530413", "01049167", "00619640"}

PC3_DECILE_THRESHOLD = 0.90   # top decile (531 company-years)


def main() -> None:
    # ── 1. Load PC3 scores ────────────────────────────────────────────────────
    pc3_path = OUTPUT / "pca_pc3_scores.csv"
    if not pc3_path.exists():
        print(f"ERROR: {pc3_path} not found. Run pca_beneish.py first.")
        sys.exit(1)

    pc3 = pd.read_csv(pc3_path)
    pc3["corp_code"] = pc3["corp_code"].astype(str).str.zfill(8)
    pc3_top = pc3[pc3["PC3_pct_rank"] >= PC3_DECILE_THRESHOLD].copy()
    print(f"PC3 top-decile company-years (rank >= {PC3_DECILE_THRESHOLD}): {len(pc3_top):,}")

    # ── 2. Load CB/BW summary ─────────────────────────────────────────────────
    cb_path = ANALYSIS / "cb_bw_summary.csv"
    if not cb_path.exists():
        print(f"ERROR: {cb_path} not found. Run run_cb_bw_timelines.py first.")
        sys.exit(1)

    cb = pd.read_csv(cb_path, low_memory=False)
    cb["corp_code"] = cb["corp_code"].astype(str).str.zfill(8)
    cb_flagged = cb[cb["flag_count"] >= 1].copy()
    print(f"CB/BW flagged events (flag_count >= 1): {len(cb_flagged):,} rows, "
          f"{cb_flagged['corp_code'].nunique():,} unique companies")

    # ── 3. Aggregate CB/BW to one row per company ─────────────────────────────
    def union_flags(series: pd.Series) -> str:
        seen: set[str] = set()
        for row in series:
            for f in str(row).split(","):
                f = f.strip()
                if f:
                    seen.add(f)
        return "; ".join(sorted(seen))

    cb_agg = (
        cb_flagged.groupby("corp_code")
        .agg(
            max_flag_count=("flag_count", "max"),
            event_count=("issue_date", "count"),
            flags=("flags", union_flags),
            repricing_flag=("repricing_flag", "any"),
            volume_flag=("volume_flag", "any"),
            holdings_flag=("holdings_flag", "any"),
            exercise_cluster_flag=("exercise_cluster_flag", "any"),
        )
        .reset_index()
    )

    # ── 4. Inner join on corp_code ─────────────────────────────────────────────
    double = pc3_top.merge(cb_agg, on="corp_code", how="inner")
    double["is_tier1"] = double["corp_code"].isin(TIER1)
    double = double.sort_values("PC3_pct_rank", ascending=False).reset_index(drop=True)

    secondary = double[~double["is_tier1"]].copy()
    tier1_rows = double[double["is_tier1"]].copy()

    # ── 5. Save output ────────────────────────────────────────────────────────
    out_cols = [
        "corp_code", "company_name", "year", "m_score",
        "PC3", "PC3_pct_rank", "pc3_flagged",
        "max_flag_count", "event_count", "flags",
        "repricing_flag", "volume_flag", "holdings_flag", "exercise_cluster_flag",
        "is_tier1",
    ]
    double[out_cols].to_csv(OUTPUT / "double_flagged_companies.csv",
                            index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT / 'double_flagged_companies.csv'}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\nDouble-flagged company-years: {len(double):,}")
    print(f"  Secondary targets: {len(secondary):,}")
    print(f"  Tier 1 leads:      {len(tier1_rows):,}")

    unique_secondary = secondary["corp_code"].nunique()
    print(f"\nUnique secondary companies: {unique_secondary}")

    # Flag type breakdown among secondary targets
    print("\nFlag type breakdown (secondary targets):")
    for col in ("volume_flag", "repricing_flag", "holdings_flag", "exercise_cluster_flag"):
        n = secondary[col].sum()
        print(f"  {col}: {n} ({n / max(unique_secondary, 1):.0%} of unique secondary companies)")

    # Print top 20 secondary targets (ascii-safe)
    print(f"\nTop 20 secondary targets by PC3 rank:")
    print(f"{'corp_code':>10}  {'year':>4}  {'PC3_rank':>8}  {'flag_cnt':>8}  flags")
    for _, row in secondary.head(20).iterrows():
        print(f"  {row['corp_code']:>8}  {int(row['year']):>4}  {row['PC3_pct_rank']:>8.4f}"
              f"  {int(row['max_flag_count']):>8}  {row['flags']}")

    # High-priority secondaries (PC3 >= 0.95, flag_count >= 2) — candidates for S8 extension
    high_priority = secondary[
        (secondary["PC3_pct_rank"] >= 0.95) & (secondary["max_flag_count"] >= 2)
    ].drop_duplicates("corp_code")
    print(f"\nHigh-priority secondaries (PC3 >= 95th pct AND flag_count >= 2): "
          f"{len(high_priority)} unique companies")
    if len(high_priority) > 0:
        print("  corp_codes for S8 --corp-codes extension:")
        print(" ", ",".join(high_priority["corp_code"].unique().tolist()))
    else:
        print("  None — run S8 at default scope (Tier 1 leads only)")


if __name__ == "__main__":
    main()
