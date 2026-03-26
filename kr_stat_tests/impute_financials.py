"""
impute_financials.py — Statistical Test 5 (ISL Ch. 12)

MICE (Iterative Imputer) fills missing financial fields without dropping rows.
Preserves small-cap companies that report subsets of financial fields.
Log-transforms positive-only columns before imputing; inverse-transforms after.

Input:  01_Data/processed/company_financials.parquet
Output: outputs/company_financials_imputed.parquet

Run from project root:
    python 03_Analysis/statistical_tests/impute_financials.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Positive-only columns: log-transformed before imputing, inverse after
LOG_COLS = ["receivables", "revenue", "ppe", "total_assets"]
# May be negative: imputed on raw scale
RAW_COLS = ["cogs", "sga", "depreciation", "lt_debt", "net_income", "cfo"]
ALL_COLS = LOG_COLS + RAW_COLS


def null_summary(df: pd.DataFrame, mask: pd.Series, label: str) -> None:
    print(f"\nNull counts {label} (balance-sheet rows only):")
    for col in ALL_COLS:
        n_null = df.loc[mask, col].isna().sum()
        total = mask.sum()
        pct = n_null / total * 100 if total > 0 else 0
        print(f"  {col:<15}: {n_null:>6,} / {total:,}  ({pct:.1f}%)")


def main() -> None:
    path = DATA / "company_financials.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")

    # Only impute rows that have a balance sheet (total_assets not null)
    has_bs = df["total_assets"].notna()
    print(f"Rows with total_assets (imputation scope): {has_bs.sum():,}")

    null_summary(df, has_bs, "BEFORE imputation")

    df_imp = df.copy()

    # Build imputation matrix for balance-sheet rows
    X = df.loc[has_bs, ALL_COLS].copy()

    # Log-transform positive columns; clip to 0 to avoid log(negative)
    log_transformed: dict[str, np.ndarray] = {}
    for col in LOG_COLS:
        original = X[col].values
        X[col] = np.log1p(np.clip(X[col].values, 0, None))
        log_transformed[col] = original  # store for verification

    imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)
    X_imputed = imputer.fit_transform(X.values)
    X_imputed_df = pd.DataFrame(X_imputed, columns=ALL_COLS, index=X.index)

    # Inverse log-transform
    for col in LOG_COLS:
        X_imputed_df[col] = np.expm1(X_imputed_df[col])

    # Write imputed values back
    for col in ALL_COLS:
        df_imp.loc[has_bs, col] = X_imputed_df[col].values

    null_summary(df_imp, has_bs, "AFTER imputation")

    # Verify: no NaN left in imputed scope
    remaining_nulls = df_imp.loc[has_bs, ALL_COLS].isna().sum().sum()
    if remaining_nulls > 0:
        print(f"\nWARNING: {remaining_nulls} NaN values remain after imputation.")
    else:
        print(f"\nVerification: 0 NaN values in imputed columns for balance-sheet rows.")

    out = OUTPUT / "company_financials_imputed.parquet"
    df_imp.to_parquet(out, index=False, engine="pyarrow")
    print(f"\nSaved: {out} ({len(df_imp):,} rows)")


if __name__ == "__main__":
    main()
