"""
fdr_timing_anomalies.py — Statistical Test 1 (ISL Ch. 13)

Applies Benjamini-Hochberg FDR correction to timing anomaly flags.
Converts 168 binary flags into q-values at q=0.05.

Input:  03_Analysis/timing_anomalies.csv (687 rows)
Output: outputs/fdr_timing_anomalies.csv
        outputs/fdr_pvalue_hist.png

Run from project root:
    python 03_Analysis/statistical_tests/fdr_timing_anomalies.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import plotly.express as px

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
ANALYSIS = ROOT / "03_Analysis"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)


def pi0_estimate(pvals: np.ndarray, lambda_: float = 0.5) -> float:
    """Storey (2002) π₀ estimator — proportion of true nulls among m tests.

    ISL §13.4: under the null, p-values are uniform on [0, 1], so the
    density above lambda is 1/(1-lambda) per unit. True alternatives
    concentrate near 0 and are absent from [lambda, 1].

        π₀ ≈ #{p > lambda} / (m × (1 − lambda))

    Clipped to [0.0, 1.0].  Pure function — no I/O or sklearn calls.

    Parameters
    ----------
    pvals : np.ndarray — raw (uncorrected) p-values in [0, 1].
    lambda_ : float — threshold for null estimation (default 0.5).

    Returns
    -------
    float in [0.0, 1.0].
    """
    m = len(pvals)
    if m == 0:
        return 1.0
    n_above = (pvals > lambda_).sum()
    pi0 = n_above / (m * (1.0 - lambda_))
    return float(min(pi0, 1.0))


def main() -> None:
    csv_path = ANALYSIS / "timing_anomalies.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_timing_anomalies.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} timing anomaly events")

    # Ensure corp_code is 8-digit string
    df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)

    price_change = pd.to_numeric(df["price_change_pct"], errors="coerce").fillna(0)
    abs_changes = price_change.abs().values
    n = len(abs_changes)

    # ── Control-group null (S10b) ─────────────────────────────────────────────
    # Build null from disclosures of 50 unflagged control companies.
    # These companies appear in cb_bw_summary with flag_count=0; their price data
    # already exists in price_volume.parquet (±60-day windows around CB/BW events).
    # Join path: disclosures.corp_code → corp_ticker_map.ticker → price_volume.(ticker, date)

    FLAGGED_CORPS = {
        "01207761", "01051092", "00530413", "01049167", "00619640",
        "01107665", "01170962", "01139266",
    }

    ctrl_null = None  # will be populated below; fallback to full dist if join fails

    try:
        disc = pd.read_parquet(ROOT / "01_Data/processed/disclosures.parquet")
        ctm  = pd.read_parquet(ROOT / "01_Data/processed/corp_ticker_map.parquet")
        pv   = pd.read_parquet(ROOT / "01_Data/processed/price_volume.parquet")

        # Keep control companies only
        ctrl_disc = disc[~disc["corp_code"].isin(FLAGGED_CORPS)].copy()
        ctrl_disc["date"] = pd.to_datetime(ctrl_disc["filed_at"], format="%Y%m%d", errors="coerce")
        ctrl_disc = ctrl_disc.dropna(subset=["date"])

        # Join corp_code → ticker (effective_from/to are None for all rows — always valid)
        ctm_dedup = ctm[["corp_code", "ticker"]].drop_duplicates("corp_code")
        ctrl_disc = ctrl_disc.merge(ctm_dedup, on="corp_code", how="left")
        ctrl_disc = ctrl_disc.dropna(subset=["ticker"])

        # Pre-compute lagged close and 20-day avg volume per ticker in price_volume
        pv["date"] = pd.to_datetime(pv["date"])
        pv_aug = pv.sort_values(["ticker", "date"]).copy()
        pv_aug["prior_close"] = pv_aug.groupby("ticker")["close"].shift(1)
        pv_aug["vol_20d_avg"] = (
            pv_aug.groupby("ticker")["volume"]
            .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        )

        # Join control disclosures with augmented price_volume on (ticker, date)
        ctrl_joined = ctrl_disc.merge(
            pv_aug[["ticker", "date", "close", "volume", "prior_close", "vol_20d_avg"]],
            on=["ticker", "date"],
            how="inner",
        )
        ctrl_joined = ctrl_joined.dropna(subset=["prior_close"])
        ctrl_joined["ctrl_price_change_pct"] = (
            ctrl_joined["close"] / ctrl_joined["prior_close"] - 1
        ) * 100
        ctrl_joined["ctrl_volume_ratio"] = (
            ctrl_joined["volume"] / ctrl_joined["vol_20d_avg"].replace(0, float("nan"))
        )

        # Use ALL control events (unfiltered) as the empirical null.
        # Filtering to "quiet" events (|price| < 1%) produces a null whose max is ~1%,
        # while all test events have |price| ≥ 5% → zero overlap → every p-value
        # floors at 1/N. The correct null is the full distribution of price moves
        # across control disclosures, which includes volatile days and spans the test range.
        ctrl_null_arr = ctrl_joined["ctrl_price_change_pct"].abs().values
        print(f"Control null: {len(ctrl_null_arr):,} events from "
              f"{ctrl_joined['corp_code'].nunique()} companies (unfiltered)")

        if len(ctrl_null_arr) >= 10:
            ctrl_null = ctrl_null_arr
        else:
            print("WARNING: Control null too small (<10). Falling back to full distribution.")

    except Exception as exc:
        print(f"WARNING: Control null build failed ({exc}). Falling back to full distribution.")

    if ctrl_null is None:
        ctrl_null = abs_changes
        print(f"Null: full distribution ({len(ctrl_null):,} events — contaminated)")
    # ── end control-group null ─────────────────────────────────────────────────

    # Empirical p-value: fraction of control null with |price_change| >= this event
    pvals = np.array([
        max((ctrl_null >= v).sum() / len(ctrl_null), 1 / len(ctrl_null))
        for v in abs_changes
    ])

    # Benjamini-Hochberg correction
    reject, qvals, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.05)

    df["p_value"] = pvals
    df["q_value"] = qvals
    df["fdr_flagged"] = reject

    # Storey π₀ estimate (ISL §13.4)
    pi0 = pi0_estimate(pvals)
    expected_fd = pi0 * len(pvals) * 0.05
    summary_df = pd.DataFrame([{
        "n_tests": len(pvals),
        "n_rejected_bh": int(reject.sum()),
        "pi0_est": round(pi0, 4),
        "expected_false_disc": round(expected_fd, 1),
    }])
    print(f"\nStorey pi0 estimate: {pi0:.4f}  (proportion of true nulls)")
    print(f"Expected false discoveries at alpha=0.05: {expected_fd:.1f}")
    summary_df.to_csv(OUTPUT / "fdr_timing_summary.csv", index=False)
    print(f"Saved: {OUTPUT / 'fdr_timing_summary.csv'}")

    # Summary
    original_flags = int(df["flag"].sum()) if "flag" in df.columns else int((price_change.abs() >= 5).sum())
    fdr_flags = int(reject.sum())
    print(f"\nOriginal binary flags:  {original_flags:,}")
    print(f"FDR-corrected (q<=0.05): {fdr_flags:,}")
    if original_flags > 0:
        print(f"Survival rate:          {fdr_flags / original_flags * 100:.1f}%")

    # P-value distribution bins
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist, _ = np.histogram(pvals, bins=bins)
    print("\nP-value distribution:")
    for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
        print(f"  [{lo:.2f}, {hi:.2f}): {cnt:,}")

    # Repeat offenders (companies with multiple FDR-flagged events)
    if fdr_flags > 0:
        repeat = df[df["fdr_flagged"]].groupby("corp_code").size().sort_values(ascending=False)
        repeat_multi = repeat[repeat > 1]
        if len(repeat_multi) > 0:
            print(f"\nRepeat FDR offenders ({len(repeat_multi)} companies with >1 flagged event):")
            print(repeat_multi.head(10).to_string())

    # Save CSV
    out_csv = OUTPUT / "fdr_timing_anomalies.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    # P-value histogram (hockey stick = real signal)
    fig = px.histogram(
        pd.DataFrame({"p_value": pvals}),
        x="p_value",
        nbins=50,
        title=(
            "P-value distribution — timing anomalies<br>"
            "<sup>Hockey-stick shape enriched near 0 = systematic pre-disclosure movement</sup>"
        ),
        labels={"p_value": "Empirical p-value"},
        color_discrete_sequence=["#4a90d9"],
        template="plotly_white",
    )
    fig.add_vline(x=0.05, line_dash="dash", line_color="red",
                  annotation_text="α=0.05 BH threshold")
    fig.update_layout(height=420, margin=dict(t=60, b=30))
    png_path = OUTPUT / "fdr_pvalue_hist.png"
    fig.write_image(str(png_path))
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
