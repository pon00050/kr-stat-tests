"""
fdr_disclosure_leakage.py — Statistical Test S11 (proper FDR design)

Correct version of the FDR timing anomaly test. Fixes KI-021.

Instead of using pre-filtered timing_anomalies.csv (only extreme >=5% events),
this script uses ALL disclosure events for both test and null:
  - Test:  flagged company disclosures (8 corps in disclosures.parquet)
  - Null:  control company disclosures (50 unflagged corps in disclosures.parquet)

Both are joined with price_volume to compute price_change_pct per disclosure date.
Empirical p-values: for each flagged event, p = fraction of control events with
|price_change| >= |this event's price_change|. BH correction at q=0.05.

Note: price_volume.parquet covers only +-60-day windows around CB/BW events.
Inner join coverage will be partial. Coverage rate is printed at runtime.

Run from project root:
    python 03_Analysis/statistical_tests/fdr_disclosure_leakage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.stats.multitest import multipletests

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
PROC   = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

FLAGGED_CORPS = {
    "01207761", "01051092", "00530413", "01049167", "00619640",
    "01107665", "01170962", "01139266",
}
BH_ALPHA = 0.05


def bonferroni_compare(pvals: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Compare Bonferroni (FWER) and Benjamini-Hochberg (FDR) rejection decisions.

    ISL §13.3 (Bonferroni) controls P(≥1 false rejection) ≤ alpha.
    ISL §13.4 (BH) controls E[false discovery rate] ≤ alpha.
    BH is guaranteed to reject at least as many hypotheses as Bonferroni
    at the same alpha level.

    Parameters
    ----------
    pvals : np.ndarray — raw p-values, one per test.
    alpha : float — family-wise / FDR threshold (default 0.05).

    Returns
    -------
    pd.DataFrame with columns ['p_value', 'bonferroni_reject', 'bh_reject',
    'agreement'], sorted by p_value ascending. Pure function.
    """
    from statsmodels.stats.multitest import multipletests
    m = len(pvals)
    bonf_reject = pvals <= (alpha / m) if m > 0 else np.zeros(m, dtype=bool)
    bh_reject, _, _, _ = multipletests(pvals, method="fdr_bh", alpha=alpha)
    df = pd.DataFrame({
        "p_value": pvals,
        "bonferroni_reject": bonf_reject,
        "bh_reject": bh_reject.astype(bool),
    })
    df["agreement"] = df["bonferroni_reject"] == df["bh_reject"]
    return df.sort_values("p_value").reset_index(drop=True)


def _load_and_join(
    corp_codes: set,
    disc: pd.DataFrame,
    ctm_dedup: pd.DataFrame,
    pv_aug: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Filter disclosures to corp_codes, join with augmented price_volume,
    compute price_change_pct and volume_ratio per disclosure date.

    Returns DataFrame with columns: corp_code, rcept_no, filed_at, date, ticker,
    close, prior_close, price_change_pct, volume_ratio.
    """
    sub = disc[disc["corp_code"].isin(corp_codes)].copy()
    sub["date"] = pd.to_datetime(sub["filed_at"], format="%Y%m%d", errors="coerce")
    sub = sub.dropna(subset=["date"])
    sub = sub.merge(ctm_dedup, on="corp_code", how="left").dropna(subset=["ticker"])

    joined = sub.merge(
        pv_aug[["ticker", "date", "close", "volume", "prior_close", "vol_20d_avg"]],
        on=["ticker", "date"],
        how="inner",
    ).dropna(subset=["prior_close"])

    joined = joined.copy()
    joined["price_change_pct"] = (joined["close"] / joined["prior_close"] - 1) * 100
    joined["volume_ratio"] = (
        joined["volume"] / joined["vol_20d_avg"].replace(0, float("nan"))
    )

    n_disc = len(sub)
    n_joined = len(joined)
    coverage = n_joined / n_disc * 100 if n_disc > 0 else 0.0
    print(
        f"{label}: {n_disc:,} disclosures -> {n_joined:,} with price data "
        f"({coverage:.1f}% coverage, {joined['corp_code'].nunique()} companies)"
    )
    return joined


def main() -> None:
    disc = pd.read_parquet(PROC / "disclosures.parquet")
    ctm  = pd.read_parquet(PROC / "corp_ticker_map.parquet")
    pv   = pd.read_parquet(PROC / "price_volume.parquet")

    disc["corp_code"] = disc["corp_code"].astype(str).str.zfill(8)
    ctm["corp_code"]  = ctm["corp_code"].astype(str).str.zfill(8)

    # corp_ticker_map: effective_from/to are all None — simple dedup join on corp_code
    ctm_dedup = ctm[["corp_code", "ticker"]].drop_duplicates("corp_code")

    # Pre-compute lagged close and 20-day trailing average volume per ticker
    pv["date"] = pd.to_datetime(pv["date"])
    pv_aug = pv.sort_values(["ticker", "date"]).copy()
    pv_aug["prior_close"] = pv_aug.groupby("ticker")["close"].shift(1)
    pv_aug["vol_20d_avg"] = (
        pv_aug.groupby("ticker")["volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )

    control_corps = set(disc["corp_code"].unique()) - FLAGGED_CORPS
    print(f"Flagged corps: {len(FLAGGED_CORPS)}  |  Control corps: {len(control_corps)}")

    flagged_df = _load_and_join(FLAGGED_CORPS, disc, ctm_dedup, pv_aug, "Flagged")
    control_df = _load_and_join(control_corps, disc, ctm_dedup, pv_aug, "Control")

    if len(flagged_df) == 0:
        print("ERROR: No flagged events joined. Check price_volume coverage.")
        return
    if len(control_df) < 10:
        print("ERROR: Control null too small (<10 events). Cannot run test.")
        return

    ctrl_abs = control_df["price_change_pct"].abs().values
    test_abs  = flagged_df["price_change_pct"].abs().values

    print(f"\nTest events:   {len(test_abs):,}")
    print(f"Control null:  {len(ctrl_abs):,} events (unfiltered)")

    # Empirical p-value: fraction of control events with |price_change| >= this test event
    pvals = np.array([
        max((ctrl_abs >= v).sum() / len(ctrl_abs), 1 / len(ctrl_abs))
        for v in test_abs
    ])

    # Benjamini-Hochberg correction
    reject, qvals, _, _ = multipletests(pvals, method="fdr_bh", alpha=BH_ALPHA)

    flagged_df = flagged_df.copy()
    flagged_df["p_value"]    = pvals
    flagged_df["q_value"]    = qvals
    flagged_df["fdr_flagged"] = reject

    # ── Summary ───────────────────────────────────────────────────────────────
    n_survivors = int(reject.sum())
    n_test      = len(test_abs)
    print(f"\nBH survivors (q<={BH_ALPHA}): {n_survivors:,} of {n_test:,} "
          f"({n_survivors / n_test * 100:.1f}%)")

    # P-value distribution (hockey-stick shape near 0 = genuine signal)
    bins     = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist, _  = np.histogram(pvals, bins=bins)
    expected = [n_test * (bins[i+1] - bins[i]) for i in range(len(bins) - 1)]
    print("\nP-value distribution (enrichment near 0 = pre-disclosure signal):")
    for lo, hi, cnt, exp in zip(bins[:-1], bins[1:], hist, expected):
        print(f"  [{lo:.2f}, {hi:.2f}): {cnt:5,}  (expected under null: {exp:.0f})")

    # Top survivors
    survivors = flagged_df[flagged_df["fdr_flagged"]].sort_values("p_value")
    if n_survivors > 0:
        print(f"\nTop 10 FDR-flagged events:")
        cols = ["corp_code", "filed_at", "price_change_pct", "p_value", "q_value"]
        print(survivors.head(10)[cols].to_string(index=False))

    # Companies with multiple survivors
    if n_survivors > 0:
        repeat      = flagged_df[flagged_df["fdr_flagged"]].groupby("corp_code").size()
        repeat_multi = repeat[repeat > 1].sort_values(ascending=False)
        if len(repeat_multi) > 0:
            print(f"\nRepeat offenders ({len(repeat_multi)} companies, >1 FDR event):")
            print(repeat_multi.head(10).to_string())

    # ── Bonferroni vs BH comparison (ISL 13.3 vs 13.4) ───────────────────────
    cmp_df = bonferroni_compare(pvals, alpha=BH_ALPHA)
    bonf_n = int(cmp_df["bonferroni_reject"].sum())
    bh_n   = int(cmp_df["bh_reject"].sum())
    agree  = int(cmp_df["agreement"].sum())
    print(f"\nISL 13.3 vs 13.4 comparison (alpha={BH_ALPHA}):")
    print(f"  Bonferroni (FWER): {bonf_n:,} rejections")
    print(f"  BH (FDR):          {bh_n:,} rejections")
    print(f"  Agreement:         {agree:,}/{len(pvals):,} tests ({agree/len(pvals)*100:.1f}%)")
    cmp_df.to_csv(OUTPUT / "fdr_bonferroni_compare.csv", index=False)
    print(f"Saved: {OUTPUT / 'fdr_bonferroni_compare.csv'}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = OUTPUT / "fdr_disclosure_leakage.csv"
    flagged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    # ── P-value histogram ─────────────────────────────────────────────────────
    fig = px.histogram(
        pd.DataFrame({"p_value": pvals}),
        x="p_value",
        nbins=50,
        title=(
            "P-value distribution - disclosure leakage FDR test (S11)<br>"
            "<sup>Hockey-stick near 0 = systematic pre-disclosure movement in flagged companies</sup>"
        ),
        labels={"p_value": "Empirical p-value"},
        color_discrete_sequence=["#4a90d9"],
        template="plotly_white",
    )
    fig.add_vline(
        x=BH_ALPHA,
        line_dash="dash",
        line_color="red",
        annotation_text=f"alpha={BH_ALPHA} BH threshold",
    )
    fig.update_layout(height=420, margin=dict(t=60, b=30))
    png_path = OUTPUT / "fdr_disclosure_leakage_hist.png"
    fig.write_image(str(png_path))
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
