"""
permutation_repricing_peak.py — Statistical Test 2 (ISL Ch. 13)

Permutation test: is CB/BW repricing density near the Jan 2021 KOSDAQ peak
non-random? Shuffles repricing dates 10,000 times and counts near-peak events.

Input:  01_Data/processed/cb_bw_events.parquet
Output: outputs/permutation_repricing.csv
        outputs/permutation_repricing_null_dist.png

Run from project root:
    python 03_Analysis/statistical_tests/permutation_repricing_peak.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

PEAK_DATE = pd.Timestamp("2021-01-06")        # KOSDAQ all-time high
WINDOWS = [30, 60, 90, 180]                   # days to test
N_PERMS = 10_000
DATE_RANGE_START = pd.Timestamp("2019-01-01")
DATE_RANGE_END = pd.Timestamp("2023-12-31")
RNG = np.random.default_rng(42)


def extract_repricing_dates(events_df: pd.DataFrame) -> pd.Series:
    """Parse repricing_history JSON strings; return flat Series of all repricing dates."""
    dates: list[pd.Timestamp] = []
    for hist_str in events_df["repricing_history"]:
        try:
            records = json.loads(hist_str) if isinstance(hist_str, str) else []
        except (json.JSONDecodeError, TypeError):
            records = []
        for rec in records:
            raw = rec.get("date") or rec.get("조정일자") or rec.get("adjustment_date")
            if raw:
                try:
                    dates.append(pd.to_datetime(str(raw)))
                except Exception:
                    pass
    return pd.Series(dates)


def count_near_peak(dates_ns: np.ndarray, window_days: int) -> int:
    """Count dates within window_days of PEAK_DATE."""
    window_ns = window_days * 86_400 * 1_000_000_000
    return int(np.sum(np.abs(dates_ns - PEAK_DATE.value) <= window_ns))


def main() -> None:
    path = DATA / "cb_bw_events.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} CB/BW events")

    repricing_dates = extract_repricing_dates(df).dropna()
    repricing_dates = repricing_dates[
        (repricing_dates >= DATE_RANGE_START) & (repricing_dates <= DATE_RANGE_END)
    ]
    n_dates = len(repricing_dates)
    print(f"Extracted {n_dates:,} repricing dates in "
          f"{DATE_RANGE_START.year}–{DATE_RANGE_END.year}")

    if n_dates == 0:
        print("No repricing dates found — SEIBRO data may not be loaded yet.")
        print("Run extract_seibro_repricing.py first, then re-run this script.")
        return

    dates_ns = repricing_dates.values.astype(np.int64)
    date_range_ns = int((DATE_RANGE_END - DATE_RANGE_START).total_seconds() * 1e9)
    start_ns = DATE_RANGE_START.value

    results = []
    null_dists: dict[int, np.ndarray] = {}

    for window in WINDOWS:
        observed = count_near_peak(dates_ns, window)

        null_counts = np.empty(N_PERMS, dtype=int)
        for i in range(N_PERMS):
            shuffled = start_ns + RNG.integers(0, date_range_ns, size=n_dates)
            null_counts[i] = count_near_peak(shuffled, window)

        p_value = float(np.mean(null_counts >= observed))
        null_dists[window] = null_counts

        results.append({
            "window_days": window,
            "observed_count": observed,
            "null_mean": round(float(null_counts.mean()), 2),
            "null_std": round(float(null_counts.std()), 2),
            "p_value": round(p_value, 4),
            "n_permutations": N_PERMS,
        })
        sig = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else ("*" if p_value < 0.10 else ""))
        print(f"  Window ±{window:3d}d: observed={observed:3d}, "
              f"null_mean={null_counts.mean():.1f} (±{null_counts.std():.1f}), "
              f"p={p_value:.4f} {sig}")

    # Save CSV
    out_csv = OUTPUT / "permutation_repricing.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Plot null distributions (2×2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Window ±{w}d (p={r['p_value']:.3f})" for w, r in zip(WINDOWS, results)],
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), result in zip(positions, results):
        window = result["window_days"]
        null = null_dists[window]
        obs = result["observed_count"]
        fig.add_trace(
            go.Histogram(x=null, name=f"±{window}d null", showlegend=False,
                         marker_color="#4a90d9", opacity=0.75, nbinsx=40),
            row=row, col=col,
        )
        fig.add_vline(
            x=obs, line_dash="dash", line_color="red",
            annotation_text=f"observed={obs}",
            row=row, col=col,
        )

    fig.update_layout(
        title=(
            f"Permutation test: CB/BW repricing near Jan 2021 KOSDAQ peak "
            f"(n={N_PERMS:,} permutations)"
        ),
        height=560,
        template="plotly_white",
    )
    png_path = OUTPUT / "permutation_repricing_null_dist.png"
    fig.write_image(str(png_path))
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
