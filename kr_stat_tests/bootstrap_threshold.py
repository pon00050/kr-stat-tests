"""
bootstrap_threshold.py — Statistical Test 7 (ISL Ch. 5)

Bootstraps the optimal Beneish M-score threshold for KOSDAQ-labeled data.
Compares the KOSDAQ-calibrated threshold + 95% CI to US-derived -1.78.

Requires labels.csv with ≥10 rows total (including auto-populated controls).

Input:  01_Data/processed/beneish_scores.parquet
        labels.csv (in same directory)
Output: outputs/bootstrap_threshold.csv
        outputs/bootstrap_threshold.png

Run from project root:
    python 03_Analysis/statistical_tests/bootstrap_threshold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import plotly.graph_objects as go

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
HERE = Path(__file__).parent
OUTPUT = HERE / "outputs"
OUTPUT.mkdir(exist_ok=True)

N_BOOTSTRAP = 2000
THRESHOLD_GRID = np.round(np.arange(-3.5, 0.05, 0.05), 2)
BENEISH_1999 = -1.78
MIN_LABELED_TOTAL = 10


def auto_controls(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Select clean controls using external criteria only — not M-score."""
    beneish = pd.read_parquet(DATA / "beneish_scores.parquet")
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)

    cb_bw = pd.read_parquet(DATA / "cb_bw_events.parquet")
    cb_bw["corp_code"] = cb_bw["corp_code"].astype(str).str.zfill(8)
    cb_corps = set(cb_bw["corp_code"])

    existing = set(labels_df["corp_code"])
    clean = (
        beneish[
            ~beneish["corp_code"].isin(existing)      # not already labeled
            & ~beneish["corp_code"].isin(cb_corps)    # no CB/BW activity (external criterion)
        ]
        .groupby("corp_code")
        .filter(lambda g: g["m_score"].notna().sum() >= 3)  # at least 3 scoreable years
        .groupby("corp_code")
        .first()
        .reset_index()
        .sort_values("corp_code")                     # neutral sort — NOT by m_score
        [["corp_code", "company_name"]]
        .head(20)
    )
    clean["fraud_label"] = 0
    return clean


def cluster_bootstrap_sample(data: pd.DataFrame, rng) -> pd.DataFrame:
    """Resample at company level, keeping all year-rows for each sampled company."""
    groups = {c: grp for c, grp in data.groupby("corp_code")}
    unique_corps = list(groups.keys())
    sampled_corps = rng.choice(unique_corps, size=len(unique_corps), replace=True)
    return pd.concat([groups[c] for c in sampled_corps], ignore_index=True)


def load_labeled_data(all_labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels with beneish_scores; one row per (corp_code, year)."""
    beneish = pd.read_parquet(DATA / "beneish_scores.parquet")
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    joined = beneish.merge(all_labels[["corp_code", "fraud_label"]], on="corp_code", how="inner")
    return joined[joined["m_score"].notna()][["corp_code", "year", "m_score", "fraud_label"]]


def optimal_threshold(m_scores: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_f1 = BENEISH_1999, 0.0
    for t in THRESHOLD_GRID:
        preds = (m_scores > t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main() -> None:
    labels_path = HERE / "labels.csv"
    if not labels_path.exists():
        print(f"ERROR: labels.csv not found at {labels_path}")
        return

    labels_df = pd.read_csv(labels_path, dtype={"corp_code": str})
    labels_df["corp_code"] = labels_df["corp_code"].str.zfill(8)

    controls = auto_controls(labels_df)
    all_labels = pd.concat([labels_df, controls], ignore_index=True).drop_duplicates("corp_code")
    n_fraud = int((all_labels["fraud_label"] == 1).sum())
    n_clean = int((all_labels["fraud_label"] == 0).sum())
    print(f"Labeled set: {n_fraud} fraud=1, {n_clean} fraud=0")

    if len(all_labels) < MIN_LABELED_TOTAL:
        print(f"Only {len(all_labels)} labeled samples (minimum {MIN_LABELED_TOTAL}). Exiting.")
        return

    data = load_labeled_data(all_labels)
    if len(data) == 0:
        print("No matching rows in beneish_scores for labeled companies. Exiting.")
        return

    print(f"Labeled company-years: {len(data):,}")

    rng = np.random.default_rng(42)
    scores_arr = data["m_score"].values
    labels_arr = data["fraud_label"].values

    opt_thresholds = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = cluster_bootstrap_sample(data, rng)
        opt_thresholds[i] = optimal_threshold(
            sample["m_score"].values, sample["fraud_label"].values
        )

    lo, hi = np.percentile(opt_thresholds, [2.5, 97.5])
    median_t = float(np.median(opt_thresholds))

    print(f"\nBootstrap results (n={N_BOOTSTRAP}):")
    print(f"  Median optimal threshold: {median_t:.2f}")
    print(f"  95% CI:                   [{lo:.2f}, {hi:.2f}]")
    print(f"  Beneish 1999 (US):        {BENEISH_1999}")
    inside = lo <= BENEISH_1999 <= hi
    print(f"  US threshold {'inside' if inside else 'OUTSIDE'} KOSDAQ 95% CI")

    # F1 curve across full labeled data (not bootstrapped — for visualization)
    full_f1_rows = []
    for t in THRESHOLD_GRID:
        preds = (scores_arr > t).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0) if preds.sum() > 0 else 0.0
        full_f1_rows.append({"threshold": float(t), "f1_score": round(f1, 4)})

    out_df = pd.DataFrame(full_f1_rows)
    out_df["bootstrap_median"] = median_t
    out_df["ci_lower"] = round(lo, 2)
    out_df["ci_upper"] = round(hi, 2)
    out_df.to_csv(OUTPUT / "bootstrap_threshold.csv", index=False)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=out_df["threshold"], y=out_df["f1_score"],
        mode="lines", name="F1 (full labeled set)",
        line=dict(color="#4a90d9", width=2),
    ))
    fig.add_vrect(
        x0=lo, x1=hi,
        fillcolor="rgba(74,144,217,0.12)", line_width=0,
        annotation_text=f"95% CI [{lo:.2f}, {hi:.2f}]",
        annotation_position="top left",
    )
    fig.add_vline(
        x=BENEISH_1999, line_dash="dash", line_color="red",
        annotation_text=f"Beneish 1999: {BENEISH_1999}",
    )
    fig.add_vline(
        x=median_t, line_dash="dot", line_color="#2ecc71",
        annotation_text=f"KOSDAQ median: {median_t:.2f}",
    )
    fig.update_layout(
        title=(
            f"Bootstrapped Optimal M-Score Threshold (n={N_BOOTSTRAP} draws)<br>"
            f"<sup>{n_fraud} fraud=1, {n_clean} fraud=0 labeled companies</sup>"
        ),
        xaxis_title="M-Score Threshold",
        yaxis_title="F1 Score",
        template="plotly_white",
        height=460,
        margin=dict(t=80, b=40),
    )
    fig.write_image(str(OUTPUT / "bootstrap_threshold.png"))

    print(f"\nSaved: {OUTPUT / 'bootstrap_threshold.csv'}")
    print(f"Saved: {OUTPUT / 'bootstrap_threshold.png'}")


if __name__ == "__main__":
    main()
