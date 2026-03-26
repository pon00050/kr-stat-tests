"""
lasso_beneish.py — Statistical Test 9 (ISL Ch. 6)

Lasso regression to identify which Beneish components carry signal in KOSDAQ.
Compares KOSDAQ-fitted weights to Beneish (1999) US-market coefficients.

Requires labels.csv with ≥10 rows total (including auto-populated controls).

Input:  01_Data/processed/beneish_scores.parquet
        labels.csv (in same directory)
Output: outputs/lasso_coefficients.csv

Run from project root:
    python 03_Analysis/statistical_tests/lasso_beneish.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path as sklearn_lasso_path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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

# Beneish (1999) original 8-variable coefficients
BENEISH_1999: dict[str, float] = {
    "dsri":  0.920,
    "gmi":   0.528,
    "aqi":   0.404,
    "sgi":   0.892,
    "depi":  0.115,
    "sgai": -0.172,
    "tata":  4.679,
    "lvgi": -0.327,
}
COMPONENTS = list(BENEISH_1999.keys())
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


def compute_lasso_path(X_scaled: np.ndarray, y: np.ndarray, components: list[str]) -> pd.DataFrame:
    """Compute full Lasso regularization path across 100 alpha values.

    Returns DataFrame with columns: ['alpha'] + components (one row per alpha,
    ordered from strongest to weakest regularization).
    At index 0 (max alpha), all coefficients are ≈ 0.
    As alpha decreases, components enter the model one by one.
    """
    alphas, coefs, _ = sklearn_lasso_path(X_scaled, y, eps=1e-3, n_alphas=100)
    # coefs shape: (n_features, n_alphas) — transpose to (n_alphas, n_features)
    path_df = pd.DataFrame(coefs.T, columns=components)
    path_df.insert(0, "alpha", alphas)
    return path_df


def events_per_variable(n_fraud: int, n_components: int) -> float:
    """Events-per-variable (EPV): n_fraud / n_components. Accepted minimum is 10–15."""
    return n_fraud / n_components if n_components > 0 else 0.0


def load_labeled_data(all_labels: pd.DataFrame) -> pd.DataFrame:
    beneish = pd.read_parquet(DATA / "beneish_scores.parquet")
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    joined = beneish.merge(all_labels[["corp_code", "fraud_label"]], on="corp_code", how="inner")
    return joined[joined["m_score"].notna()][COMPONENTS + ["corp_code", "fraud_label"]]


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
    epv = events_per_variable(n_fraud, len(COMPONENTS))
    print(f"Labeled set: {n_fraud} fraud=1, {n_clean} fraud=0")
    print(f"EPV (events-per-variable): {epv:.1f}  [accepted minimum: 10–15]")
    if epv < 10:
        print(
            f"WARNING: EPV={epv:.1f} is below the accepted minimum of 10–15. "
            "Lasso coefficients are directional only — findings may not generalize."
        )

    if len(all_labels) < MIN_LABELED_TOTAL:
        print(f"Only {len(all_labels)} labeled samples (minimum {MIN_LABELED_TOTAL}). Exiting.")
        return

    data = load_labeled_data(all_labels)
    if len(data) == 0:
        print("No matching beneish rows for labeled companies. Exiting.")
        return

    print(f"Company-years: {len(data):,}")

    X_raw = data[COMPONENTS].copy()
    for col in COMPONENTS:
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    y = data["fraud_label"].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # GroupKFold ensures no company's years appear in both train and validation folds
    corp_codes = data["corp_code"].values
    groups = pd.factorize(corp_codes)[0]
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    # Pre-generate GroupKFold splits so groups don't need to be passed to fit()
    cv_splits = list(gkf.split(X_scaled, y, groups=groups))

    alphas = np.logspace(-4, 1, 50)
    lasso = LassoCV(alphas=alphas, cv=cv_splits, max_iter=5000, random_state=42)
    lasso.fit(X_scaled, y)

    r2 = lasso.score(X_scaled, y)
    print(f"\nOptimal alpha: {lasso.alpha_:.6f}")
    print(f"R² (in-sample): {r2:.4f}")

    coefs = pd.DataFrame({
        "feature": COMPONENTS,
        "lasso_coef": lasso.coef_.round(6),
        "beneish_1999_weight": [BENEISH_1999[c] for c in COMPONENTS],
    })
    coefs["is_active"] = coefs["lasso_coef"].abs() > 1e-6
    coefs["direction_match"] = (
        np.sign(coefs["lasso_coef"]) == np.sign(coefs["beneish_1999_weight"])
    )
    # direction_match is undefined when lasso zeroes out the component
    coefs.loc[~coefs["is_active"], "direction_match"] = pd.NA

    coefs = coefs.sort_values("lasso_coef", key=abs, ascending=False)
    coefs["epv"] = round(epv, 2)

    print(f"\nLasso coefficients vs Beneish 1999:")
    print(coefs[["feature", "lasso_coef", "beneish_1999_weight", "is_active", "direction_match"]]
          .to_string(index=False))

    active = coefs[coefs["is_active"]]
    match_count = active["direction_match"].sum()
    print(f"\nActive features (non-zero Lasso): {list(active['feature'])}")
    if len(active) > 0:
        print(f"Direction matches Beneish 1999: {match_count}/{len(active)}")

    coefs.to_csv(OUTPUT / "lasso_coefficients.csv", index=False)
    print(f"\nSaved: {OUTPUT / 'lasso_coefficients.csv'}")

    # Regularization path — full trajectory across 100 alpha values
    path_df = compute_lasso_path(X_scaled, y, COMPONENTS)
    path_df.to_csv(OUTPUT / "lasso_path.csv", index=False)
    print(f"Saved: {OUTPUT / 'lasso_path.csv'}")

    import plotly.graph_objects as go
    fig = go.Figure()
    for comp in COMPONENTS:
        fig.add_trace(go.Scatter(
            x=-np.log(path_df["alpha"]),
            y=path_df[comp],
            mode="lines",
            name=comp,
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=(
            "Lasso Regularization Path — 8 Beneish Components<br>"
            "<sup>\u2190 More regularization  |  -log(lambda)  |  Less regularization \u2192</sup>"
        ),
        xaxis_title="-log(lambda)",
        yaxis_title="Standardized Coefficient",
        template="plotly_white",
        height=460,
        margin=dict(t=80, b=40),
    )
    fig.write_image(str(OUTPUT / "lasso_path.png"))
    print(f"Saved: {OUTPUT / 'lasso_path.png'}")

    # Interpretation hint
    inactive = coefs[~coefs["is_active"]]
    if len(inactive) > 0:
        print(f"\nComponents zeroed by Lasso (no signal in KOSDAQ labeled data): "
              f"{list(inactive['feature'])}")
        print("These components may be uninformative for Korean CB-heavy companies.")


if __name__ == "__main__":
    main()
