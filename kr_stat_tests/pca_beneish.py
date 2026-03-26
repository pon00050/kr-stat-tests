"""
pca_beneish.py — Statistical Test 3 (ISL Ch. 12)

PCA on the 8 Beneish components. Reveals independent risk dimensions and
surfaces companies that are anomalous on different component combinations.
Two companies can share the same M-score for completely different reasons.

Input:  01_Data/processed/beneish_scores.parquet
Output: outputs/pca_beneish_scatter.png
        outputs/pca_beneish_loadings.csv
        outputs/pca_beneish_variance.csv

Run from project root:
    python 03_Analysis/statistical_tests/pca_beneish.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

COMPONENTS = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
RISK_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Critical": "#8e44ad",
}


def pca_top_loadings(
    components_matrix: np.ndarray,
    feature_names: list,
    n_top: int = 3,
) -> pd.DataFrame:
    """Extract the top-n features by absolute loading for each principal component.

    Parameters
    ----------
    components_matrix : np.ndarray, shape (n_pcs, n_features)
        pca.components_ from a fitted sklearn PCA object.
    feature_names : list[str]
        Feature names corresponding to columns of the original X matrix.
    n_top : int
        Number of top features to return per PC (default 3).

    Returns
    -------
    pd.DataFrame with columns ['pc', 'feature', 'loading', 'abs_loading'],
    one row per (PC, top feature), sorted by pc then abs_loading descending.
    Pure function — no sklearn calls inside.
    """
    rows = []
    n_pcs = components_matrix.shape[0]
    for i in range(n_pcs):
        pc_label = f"PC{i + 1}"
        loadings = components_matrix[i]
        idx_sorted = np.argsort(np.abs(loadings))[::-1][:n_top]
        for rank, j in enumerate(idx_sorted, 1):
            rows.append({
                "pc": pc_label,
                "feature": feature_names[j],
                "loading": round(float(loadings[j]), 4),
                "abs_loading": round(float(abs(loadings[j])), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    path = DATA / "beneish_scores.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} Beneish score rows")

    df = df[df["m_score"].notna()].copy()
    print(f"Rows with valid m_score: {len(df):,}")

    # Build component matrix; fill NaN with column median
    X = df[COMPONENTS].copy()
    for col in COMPONENTS:
        X[col] = X[col].fillna(X[col].median())

    # Standardize → PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=8, random_state=42)
    scores = pca.fit_transform(X_scaled)

    # Explained variance
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    n_80 = int(np.searchsorted(cum_var, 0.80)) + 1

    print(f"\nPCA explained variance:")
    for i, (v, c) in enumerate(zip(var_ratio, cum_var), 1):
        print(f"  PC{i}: {v * 100:.1f}%  (cumulative: {c * 100:.1f}%)")
    print(f"\nComponents needed for 80% variance: {n_80}")

    # Save variance table
    var_df = pd.DataFrame({
        "component": [f"PC{i}" for i in range(1, 9)],
        "explained_variance_ratio": var_ratio.round(6),
        "cumulative_variance_ratio": cum_var.round(6),
    })
    var_df.to_csv(OUTPUT / "pca_beneish_variance.csv", index=False)

    # Loadings table: features × components
    loadings = pd.DataFrame(
        pca.components_.T,
        index=COMPONENTS,
        columns=[f"PC{i}" for i in range(1, 9)],
    ).round(4)
    loadings.to_csv(OUTPUT / "pca_beneish_loadings.csv")

    print(f"\nTop loadings on PC1: "
          f"{loadings['PC1'].abs().sort_values(ascending=False).head(3).index.tolist()}")
    print(f"Top loadings on PC2: "
          f"{loadings['PC2'].abs().sort_values(ascending=False).head(3).index.tolist()}")

    # PC3 manipulation screen: DSRI (+0.51) + TATA (+0.68) — manipulation-specific dimensions
    TIER1_CODES = {"01051092", "01207761", "00530413", "01049167", "00619640"}

    df["PC1"] = scores[:, 0]
    df["PC2"] = scores[:, 1]
    df["PC3"] = scores[:, 2]
    df["PC3_pct_rank"] = df["PC3"].rank(pct=True)
    df["pc3_flagged"] = df["PC3_pct_rank"] >= 0.90  # top decile

    pc3_out = df[["corp_code", "company_name", "year", "m_score",
                  "PC3", "PC3_pct_rank", "pc3_flagged"]].sort_values("PC3", ascending=False)
    pc3_out.to_csv(OUTPUT / "pca_pc3_scores.csv", index=False, encoding="utf-8-sig")
    print(f"\nPC3 top-decile flags: {df['pc3_flagged'].sum():,} company-years")

    tier1_df = pc3_out[pc3_out["corp_code"].isin(TIER1_CODES)].copy()
    if len(tier1_df) > 0:
        print("\nTier 1 leads -- PC3 vs M-score:")
        tier1_df["company_name_safe"] = tier1_df["company_name"].str.encode("ascii", "replace").str.decode("ascii")
        print(tier1_df[["company_name_safe", "year", "m_score", "PC3", "PC3_pct_rank"]].to_string(index=False))
    else:
        print("\nNo Tier 1 leads found in PC3 output (check corp_code format).")

    # Add risk_tier if missing
    if "risk_tier" not in df.columns:
        df["risk_tier"] = "Low"

    pct1 = var_ratio[0] * 100
    pct2 = var_ratio[1] * 100

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        hover_data={
            "corp_code": True,
            "company_name": True,
            "year": True,
            "m_score": ":.3f",
            "PC1": ":.3f",
            "PC2": ":.3f",
        },
        title=(
            f"PCA of Beneish Components — "
            f"PC1 ({pct1:.1f}%) vs PC2 ({pct2:.1f}%)<br>"
            f"<sup>{len(df):,} company-years; clipped at 1st/99th pct per component</sup>"
        ),
        template="plotly_white",
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=620, margin=dict(t=70, b=30))

    png_path = OUTPUT / "pca_beneish_scatter.png"
    fig.write_image(str(png_path))

    top_loadings = pca_top_loadings(pca.components_, COMPONENTS, n_top=3)
    top_loadings.to_csv(OUTPUT / "pca_top_loadings.csv", index=False)
    print(f"\nTop 3 features per PC (by |loading|):")
    for pc, grp in top_loadings.groupby("pc", sort=False):
        features_str = ", ".join(f"{r.feature}({r.loading:+.3f})" for _, r in grp.iterrows())
        print(f"  {pc}: {features_str}")
    print(f"Saved: {OUTPUT / 'pca_top_loadings.csv'}")

    print(f"\nSaved: {png_path}")
    print(f"Saved: {OUTPUT / 'pca_beneish_loadings.csv'}")
    print(f"Saved: {OUTPUT / 'pca_beneish_variance.csv'}")
    print(f"Saved: {OUTPUT / 'pca_pc3_scores.csv'}")


if __name__ == "__main__":
    main()
