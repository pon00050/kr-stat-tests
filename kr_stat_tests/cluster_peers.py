"""
cluster_peers.py — Statistical Test 4 (ISL Ch. 12)

K-Means peer-group clustering. KOSDAQ companies clustered by size, revenue
intensity, leverage, and industry code. M-score z-scores computed within each
peer cluster × year — a normalized signal meaningful relative to comparable
Korean companies rather than the US-calibrated -1.78 threshold.

Input:  01_Data/processed/company_financials.parquet
        01_Data/processed/beneish_scores.parquet
Output: outputs/peer_clusters.csv
        outputs/cluster_silhouette.csv

Run from project root:
    python 03_Analysis/statistical_tests/cluster_peers.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

K_VALUES = [6, 8, 10]
ZSCORE_FLAG_THRESHOLD = 2.0


def build_features(fin: pd.DataFrame) -> pd.DataFrame:
    """Build per-(corp_code, year) clustering features."""
    f = fin[["corp_code", "year", "total_assets", "revenue", "lt_debt", "ksic_code"]].copy()

    f["log_assets"] = np.log1p(f["total_assets"].clip(lower=0))

    ta = f["total_assets"].replace(0, np.nan)
    f["revenue_to_assets"] = f["revenue"] / ta
    f["debt_to_assets"] = f["lt_debt"].fillna(0) / ta

    # KSIC 2-digit industry code as numeric feature
    f["ksic_2d"] = (
        pd.to_numeric(f["ksic_code"].astype(str).str[:2], errors="coerce")
        .fillna(0)
        .astype(float)
    )

    return f[["corp_code", "year", "log_assets", "revenue_to_assets", "debt_to_assets", "ksic_2d"]]


def within_group_zscore(series: pd.Series) -> pd.Series:
    # Exclude inf/-inf from statistics; those entries receive NaN z-score (correct — not flagged)
    finite = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(finite) < 2:
        return pd.Series(np.nan, index=series.index)
    std = finite.std()
    mean = finite.mean()
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


def gmm_aic_bic(X: np.ndarray, k_values: list) -> pd.DataFrame:
    """Fit Gaussian Mixture Models at each k and return AIC/BIC for model selection.

    ISL §12.4 recommends AIC/BIC over silhouette for GMM — lower is better.
    BIC is more conservative (penalises complexity more) and is preferred for
    selecting the number of components.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features) — must be pre-scaled.
    k_values : list[int] — number of GMM components to evaluate.

    Returns
    -------
    pd.DataFrame with columns ['k', 'gmm_aic', 'gmm_bic'],
    one row per k value, values rounded to 2 decimal places.
    """
    from sklearn.mixture import GaussianMixture
    rows = []
    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(X)
        rows.append({
            "k": k,
            "gmm_aic": round(gmm.aic(X), 2),
            "gmm_bic": round(gmm.bic(X), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    fin_path = DATA / "company_financials.parquet"
    beneish_path = DATA / "beneish_scores.parquet"

    if not fin_path.exists() or not beneish_path.exists():
        print("ERROR: Required parquet files not found.")
        return

    fin = pd.read_parquet(fin_path)
    beneish = pd.read_parquet(beneish_path)
    fin["corp_code"] = fin["corp_code"].astype(str).str.zfill(8)
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    print(f"Loaded {len(fin):,} financial rows, {len(beneish):,} Beneish rows")

    features = build_features(fin)
    feat_cols = ["log_assets", "revenue_to_assets", "debt_to_assets", "ksic_2d"]

    # Require log_assets (must have total_assets)
    features = features.dropna(subset=["log_assets"])

    # Impute remaining NaN with column median
    for col in feat_cols:
        features[col] = features[col].fillna(features[col].median())

    print(f"Feature matrix: {len(features):,} rows × {len(feat_cols)} features")

    X = StandardScaler().fit_transform(features[feat_cols].values)

    # Try k values; pick by silhouette score
    sil_results = []
    print("\nSilhouette scores:")
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sample_size = min(5000, len(X))
        sil = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        sil_results.append({"k": k, "silhouette_score": round(sil, 4)})
        print(f"  k={k}: silhouette={sil:.4f}")

    sil_df = pd.DataFrame(sil_results)
    sil_df.to_csv(OUTPUT / "cluster_silhouette.csv", index=False)

    best_k = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])
    print(f"\nBest k={best_k} (silhouette={sil_df['silhouette_score'].max():.4f})")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features = features.copy()
    features["peer_cluster"] = km_best.fit_predict(X)

    # Merge cluster labels into beneish_scores
    merged = beneish[["corp_code", "year", "m_score", "company_name"]].merge(
        features[["corp_code", "year", "peer_cluster"]],
        on=["corp_code", "year"],
        how="left",
    )

    # Compute M-score z-score within (cluster × year)
    merged["mscore_zscore_within_cluster"] = merged.groupby(
        ["peer_cluster", "year"]
    )["m_score"].transform(within_group_zscore)

    merged["cluster_flag"] = merged["mscore_zscore_within_cluster"] > ZSCORE_FLAG_THRESHOLD

    # Stats
    scored = merged["m_score"].notna()
    flagged = merged["cluster_flag"].sum()
    print(f"Cluster-relative flags (z>{ZSCORE_FLAG_THRESHOLD}): {flagged:,} "
          f"of {scored.sum():,} scored company-years")

    # Cluster size summary
    cluster_sizes = features.groupby("peer_cluster").size().sort_values(ascending=False)
    print(f"\nCluster sizes:\n{cluster_sizes.to_string()}")

    print("\nGMM AIC/BIC (ISL 12.4 model selection - lower BIC is better):")
    gmm_df = gmm_aic_bic(X, K_VALUES)
    for _, row in gmm_df.iterrows():
        print(f"  k={int(row['k'])}: AIC={row['gmm_aic']:.1f}  BIC={row['gmm_bic']:.1f}")
    best_gmm_k = int(gmm_df.loc[gmm_df["gmm_bic"].idxmin(), "k"])
    print(f"Best k by BIC: {best_gmm_k}")
    gmm_df.to_csv(OUTPUT / "cluster_gmm_fit.csv", index=False)
    print(f"Saved: {OUTPUT / 'cluster_gmm_fit.csv'}")

    out = OUTPUT / "peer_clusters.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}")
    print(f"Saved: {OUTPUT / 'cluster_silhouette.csv'}")


if __name__ == "__main__":
    main()
