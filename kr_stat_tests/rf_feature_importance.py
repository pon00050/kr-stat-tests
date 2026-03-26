"""
rf_feature_importance.py — Statistical Test 6 (ISL Ch. 8)

Random Forest feature importance across all 4 analysis milestones.
Ranks which signals carry real predictive weight when distinguishing
the 5 Tier 1 leads (fraud=1) from auto-selected clean controls (fraud=0).

Requires labels.csv with ≥10 rows total (including auto-populated controls).

Input:  labels.csv (in same directory)
        01_Data/processed/beneish_scores.parquet
        03_Analysis/cb_bw_summary.csv
        03_Analysis/timing_anomalies.csv  (or outputs/fdr_timing_anomalies.csv)
        03_Analysis/officer_network/centrality_report.csv
Output: outputs/rf_importance.csv
        outputs/rf_importance.png

Run from project root:
    python 03_Analysis/statistical_tests/rf_feature_importance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
import plotly.express as px

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
ANALYSIS = ROOT / "03_Analysis"
HERE = Path(__file__).parent
OUTPUT = HERE / "outputs"
OUTPUT.mkdir(exist_ok=True)

MIN_LABELED_TOTAL = 10


def rf_events_per_variable(n_fraud: int, n_features_used: int) -> float:
    """Events-per-variable (EPV): n_fraud / n_features_used. Accepted minimum is 10–15."""
    return n_fraud / n_features_used if n_features_used > 0 else 0.0


def compare_importance_methods(
    feature_cols: list[str],
    rf_importances: np.ndarray,
    perm_importances_mean: np.ndarray,
) -> pd.DataFrame:
    """Compare impurity-based and permutation-based feature importance rankings.

    Returns DataFrame with columns:
    ['feature', 'rf_rank', 'perm_rank', 'rf_importance', 'permutation_importance', 'rank_divergence']
    sorted by perm_rank ascending (permutation is the methodologically sound ranking).
    rank_divergence = |rf_rank - perm_rank|: large values flag disagreements between methods.
    """
    df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importances,
        "permutation_importance": perm_importances_mean,
    })
    # Rank descending (1 = most important); method='first' avoids ties
    df["rf_rank"] = df["rf_importance"].rank(ascending=False, method="first").astype(int)
    df["perm_rank"] = df["permutation_importance"].rank(ascending=False, method="first").astype(int)
    df["rank_divergence"] = (df["rf_rank"] - df["perm_rank"]).abs()
    return (
        df[["feature", "rf_rank", "perm_rank", "rf_importance",
            "permutation_importance", "rank_divergence"]]
        .sort_values("perm_rank")
        .reset_index(drop=True)
    )


def load_labels() -> pd.DataFrame:
    labels_path = HERE / "labels.csv"
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path, dtype={"corp_code": str})
    labels["corp_code"] = labels["corp_code"].str.zfill(8)
    return labels


def auto_controls(labels_df: pd.DataFrame, n_controls: int = 20) -> pd.DataFrame:
    """Select clean controls using external criteria only — not M-score."""
    beneish = pd.read_parquet(DATA / "beneish_scores.parquet")
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    labeled_codes = set(labels_df["corp_code"])

    cb_codes: set[str] = set()
    cb_path = ANALYSIS / "cb_bw_summary.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path, dtype=str)
        cb_codes = set(cb["corp_code"].astype(str).str.zfill(8))
    else:
        cb_bw = pd.read_parquet(DATA / "cb_bw_events.parquet")
        cb_bw["corp_code"] = cb_bw["corp_code"].astype(str).str.zfill(8)
        cb_codes = set(cb_bw["corp_code"])

    clean = (
        beneish[
            ~beneish["corp_code"].isin(labeled_codes)   # not already labeled
            & ~beneish["corp_code"].isin(cb_codes)      # no CB/BW activity (external criterion)
        ]
        .groupby("corp_code")
        .filter(lambda g: g["m_score"].notna().sum() >= 3)  # at least 3 scoreable years
        .groupby("corp_code")
        .first()
        .reset_index()
        .sort_values("corp_code")                       # neutral sort — NOT by m_score
        [["corp_code", "company_name"]]
        .head(n_controls)
    )
    clean["fraud_label"] = 0
    clean["notes"] = "auto-selected control (no CB/BW, ≥3 scoreable years)"
    return clean


def build_feature_matrix(corp_codes: np.ndarray) -> pd.DataFrame:
    """Merge features from all 4 milestones per corp_code (most recent year)."""
    # --- Beneish features ---
    beneish = pd.read_parquet(DATA / "beneish_scores.parquet")
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    beneish_feats = (
        beneish[beneish["m_score"].notna()]
        .sort_values("year", ascending=False)
        .groupby("corp_code")
        .first()
        .reset_index()[
            [
                "corp_code", "m_score", "dsri", "gmi", "aqi", "sgi",
                "depi", "sgai", "lvgi", "tata", "sector_percentile",
            ]
        ]
    )

    # --- CB/BW features ---
    cb_path = ANALYSIS / "cb_bw_summary.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path, dtype=str)
        cb["corp_code"] = cb["corp_code"].astype(str).str.zfill(8)
        cb["anomaly_score"] = pd.to_numeric(cb["anomaly_score"], errors="coerce").fillna(0)
        cb["repricing_flag"] = (
            cb["repricing_flag"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
        )
        cb_agg = (
            cb.groupby("corp_code")
            .agg(
                max_cb_anomaly_score=("anomaly_score", "max"),
                n_cb_flagged_events=("anomaly_score", lambda x: (x > 0).sum()),
                has_repricing=("repricing_flag", "max"),
            )
            .reset_index()
        )
    else:
        cb_agg = pd.DataFrame({"corp_code": corp_codes})
        cb_agg[["max_cb_anomaly_score", "n_cb_flagged_events", "has_repricing"]] = np.nan

    # --- Timing anomaly features (prefer FDR-corrected if available) ---
    fdr_path = OUTPUT / "fdr_timing_anomalies.csv"
    timing_path = ANALYSIS / "timing_anomalies.csv"
    if fdr_path.exists():
        timing = pd.read_csv(fdr_path, dtype=str)
        timing["corp_code"] = timing["corp_code"].astype(str).str.zfill(8)
        timing["fdr_flagged"] = timing["fdr_flagged"].map({"True": 1, "False": 0}).fillna(0)
        timing["volume_ratio"] = pd.to_numeric(timing["volume_ratio"], errors="coerce")
        timing_agg = (
            timing.groupby("corp_code")
            .agg(n_timing_flags=("fdr_flagged", "sum"), max_volume_ratio=("volume_ratio", "max"))
            .reset_index()
        )
    elif timing_path.exists():
        timing = pd.read_csv(timing_path, dtype=str)
        timing["corp_code"] = timing["corp_code"].astype(str).str.zfill(8)
        timing["flag"] = timing["flag"].map({"True": 1, "False": 0}).fillna(0)
        timing["volume_ratio"] = pd.to_numeric(timing["volume_ratio"], errors="coerce")
        timing_agg = (
            timing.groupby("corp_code")
            .agg(n_timing_flags=("flag", "sum"), max_volume_ratio=("volume_ratio", "max"))
            .reset_index()
        )
    else:
        timing_agg = pd.DataFrame({"corp_code": corp_codes})
        timing_agg[["n_timing_flags", "max_volume_ratio"]] = np.nan

    # --- Officer network centrality features ---
    net_path = ANALYSIS / "officer_network" / "centrality_report.csv"
    if net_path.exists():
        net = pd.read_csv(net_path)
        rows = []
        for _, r in net.iterrows():
            for cc in str(r.get("companies", "")).split("|"):
                cc = cc.strip().zfill(8)
                if cc:
                    rows.append({
                        "corp_code": cc,
                        "betweenness": float(r.get("betweenness_centrality", 0) or 0),
                    })
        if rows:
            net_exp = pd.DataFrame(rows)
            net_agg = (
                net_exp.groupby("corp_code")
                .agg(max_betweenness=("betweenness", "max"), n_network_officers=("betweenness", "count"))
                .reset_index()
            )
        else:
            net_agg = pd.DataFrame({"corp_code": corp_codes})
            net_agg[["max_betweenness", "n_network_officers"]] = np.nan
    else:
        net_agg = pd.DataFrame({"corp_code": corp_codes})
        net_agg[["max_betweenness", "n_network_officers"]] = np.nan

    # Merge all onto corp_codes
    base = pd.DataFrame({"corp_code": corp_codes})
    feat = (
        base
        .merge(beneish_feats, on="corp_code", how="left")
        .merge(cb_agg, on="corp_code", how="left")
        .merge(timing_agg, on="corp_code", how="left")
        .merge(net_agg, on="corp_code", how="left")
    )
    return feat


def main() -> None:
    labels = load_labels()
    if len(labels) == 0:
        print(f"ERROR: labels.csv not found at {HERE / 'labels.csv'}")
        return

    controls = auto_controls(labels, n_controls=20)
    if len(controls) < 5:
        print(f"Warning: only {len(controls)} clean controls found (expected ≥20).")

    all_labels = pd.concat([labels, controls], ignore_index=True).drop_duplicates("corp_code")
    n_fraud = int((all_labels["fraud_label"] == 1).sum())
    n_clean = int((all_labels["fraud_label"] == 0).sum())
    print(f"Labeled set: {n_fraud} fraud=1, {n_clean} fraud=0  (total={len(all_labels)})")

    if len(all_labels) < MIN_LABELED_TOTAL:
        print(f"Only {len(all_labels)} labeled samples (minimum {MIN_LABELED_TOTAL}). "
              "Add more companies to labels.csv.")
        return

    feats = build_feature_matrix(all_labels["corp_code"].values)
    merged = all_labels.merge(feats, on="corp_code", how="left")

    non_feat_cols = {"corp_code", "company_name", "fraud_label", "notes"}
    feature_cols = [c for c in merged.columns if c not in non_feat_cols]

    # Drop features with >30% missing across labeled set
    miss_rate = merged[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if miss_rate[c] <= 0.30]
    print(f"Features after >30% missing filter: {len(feature_cols)}")
    print(f"  {feature_cols}")

    epv = rf_events_per_variable(n_fraud, len(feature_cols))
    print(f"RF EPV (events-per-variable): {epv:.1f}  [features used: {len(feature_cols)}, fraud=1: {n_fraud}]")
    if epv < 10:
        print(
            f"WARNING: RF EPV={epv:.1f} is below the accepted minimum of 10–15. "
            "Feature importance rankings are directional only — findings may not generalize."
        )

    X = merged[feature_cols].fillna(merged[feature_cols].median())
    y = merged["fraud_label"].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
    )

    # GroupKFold ensures no company's years appear in both train and validation folds
    n = len(y)
    groups = pd.factorize(merged["corp_code"].values)[0]
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    cv_name = f"GroupKFold(k={n_splits})"

    cv_scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    print(f"\n{cv_name} AUC-ROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full set for importances
    clf.fit(X, y)

    perm = permutation_importance(clf, X, y, n_repeats=20, random_state=42)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": clf.feature_importances_.round(6),
        "permutation_importance": perm.importances_mean.round(6),
        "permutation_std": perm.importances_std.round(6),
    }).sort_values("permutation_importance", ascending=False)

    imp_df["epv"] = round(epv, 2)
    imp_df.to_csv(OUTPUT / "rf_importance.csv", index=False)

    print(f"\nTop 5 features (permutation importance):")
    print(imp_df[["feature", "permutation_importance"]].head(5).to_string(index=False))

    # Impurity vs. permutation importance comparison
    comparison_df = compare_importance_methods(
        feature_cols,
        clf.feature_importances_,
        perm.importances_mean,
    )
    comparison_df.to_csv(OUTPUT / "rf_importance_comparison.csv", index=False)

    divergent = comparison_df[comparison_df["rank_divergence"] >= 2].head(3)
    if len(divergent) > 0:
        print(f"\nFeatures where impurity and permutation methods disagree (rank_divergence >= 2):")
        print(divergent[["feature", "rf_rank", "perm_rank", "rank_divergence"]].to_string(index=False))

    scatter = px.scatter(
        comparison_df,
        x="rf_rank",
        y="perm_rank",
        text="feature",
        title=(
            "Impurity vs. Permutation Importance Ranking<br>"
            "<sup>Diagonal = perfect agreement; off-diagonal = method disagreement</sup>"
        ),
        labels={"rf_rank": "Impurity rank (1 = most important)", "perm_rank": "Permutation rank"},
        template="plotly_white",
    )
    max_rank = len(feature_cols)
    scatter.add_shape(
        type="line", x0=1, y0=1, x1=max_rank, y1=max_rank,
        line=dict(dash="dot", color="gray"),
    )
    scatter.update_traces(textposition="top center")
    scatter.update_layout(height=460, margin=dict(t=80, b=40))
    scatter.write_image(str(OUTPUT / "rf_importance_comparison.png"))

    print(f"\nSaved: {OUTPUT / 'rf_importance_comparison.csv'}")
    print(f"Saved: {OUTPUT / 'rf_importance_comparison.png'}")

    # Horizontal bar chart
    plot_df = imp_df.sort_values("permutation_importance")
    fig = px.bar(
        plot_df,
        x="permutation_importance",
        y="feature",
        orientation="h",
        error_x="permutation_std",
        title=(
            f"Random Forest Feature Importance<br>"
            f"<sup>n={n} ({n_fraud} fraud, {n_clean} clean) | "
            f"{cv_name} AUC={cv_scores.mean():.3f}</sup>"
        ),
        template="plotly_white",
        color_discrete_sequence=["#4a90d9"],
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(420, len(feature_cols) * 28), margin=dict(t=70, b=30))
    fig.write_image(str(OUTPUT / "rf_importance.png"))

    print(f"\nSaved: {OUTPUT / 'rf_importance.csv'}")
    print(f"Saved: {OUTPUT / 'rf_importance.png'}")


if __name__ == "__main__":
    main()
