"""
survival_repricing.py — Statistical Test 10 (ISL Ch. 11)

Cox proportional hazards model on time-to-first-repricing.
Tests whether Beneish M-score and officer network centrality predict faster
CB/BW repricing — a direct statistical test of the manipulation hypothesis.

Requires SEIBRO repricing data loaded into cb_bw_events.parquet
(run extract_seibro_repricing.py first).

Input:  01_Data/processed/cb_bw_events.parquet
        01_Data/processed/beneish_scores.parquet
        03_Analysis/officer_network/centrality_report.csv
Output: outputs/survival_cox_summary.csv
        outputs/survival_km.png

Run from project root:
    python 03_Analysis/statistical_tests/survival_repricing.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
ANALYSIS = ROOT / "03_Analysis"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

OBSERVATION_END = pd.Timestamp("2024-01-01")
MIN_REPRICING_EVENTS = 20   # minimum for a Cox model


def parse_first_repricing(hist_str) -> pd.Timestamp | None:
    """Return the earliest repricing date from a JSON repricing_history string."""
    try:
        records = json.loads(hist_str) if isinstance(hist_str, str) else []
    except (json.JSONDecodeError, TypeError):
        return None
    dates: list[pd.Timestamp] = []
    for rec in records:
        raw = rec.get("date") or rec.get("조정일자") or rec.get("adjustment_date")
        if raw:
            try:
                dates.append(pd.to_datetime(str(raw)))
            except Exception:
                pass
    return min(dates) if dates else None


def compute_survival_row(issue_dt, first_repricing_dt) -> tuple[float | None, bool]:
    """Return (duration_days, event_observed)."""
    if pd.isna(issue_dt):
        return None, False
    if first_repricing_dt is not None and first_repricing_dt > issue_dt:
        return float((first_repricing_dt - issue_dt).days), True
    # Censored at observation end
    duration = float((OBSERVATION_END - issue_dt).days)
    return max(duration, 1.0), False


def main() -> None:
    cb_path = DATA / "cb_bw_events.parquet"
    beneish_path = DATA / "beneish_scores.parquet"

    if not cb_path.exists() or not beneish_path.exists():
        print("ERROR: Required parquet files not found.")
        return

    cb = pd.read_parquet(cb_path)
    beneish = pd.read_parquet(beneish_path)
    cb["corp_code"] = cb["corp_code"].astype(str).str.zfill(8)
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)
    print(f"Loaded {len(cb):,} CB/BW events")

    # Parse issue date
    cb["issue_dt"] = pd.to_datetime(cb["issue_date"], errors="coerce")

    # Parse first repricing date from JSON
    cb["first_repricing_dt"] = cb["repricing_history"].apply(parse_first_repricing)

    # Compute duration and event flag
    survival = cb.apply(
        lambda r: pd.Series(compute_survival_row(r["issue_dt"], r["first_repricing_dt"])),
        axis=1,
    )
    cb["duration_days"] = survival[0]
    cb["event_observed"] = survival[1]

    cb_valid = cb[cb["duration_days"].notna() & (cb["duration_days"] > 0)].copy()
    n_events = int(cb_valid["event_observed"].sum())
    print(f"Valid CB/BW events: {len(cb_valid):,}")
    print(f"Events with repricing (not censored): {n_events:,}")

    if n_events < MIN_REPRICING_EVENTS:
        print(f"\nOnly {n_events} repricing events (minimum {MIN_REPRICING_EVENTS} for Cox model).")
        print("Load SEIBRO repricing data first:")
        print("  python 02_Pipeline/extract_seibro_repricing.py")
        return

    # Join Beneish M-score: most recent scored year per company
    beneish_latest = (
        beneish[beneish["m_score"].notna()]
        .sort_values("year", ascending=False)
        .groupby("corp_code")[["m_score"]]
        .first()
        .reset_index()
    )
    cb_valid = cb_valid.merge(beneish_latest, on="corp_code", how="left")

    # Join officer centrality (max betweenness per company)
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
                        "betweenness_centrality": float(r.get("betweenness_centrality", 0) or 0),
                    })
        if rows:
            net_per_co = (
                pd.DataFrame(rows)
                .groupby("corp_code")["betweenness_centrality"]
                .max()
                .reset_index()
            )
            cb_valid = cb_valid.merge(net_per_co, on="corp_code", how="left")
        else:
            cb_valid["betweenness_centrality"] = np.nan
    else:
        cb_valid["betweenness_centrality"] = np.nan

    cb_valid["betweenness_centrality"] = cb_valid["betweenness_centrality"].fillna(0)
    cb_valid["bond_type_cb"] = (cb_valid["bond_type"] == "CB").astype(float)

    # --- Cox model ---
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
    except ImportError:
        print("ERROR: lifelines not installed. Run: uv add lifelines && uv sync")
        return

    cox_cols = [
        "duration_days", "event_observed",
        "m_score", "betweenness_centrality", "bond_type_cb",
    ]
    cox_df = cb_valid[cox_cols].dropna(subset=["m_score"]).copy()
    print(f"\nCox model input: {len(cox_df):,} rows "
          f"({int(cox_df['event_observed'].sum())} repriced, "
          f"{int((~cox_df['event_observed']).sum())} censored)")

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_days", event_col="event_observed")
    cph.print_summary()

    # Extract summary
    summary = cph.summary.reset_index()
    rename = {
        "index": "covariate",
        "exp(coef)": "hazard_ratio",
        "exp(coef) lower 95%": "ci_lower",
        "exp(coef) upper 95%": "ci_upper",
        "p": "p_value",
    }
    summary = summary.rename(columns=rename)
    out_cols = [c for c in ["covariate", "coef", "hazard_ratio", "ci_lower", "ci_upper", "p_value"]
                if c in summary.columns]
    summary[out_cols].to_csv(OUTPUT / "survival_cox_summary.csv", index=False)

    # Concordance
    print(f"\nConcordance index: {cph.concordance_index_:.3f}")
    if cph.concordance_index_ > 0.7:
        print("Model meaningfully discriminates repricing timing.")
    else:
        print("Concordance index < 0.7 — model has weak discrimination (expected with small n).")

    # --- Kaplan-Meier split by m_score > / ≤ -1.78 ---
    THRESHOLD = -1.78
    fig = go.Figure()
    for subset, label, color in [
        (cox_df[cox_df["m_score"] > THRESHOLD], f"m_score > {THRESHOLD}", "#e74c3c"),
        (cox_df[cox_df["m_score"] <= THRESHOLD], f"m_score ≤ {THRESHOLD}", "#2ecc71"),
    ]:
        if len(subset) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            subset["duration_days"],
            event_observed=subset["event_observed"],
            label=label,
        )
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_.iloc[:, 0].values
        fig.add_trace(go.Scatter(x=t, y=s, name=label, mode="lines",
                                 line=dict(color=color, width=2)))

    fig.update_layout(
        title=(
            "Kaplan-Meier: Time to First CB/BW Repricing by Beneish Risk<br>"
            "<sup>Downward-faster red curve = higher Beneish score predicts faster repricing</sup>"
        ),
        xaxis_title="Days from CB/BW Issuance",
        yaxis_title="Proportion Not Yet Repriced",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=460,
        margin=dict(t=80, b=40),
    )
    fig.write_image(str(OUTPUT / "survival_km.png"))

    print(f"\nSaved: {OUTPUT / 'survival_cox_summary.csv'}")
    print(f"Saved: {OUTPUT / 'survival_km.png'}")


if __name__ == "__main__":
    main()
