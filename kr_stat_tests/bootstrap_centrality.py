"""
bootstrap_centrality.py — Statistical Test 8 (ISL Ch. 5)

Validates officer network centrality signal via 200 bootstrap resamples.
Individuals consistently ranked in top-20 by betweenness centrality
across resamples are genuine signal; those appearing rarely are noise.

Input:  01_Data/processed/officer_holdings.parquet
        01_Data/processed/beneish_scores.parquet
Output: outputs/bootstrap_centrality.csv

Run from project root:
    python 03_Analysis/statistical_tests/bootstrap_centrality.py
"""

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from kr_forensic_core.paths import data_dir as _data_dir
    _PROCESSED_DIR = _data_dir(repo_root=Path(__file__).resolve().parents[1])
except Exception:
    _PROCESSED_DIR = Path(__file__).resolve().parents[1] / "01_Data" / "processed"
ROOT = _PROCESSED_DIR.parent.parent
DATA = ROOT / "01_Data" / "processed"
OUTPUT = Path(__file__).parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

N_BOOTSTRAP = 200
TOP_N = 20
STABLE_THRESHOLD = 0.80   # ≥80% appearances = genuine signal
NOISE_THRESHOLD = 0.50    # <50% = noise


def build_graph(holdings: pd.DataFrame, flagged_codes: set[str]) -> nx.Graph:
    """Build officer–company bipartite graph restricted to flagged companies."""
    G = nx.Graph()
    for _, row in holdings.iterrows():
        cc = str(row["corp_code"]).zfill(8)
        if cc not in flagged_codes:
            continue
        officer = str(row["officer_name"]).strip() if not pd.isna(row["officer_name"]) else ""
        if not officer:
            continue
        # Add edge: officer name → corp_code (both are nodes; type inferred by format)
        G.add_edge(f"PERSON:{officer}", f"CORP:{cc}")
    return G


def top_n_officers_by_betweenness(G: nx.Graph, n: int) -> list[str]:
    """Return top-n officer nodes ranked by betweenness centrality."""
    if G.number_of_edges() == 0:
        return []
    officer_nodes = [node for node in G.nodes if node.startswith("PERSON:")]
    if len(officer_nodes) < 2:
        return officer_nodes[:n]
    try:
        bc = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        return []
    officer_bc = {k: v for k, v in bc.items() if k.startswith("PERSON:")}
    ranked = sorted(officer_bc, key=officer_bc.get, reverse=True)
    return ranked[:n]


def main() -> None:
    holdings_path = DATA / "officer_holdings.parquet"
    beneish_path = DATA / "beneish_scores.parquet"

    if not holdings_path.exists() or not beneish_path.exists():
        print("ERROR: Required parquet files not found.")
        return

    holdings = pd.read_parquet(holdings_path)
    beneish = pd.read_parquet(beneish_path)

    holdings["corp_code"] = holdings["corp_code"].astype(str).str.zfill(8)
    beneish["corp_code"] = beneish["corp_code"].astype(str).str.zfill(8)

    flagged_codes = set(beneish[beneish["flag"] == True]["corp_code"].unique())  # noqa: E712
    print(f"Loaded {len(holdings):,} officer holding rows")
    print(f"Flagged companies: {len(flagged_codes):,}")

    holdings_flagged = holdings[holdings["corp_code"].isin(flagged_codes)].copy()
    n = len(holdings_flagged)
    print(f"Holdings in flagged companies: {n:,}")

    if n == 0:
        print("No holdings in flagged companies. Exiting.")
        return

    rng = np.random.default_rng(42)
    appearance_counts: dict[str, int] = defaultdict(int)

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        sample = holdings_flagged.iloc[idx]
        G = build_graph(sample, flagged_codes)
        top = top_n_officers_by_betweenness(G, TOP_N)
        for person_node in top:
            appearance_counts[person_node] += 1
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap {b + 1}/{N_BOOTSTRAP}...")

    results = []
    for person_node, count in appearance_counts.items():
        # Strip the PERSON: prefix for display
        name = person_node.removeprefix("PERSON:")
        stability = count / N_BOOTSTRAP
        results.append({
            "person_name": name,
            "n_top20_appearances": count,
            "stability_pct": round(stability * 100, 1),
            "is_stable": stability >= STABLE_THRESHOLD,
            "is_noise": stability < NOISE_THRESHOLD,
        })

    out_df = pd.DataFrame(results).sort_values("n_top20_appearances", ascending=False)

    stable = out_df[out_df["is_stable"]]
    noise = out_df[out_df["is_noise"]]

    # Save CSV first (before any prints that might encode Korean names)
    out = OUTPUT / "bootstrap_centrality.csv"
    out_df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nResults (n={N_BOOTSTRAP} bootstrap draws, top-{TOP_N} per draw):")
    print(f"  Stable signal (>={STABLE_THRESHOLD*100:.0f}%): {len(stable):,} individuals")
    print(f"  Noise (<{NOISE_THRESHOLD*100:.0f}%):           {len(noise):,} individuals")

    if len(stable) > 0:
        print(f"\nTop stable individuals (stability%):")
        for _, row in out_df.head(10).iterrows():
            name_safe = row["person_name"].encode("ascii", "replace").decode("ascii")
            print(f"  {name_safe}: {row['stability_pct']}%")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
