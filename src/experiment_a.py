"""
experiment_a.py — Retrieval Bias Audit (Experiment A)

Runs all 100 standardized queries through the baseline cosine similarity
retriever and measures the institutional distribution of top-10 results.
Papers with unknown institution labels are excluded from all calculations.

Uses Fairlearn to compute:
  - Selection Rate per group (privileged / underrepresented)
  - Selection Rate Ratio = underrepresented_rate / privileged_rate
    (1.0 = perfect parity, < 1.0 = underrepresented group selected less)
  - Statistical Parity Difference (SPD) = privileged_rate - underrepresented_rate

Output:
  - Console table
  - data/results/experiment_a_results.json

Run from inside src/:
    python experiment_a.py
"""

import json
from pathlib import Path
from statistics import mean

import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate

from retriever import load_model, connect_qdrant, search

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_queries.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_JSON = RESULTS_DIR / "experiment_a_results.json"

TOP_K = 10
OVERSAMPLE_K = 100  # retrieve extra so top-10 can be filled with labeled papers only


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_query(query: str, model, client) -> dict | None:
    # Over-retrieve, then keep the top-k *labeled* papers so unknown-label
    # papers never occupy result slots. This measures ranking bias among the
    # papers we can actually classify.
    hits = search(query, model, client, k=OVERSAMPLE_K)

    labeled = [
        hit for hit in hits
        if hit.payload.get("institution_label") in ("privileged", "underrepresented")
    ][:TOP_K]

    if not labeled:
        return None

    n = len(labeled)
    n_priv = sum(1 for h in labeled if h.payload["institution_label"] == "privileged")
    n_under = n - n_priv

    priv_rate = n_priv / n
    under_rate = n_under / n
    spd = round(priv_rate - under_rate, 4)
    ratio = round(under_rate / priv_rate, 4) if n_priv > 0 else None

    return {
        "query": query,
        "top_k": TOP_K,
        "labeled_results": n,
        "counts": {"privileged": n_priv, "underrepresented": n_under},
        "privileged_rate": round(priv_rate, 4),
        "underrepresented_rate": round(under_rate, 4),
        "spd": spd,
        "selection_rate_ratio": ratio,
        "retrieved": [
            {
                "title": hit.payload.get("title", ""),
                "institution_label": hit.payload.get("institution_label"),
                "score": round(hit.score, 4),
            }
            for hit in labeled
        ],
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(per_query: list, summary: dict):
    def short(q, n=45):
        return q if len(q) <= n else q[:n - 3] + "..."

    header = f"{'Query':45}  {'Priv':>4}  {'Under':>5}  {'SPD':>7}  {'Ratio':>6}"
    sep = "-" * len(header)
    print("\nPer-query results (unknown labels excluded):")
    print(header)
    print(sep)
    for r in per_query:
        ratio_str = f"{r['selection_rate_ratio']:.4f}" if r["selection_rate_ratio"] is not None else "   N/A"
        print(
            f"{short(r['query']):45}  "
            f"{r['counts']['privileged']:>4}  "
            f"{r['counts']['underrepresented']:>5}  "
            f"{r['spd']:>+7.4f}  "
            f"{ratio_str:>6}"
        )
    print(sep)
    fa = summary["fairlearn_aggregate"]
    cb = summary["corpus_base_rates"]
    print(f"\nSummary (unknown labels excluded)")
    print(f"  Queries run:                    {summary['num_queries']}")
    print(f"  Queries with labeled hits:      {summary['queries_with_labeled_hits']}")
    print(f"  Queries skipped (all unknown):  {summary['num_queries'] - summary['queries_with_labeled_hits']}")
    print(f"")
    print(f"  Fairlearn aggregate (chance of a corpus paper being retrieved, pooled):")
    print(f"    Privileged selection rate:      {fa['privileged_selection_rate']:.6f}")
    print(f"    Underrepresented selection rate:{fa['underrepresented_selection_rate']:.6f}")
    print(f"    Selection rate ratio (priv/under): {fa['selection_rate_ratio']:.4f}")
    print(f"    SPD:                            {fa['spd']:+.6f}")
    print(f"")
    print(f"  Per-query means:")
    print(f"    Mean privileged rate:           {summary['mean_privileged_rate']:.4f}")
    print(f"    Mean underrepresented rate:     {summary['mean_underrepresented_rate']:.4f}")
    print(f"    Mean selection rate ratio:      {summary['mean_selection_rate_ratio']:.4f}")
    print(f"    Mean SPD:                       {summary['mean_spd']:+.4f}")
    print(f"")
    print(f"  Corpus base rates (labeled papers only):")
    print(f"    Privileged:                     {cb['privileged_base_rate']:.4f} ({cb['privileged_count']:,} papers)")
    print(f"    Underrepresented:               {cb['underrepresented_base_rate']:.4f} ({cb['underrepresented_count']:,} papers)")
    print(f"")
    priv_lift = round(summary["mean_privileged_rate"] - cb["privileged_base_rate"], 4)
    under_lift = round(summary["mean_underrepresented_rate"] - cb["underrepresented_base_rate"], 4)
    print(f"  Retrieval lift vs. corpus base rate:")
    print(f"    Privileged:      {priv_lift:+.4f}  ({'over' if priv_lift > 0 else 'under'}-represented in retrieval)")
    print(f"    Underrepresented:{under_lift:+.4f}  ({'over' if under_lift > 0 else 'under'}-represented in retrieval)")
    print(f"")
    print(f"  SPD > 0  → privileged over-represented vs. underrepresented")
    print(f"  Lift > 0 → group retrieved more than its share of the corpus")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_corpus_rates(client) -> dict:
    """Scroll the full collection and compute privileged/underrepresented base rates."""
    counts = {"privileged": 0, "underrepresented": 0}
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name="fairsearch_arxiv",
            limit=1000,
            offset=offset,
            with_payload=["institution_label"],
            with_vectors=False,
        )
        for point in batch:
            label = point.payload.get("institution_label")
            if label in counts:
                counts[label] += 1
        if offset is None:
            break

    total = counts["privileged"] + counts["underrepresented"]
    return {
        "privileged_count": counts["privileged"],
        "underrepresented_count": counts["underrepresented"],
        "total_labeled": total,
        "privileged_base_rate": round(counts["privileged"] / total, 4) if total else None,
        "underrepresented_base_rate": round(counts["underrepresented"] / total, 4) if total else None,
    }


def main():
    print("Loading model and connecting to Qdrant...")
    model = load_model()
    client = connect_qdrant()

    print("Computing corpus base rates (labeled papers only)...")
    corpus = get_corpus_rates(client)
    print(f"  Privileged:      {corpus['privileged_count']:,} ({corpus['privileged_base_rate']:.4f})")
    print(f"  Underrepresented:{corpus['underrepresented_count']:,} ({corpus['underrepresented_base_rate']:.4f})")
    print()

    queries = json.loads(EVAL_FILE.read_text())
    print(f"Loaded {len(queries)} queries. Running top-{TOP_K} retrieval...\n")

    per_query = []
    skipped = 0
    for i, item in enumerate(queries, start=1):
        q = item["query"]
        print(f"[{i:>3}/{len(queries)}] {q[:70]}")
        result = measure_query(q, model, client)
        if result is None:
            skipped += 1
        else:
            per_query.append(result)

    # Fairlearn aggregate: selection rate = fraction of each group's corpus
    # papers selected into top-k, pooled across all queries. Each query gives
    # every labeled corpus paper one selection opportunity, so trials =
    # corpus_count x num_queries and positives = retrieved slots for that group.
    priv_hits = sum(r["counts"]["privileged"] for r in per_query)
    under_hits = sum(r["counts"]["underrepresented"] for r in per_query)
    priv_trials = corpus["privileged_count"] * len(per_query)
    under_trials = corpus["underrepresented_count"] * len(per_query)

    y_pred = np.concatenate([
        np.ones(priv_hits, dtype=int),
        np.zeros(priv_trials - priv_hits, dtype=int),
        np.ones(under_hits, dtype=int),
        np.zeros(under_trials - under_hits, dtype=int),
    ])
    y_true = np.zeros_like(y_pred)  # unused by selection_rate
    all_groups = np.concatenate([
        np.full(priv_trials, "privileged"),
        np.full(under_trials, "underrepresented"),
    ])

    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=all_groups,
    )
    by_group = mf.by_group.to_dict()
    agg_priv_rate = by_group.get("privileged", 0.0)
    agg_under_rate = by_group.get("underrepresented", 0.0)
    agg_spd = round(float(agg_priv_rate - agg_under_rate), 6)
    agg_ratio = round(float(agg_priv_rate / agg_under_rate), 4) if agg_under_rate > 0 else None

    ratios = [r["selection_rate_ratio"] for r in per_query if r["selection_rate_ratio"] is not None]

    summary = {
        "top_k": TOP_K,
        "num_queries": len(queries),
        "queries_with_labeled_hits": len(per_query),
        "corpus_base_rates": corpus,
        "fairlearn_aggregate": {
            "privileged_selection_rate": round(float(agg_priv_rate), 6),
            "underrepresented_selection_rate": round(float(agg_under_rate), 6),
            "spd": agg_spd,
            "selection_rate_ratio": agg_ratio,
        },
        "mean_privileged_rate": round(mean(r["privileged_rate"] for r in per_query), 4),
        "mean_underrepresented_rate": round(mean(r["underrepresented_rate"] for r in per_query), 4),
        "mean_spd": round(mean(r["spd"] for r in per_query), 4),
        "mean_selection_rate_ratio": round(mean(ratios), 4) if ratios else None,
    }

    output = {"summary": summary, "per_query": per_query}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))

    print_table(per_query, summary)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
