"""
experiment_a_mmr.py — Retrieval Bias Audit with MMR

Runs all standardized queries through the MMR retriever
and measures the institutional distribution of top-10 results.
Papers with unknown institution labels are excluded from all calculations.

Output:
- Console table
- data/results/experiment_a_mmr_results.json

Run from inside src/:
python experiment_a_mmr.py
"""

import json
from pathlib import Path
from statistics import mean

import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate

from retriever_mmr import load_model, connect_qdrant, search_mmr


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_queries.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_JSON = RESULTS_DIR / "experiment_a_mmr_results.json"

TOP_K = 10
FETCH_K = 100
MMR_LAMBDA = 0.5


def load_queries(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError("retrieval_eval_queries.json must contain a JSON list.")

    queries = []
    for item in raw:
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict) and "query" in item:
            q = str(item["query"]).strip()
        else:
            raise ValueError("Each query must be a string or an object with a 'query' field.")

        if q:
            queries.append(q)

    return queries


def hit_to_dict(hit) -> dict:
    payload = hit.payload or {}
    return {
        "paper_id": payload.get("paper_id"),
        "title": payload.get("title", ""),
        "category": payload.get("category"),
        "year": payload.get("year"),
        "institution_label": payload.get("institution_label"),
        "score": round(float(hit.score), 4),
        "mmr_rank": payload.get("mmr_rank"),
        "mmr_lambda": payload.get("mmr_lambda"),
        "query_similarity": payload.get("query_similarity"),
    }


def measure_query(query: str, model, client) -> dict | None:
    hits = search_mmr(
        query=query,
        model=model,
        client=client,
        k=TOP_K,
        fetch_k=FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )

    labeled = [
        hit for hit in hits
        if (hit.payload or {}).get("institution_label") in ("privileged", "underrepresented")
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
        "retrieval_method": "mmr",
        "mmr_lambda": MMR_LAMBDA,
        "fetch_k": FETCH_K,
        "labeled_results": n,
        "counts": {
            "privileged": n_priv,
            "underrepresented": n_under,
        },
        "privileged_rate": round(priv_rate, 4),
        "underrepresented_rate": round(under_rate, 4),
        "spd": spd,
        "selection_rate_ratio": ratio,
        "retrieved": [hit_to_dict(hit) for hit in labeled],
    }


def print_table(per_query: list, summary: dict):
    def short(q, n=45):
        return q if len(q) <= n else q[: n - 3] + "..."

    header = f"{'Query':45} {'Priv':>4} {'Under':>5} {'SPD':>7} {'Ratio':>6}"
    sep = "-" * len(header)

    print("\nPer-query MMR results (unknown labels excluded):")
    print(header)
    print(sep)

    for r in per_query:
        ratio_str = f"{r['selection_rate_ratio']:.4f}" if r["selection_rate_ratio"] is not None else " N/A"
        print(
            f"{short(r['query']):45} "
            f"{r['counts']['privileged']:>4} "
            f"{r['counts']['underrepresented']:>5} "
            f"{r['spd']:>+7.4f} "
            f"{ratio_str:>6}"
        )

    print(sep)

    fa = summary["fairlearn_aggregate"]
    cb = summary["corpus_base_rates"]

    print("\nSummary (unknown labels excluded)")
    print(f" Retrieval method: {summary['retrieval_method']}")
    print(f" MMR lambda: {summary['mmr_lambda']}")
    print(f" Fetch-k: {summary['fetch_k']}")
    print(f" Queries run: {summary['num_queries']}")
    print(f" Queries with labeled hits: {summary['queries_with_labeled_hits']}")
    print(f" Queries skipped (all unknown): {summary['num_queries'] - summary['queries_with_labeled_hits']}")
    print()
    print(" Fairlearn aggregate (chance of a corpus paper being retrieved, pooled):")
    print(f" Privileged selection rate: {fa['privileged_selection_rate']:.6f}")
    print(f" Underrepresented selection rate:{fa['underrepresented_selection_rate']:.6f}")
    print(f" Selection rate ratio (priv/under): {fa['selection_rate_ratio']:.4f}" if fa["selection_rate_ratio"] is not None else " Selection rate ratio (priv/under): N/A")
    print(f" SPD: {fa['spd']:+.6f}")
    print()
    print(" Per-query means:")
    print(f" Mean privileged rate: {summary['mean_privileged_rate']:.4f}")
    print(f" Mean underrepresented rate: {summary['mean_underrepresented_rate']:.4f}")
    print(f" Mean selection rate ratio: {summary['mean_selection_rate_ratio']:.4f}" if summary["mean_selection_rate_ratio"] is not None else " Mean selection rate ratio: N/A")
    print(f" Mean SPD: {summary['mean_spd']:+.4f}")
    print()
    print(" Corpus base rates (labeled papers only):")
    print(f" Privileged: {cb['privileged_count']:,} ({cb['privileged_base_rate']:.4f})")
    print(f" Underrepresented: {cb['underrepresented_count']:,} ({cb['underrepresented_base_rate']:.4f})")


def get_corpus_rates(client) -> dict:
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
            label = (point.payload or {}).get("institution_label")
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
    print(f" Privileged: {corpus['privileged_count']:,} ({corpus['privileged_base_rate']:.4f})")
    print(f" Underrepresented:{corpus['underrepresented_count']:,} ({corpus['underrepresented_base_rate']:.4f})")
    print()

    queries = load_queries(EVAL_FILE)
    print(f"Loaded {len(queries)} queries. Running top-{TOP_K} MMR retrieval...\n")

    per_query = []
    for i, q in enumerate(queries, start=1):
        print(f"[{i:>3}/{len(queries)}] {q[:70]}")
        result = measure_query(q, model, client)
        if result is not None:
            per_query.append(result)

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
    y_true = np.zeros_like(y_pred)
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
        "retrieval_method": "mmr",
        "top_k": TOP_K,
        "fetch_k": FETCH_K,
        "mmr_lambda": MMR_LAMBDA,
        "num_queries": len(queries),
        "queries_with_labeled_hits": len(per_query),
        "corpus_base_rates": corpus,
        "fairlearn_aggregate": {
            "privileged_selection_rate": round(float(agg_priv_rate), 6),
            "underrepresented_selection_rate": round(float(agg_under_rate), 6),
            "spd": agg_spd,
            "selection_rate_ratio": agg_ratio,
        },
        "mean_privileged_rate": round(mean(r["privileged_rate"] for r in per_query), 4) if per_query else None,
        "mean_underrepresented_rate": round(mean(r["underrepresented_rate"] for r in per_query), 4) if per_query else None,
        "mean_spd": round(mean(r["spd"] for r in per_query), 4) if per_query else None,
        "mean_selection_rate_ratio": round(mean(ratios), 4) if ratios else None,
    }

    output = {
        "summary": summary,
        "per_query": per_query,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(per_query, summary)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()