"""
sweep_mmr_lambda.py

Runs the Experiment A retrieval audit across a range of MMR lambda values and
collects the fairness and utility metrics at each point, producing the
fairness-utility tradeoff curve.

MMR(d) = lambda * Sim(d, q) - (1 - lambda) * max Sim(d, d')

lambda = 1.0 is pure relevance and should reproduce the baseline retriever;
lower values weight diversity more heavily. Reporting a single lambda says
nothing about the shape of the tradeoff, which is the actual question behind
RQ3.

IMPORTANT — judged-pool coverage. qrels_auto.json was pooled from the baseline
top-30 and the MMR(lambda=0.5) top-30. Documents outside that pool have no
judgment and score as relevance 0, so a lambda whose results fall outside the
pool is penalised by the *evaluation*, not by its own ranking quality. Measured
coverage of the MMR top-10:

    lambda 0.9 -> 100%    lambda 0.5 -> 100%
    lambda 0.7 ->  99%    lambda 0.3 ->  79%

High lambda converges on the baseline ranking, which is fully judged, so
lambda >= 0.5 is safe to evaluate against the existing qrels. Below 0.5 the
results drift out of the pool and the NDCG/MRR numbers become artifacts unless
the pool is topped up first (re-run build_qrels_llm.py over the new runs).
This script therefore refuses to report utility for out-of-pool lambdas unless
--allow-unjudged is passed, and always reports the coverage it measured.

Run from the repository root:
    python experiments/sweep_mmr_lambda.py
    python experiments/sweep_mmr_lambda.py --lambdas 0.1,0.3,0.5,0.7,0.9 --allow-unjudged
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate

# Resolve src/ imports from the repo root so this runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_a_mmr as exp_mmr  # noqa: E402
import evaluate_equalized_odds as eo_mod  # noqa: E402
import evaluate_ndcg_mrr as ndcg_mod  # noqa: E402
from src.retriever import load_model, connect_qdrant  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SWEEP_DIR = RESULTS_DIR / "sweep"
OUTPUT_JSON = RESULTS_DIR / "mmr_lambda_sweep.json"
QRELS_FILE = PROJECT_ROOT / "data" / "eval" / "qrels_auto.json"

DEFAULT_LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Below this, the MMR top-10 drifts outside the judged pool (see module docstring).
POOL_SAFE_LAMBDA = 0.5


def run_retrieval(lam: float, model, client, queries: list[str], corpus: dict) -> dict:
    """
    Execute the audit at one lambda, returning output in exactly the shape
    experiment_a_mmr.py writes — so the existing evaluators consume it unchanged.

    measure_query() reads the lambda from its module global, so we set it here
    rather than duplicating the per-query metric logic and risking divergence
    from the single-lambda pipeline.
    """
    exp_mmr.MMR_LAMBDA = lam

    per_query = []
    for query in queries:
        result = exp_mmr.measure_query(query, model, client)
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
    agg_priv = by_group.get("privileged", 0.0)
    agg_under = by_group.get("underrepresented", 0.0)

    ratios = [r["selection_rate_ratio"] for r in per_query if r["selection_rate_ratio"] is not None]

    summary = {
        "retrieval_method": "mmr",
        "top_k": exp_mmr.TOP_K,
        "fetch_k": exp_mmr.FETCH_K,
        "mmr_lambda": lam,
        "num_queries": len(queries),
        "queries_with_labeled_hits": len(per_query),
        "corpus_base_rates": corpus,
        "fairlearn_aggregate": {
            "privileged_selection_rate": round(float(agg_priv), 6),
            "underrepresented_selection_rate": round(float(agg_under), 6),
            "spd": round(float(agg_priv - agg_under), 6),
            "selection_rate_ratio": round(float(agg_priv / agg_under), 4) if agg_under > 0 else None,
        },
        "mean_privileged_rate": round(mean(r["privileged_rate"] for r in per_query), 4) if per_query else None,
        "mean_spd": round(mean(r["spd"] for r in per_query), 4) if per_query else None,
        "mean_selection_rate_ratio": round(mean(ratios), 4) if ratios else None,
    }
    return {"summary": summary, "per_query": per_query}


def load_labels(client) -> dict[str, str]:
    """
    paper_id -> institution_label, reusing the caller's client.

    evaluate_equalized_odds.load_labels_from_qdrant() opens its own connection,
    which local Qdrant refuses while this script already holds one.
    """
    labels = {}
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=eo_mod.COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["paper_id", "institution_label"],
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            pid = payload.get("paper_id")
            if pid:
                labels[pid] = payload.get("institution_label", "unknown")
        if offset is None:
            break
    return labels


def judged_coverage(run: dict, qrels_by_query: dict) -> float:
    """Fraction of retrieved documents that carry a relevance judgment."""
    judged = total = 0
    for entry in run["per_query"]:
        pool = qrels_by_query.get(ndcg_mod.normalize_query(entry["query"]))
        if pool is None:
            continue
        for hit in entry["retrieved"]:
            total += 1
            if hit.get("paper_id") in pool:
                judged += 1
    return judged / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambdas",
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="Comma-separated MMR lambda values to evaluate.",
    )
    parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help="Report NDCG/MRR even for lambdas whose results fall outside the "
             "judged pool. Off by default because those numbers understate "
             "utility for reasons unrelated to ranking quality.",
    )
    args = parser.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print("Loading model and connecting to Qdrant...")
    model = load_model()
    client = connect_qdrant()

    corpus = exp_mmr.get_corpus_rates(client)
    print(f"  Corpus base rates — privileged {corpus['privileged_base_rate']:.4f}, "
          f"underrepresented {corpus['underrepresented_base_rate']:.4f}")

    queries = exp_mmr.load_queries(exp_mmr.EVAL_FILE)
    qrels_raw = json.loads(QRELS_FILE.read_text())
    qrels_for_eo = qrels_raw
    qrels_for_ndcg = ndcg_mod.load_qrels(QRELS_FILE)
    qrels_by_query = {
        ndcg_mod.normalize_query(e["query"]): {j["paper_id"] for j in e["judgments"]}
        for e in qrels_raw
    }
    labels = load_labels(client)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    points = []

    for lam in lambdas:
        print(f"\n=== lambda = {lam} ===")
        run = run_retrieval(lam, model, client, queries, corpus)

        run_path = SWEEP_DIR / f"experiment_a_mmr_lambda_{lam:.2f}.json"
        run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

        coverage = judged_coverage(run, qrels_by_query)
        utility = ndcg_mod.evaluate_run(run_path, qrels_for_ndcg, queries)
        eo = eo_mod.compute_eo(run, qrels_for_eo, labels)

        in_pool = lam >= POOL_SAFE_LAMBDA or args.allow_unjudged
        point = {
            "mmr_lambda": lam,
            "judged_pool_coverage": round(coverage, 4),
            "utility_trustworthy": lam >= POOL_SAFE_LAMBDA,
            "selection_rate_ratio": run["summary"]["fairlearn_aggregate"]["selection_rate_ratio"],
            "mean_privileged_rate": run["summary"]["mean_privileged_rate"],
            "spd": run["summary"]["fairlearn_aggregate"]["spd"],
            "equalized_odds_difference": eo["equalized_odds_difference"],
            "tpr_privileged": eo["by_group"]["privileged"]["tpr"],
            "tpr_underrepresented": eo["by_group"]["underrepresented"]["tpr"],
            "ndcg@10": utility["mean_ndcg@10"] if in_pool else None,
            "mrr": utility["mean_mrr"] if in_pool else None,
            "run_file": str(run_path.relative_to(PROJECT_ROOT)),
        }
        points.append(point)

        flag = "" if point["utility_trustworthy"] else "   [OUT OF POOL]"
        print(f"  judged coverage      {coverage:.0%}{flag}")
        print(f"  selection rate ratio {point['selection_rate_ratio']}")
        print(f"  equalized odds diff  {point['equalized_odds_difference']}")
        print(f"  NDCG@10 / MRR        {point['ndcg@10']} / {point['mrr']}")

    output = {
        "note": (
            "Fairness-utility tradeoff across MMR lambda. NDCG/MRR are null "
            "where the run falls outside the judged pool (lambda < "
            f"{POOL_SAFE_LAMBDA}) unless --allow-unjudged was passed; see "
            "judged_pool_coverage."
        ),
        "qrels_source": QRELS_FILE.name,
        "corpus_base_rates": corpus,
        "points": points,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'lambda':>8s}{'cover':>8s}{'sel.ratio':>11s}{'EO diff':>10s}{'NDCG@10':>10s}{'MRR':>8s}")
    for p in points:
        n = f"{p['ndcg@10']:.4f}" if p["ndcg@10"] is not None else "  n/a"
        m = f"{p['mrr']:.4f}" if p["mrr"] is not None else "  n/a"
        print(f"{p['mmr_lambda']:8.2f}{p['judged_pool_coverage']:8.0%}"
              f"{p['selection_rate_ratio']:11.4f}{p['equalized_odds_difference']:10.4f}{n:>10s}{m:>8s}")
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
