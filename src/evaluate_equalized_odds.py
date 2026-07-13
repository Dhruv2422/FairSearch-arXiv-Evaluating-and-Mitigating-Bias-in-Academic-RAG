"""
evaluate_equalized_odds.py — Equalized Odds for the retrieval bias audit.

Treats retrieval as a binary classifier over the judged pool in
qrels_auto.json (built by build_qrels_llm.py: LLM-judged, pooled from the
baseline and MMR retrievers' top-30, with explicit grade-0 negatives): for
each query, a judged paper is "positive" if relevant (relevance >=
REL_THRESHOLD) and "predicted positive" if it appears in the retrieved
top-k from experiment A. Computes, per institution group:

  TPR = P(retrieved | relevant)       — true positive rate
  FPR = P(retrieved | not relevant)   — false positive rate

Equalized Odds holds when both rates are equal across groups. We report the
gaps (privileged minus underrepresented) and Fairlearn's aggregate measures
(true_positive_rate / false_positive_rate per group via MetricFrame).

Institution labels are joined live from Qdrant, NOT taken from the snapshot
frozen inside qrels_auto.json, so results always reflect current labels.

Caveat (report this alongside any numbers): judgments are LLM-generated
(Gemini), not human relevance labels.

Run from inside src/:
    python evaluate_equalized_odds.py
"""

import json
from pathlib import Path

import numpy as np
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate

from retriever import connect_qdrant

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

QRELS_FILE = PROJECT_ROOT / "data" / "eval" / "qrels_auto.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "experiment_a_results.json"
MMR_RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "experiment_a_mmr_results.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results" / "equalized_odds_results.json"

COLLECTION = "fairsearch_arxiv"
REL_THRESHOLD = 2  # judgment grades 0-3; grade >= 2 counts as relevant


def normalize_query(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip().lower()


def load_labels_from_qdrant() -> dict[str, str]:
    """paper_id -> institution_label, straight from the current index."""
    client = connect_qdrant()
    labels = {}
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
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


def compute_eo(results: dict, qrels: list, labels: dict) -> dict:
    """Equalized Odds for one retrieval run against the judged pool."""
    retrieved_by_query = {
        normalize_query(r["query"]): {
            hit["paper_id"] for hit in r.get("retrieved", []) if hit.get("paper_id")
        }
        for r in results.get("per_query", [])
    }

    # One row per (query, judged paper): group, relevant?, retrieved?
    groups, y_true, y_pred = [], [], []
    skipped_queries = 0
    skipped_unlabeled = 0

    for entry in qrels:
        qnorm = normalize_query(entry["query"])
        retrieved = retrieved_by_query.get(qnorm)
        if retrieved is None:
            skipped_queries += 1
            continue

        for judgment in entry.get("judgments", []):
            pid = judgment.get("paper_id")
            if not pid:
                continue

            group = labels.get(pid, "unknown")
            if group not in ("privileged", "underrepresented"):
                skipped_unlabeled += 1
                continue

            groups.append(group)
            y_true.append(1 if judgment.get("relevance", 0) >= REL_THRESHOLD else 0)
            y_pred.append(1 if pid in retrieved else 0)

    groups = np.array(groups)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mf = MetricFrame(
        metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    by_group = mf.by_group

    tpr_priv = float(by_group.loc["privileged", "tpr"])
    tpr_under = float(by_group.loc["underrepresented", "tpr"])
    fpr_priv = float(by_group.loc["privileged", "fpr"])
    fpr_under = float(by_group.loc["underrepresented", "fpr"])

    tpr_gap = round(tpr_priv - tpr_under, 4)
    fpr_gap = round(fpr_priv - fpr_under, 4)
    # Standard single-number summary: worst of the two gaps
    eo_difference = round(max(abs(tpr_gap), abs(fpr_gap)), 4)

    def group_counts(g):
        mask = groups == g
        return {
            "judged": int(mask.sum()),
            "relevant": int(y_true[mask].sum()),
            "retrieved": int(y_pred[mask].sum()),
        }

    return {
        "counts": {
            "privileged": group_counts("privileged"),
            "underrepresented": group_counts("underrepresented"),
            "skipped_unlabeled_judgments": skipped_unlabeled,
            "skipped_queries_missing_from_results": skipped_queries,
        },
        "by_group": {
            "privileged": {"tpr": round(tpr_priv, 4), "fpr": round(fpr_priv, 4)},
            "underrepresented": {"tpr": round(tpr_under, 4), "fpr": round(fpr_under, 4)},
        },
        "tpr_gap_priv_minus_under": tpr_gap,
        "fpr_gap_priv_minus_under": fpr_gap,
        "equalized_odds_difference": eo_difference,
    }


def print_run(name: str, run: dict):
    bg = run["by_group"]
    c = run["counts"]
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  Judged papers used:  privileged={c['privileged']['judged']}, "
          f"underrepresented={c['underrepresented']['judged']}")
    print(f"                    {'TPR':>8}  {'FPR':>8}")
    print(f"  Privileged        {bg['privileged']['tpr']:>8.4f}  {bg['privileged']['fpr']:>8.4f}")
    print(f"  Underrepresented  {bg['underrepresented']['tpr']:>8.4f}  {bg['underrepresented']['fpr']:>8.4f}")
    print(f"  TPR gap (priv - under):        {run['tpr_gap_priv_minus_under']:+.4f}")
    print(f"  FPR gap (priv - under):        {run['fpr_gap_priv_minus_under']:+.4f}")
    print(f"  Equalized Odds difference:     {run['equalized_odds_difference']:.4f}  (0 = parity)")


def main():
    qrels = json.loads(QRELS_FILE.read_text(encoding="utf-8"))

    print("Loading institution labels from Qdrant...")
    labels = load_labels_from_qdrant()

    runs = {}
    run_files = {
        "baseline": RESULTS_FILE,
        "mmr": MMR_RESULTS_FILE,
    }
    for name, path in run_files.items():
        if not path.exists():
            print(f"  ({name}: {path.name} not found — skipping)")
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        runs[name] = compute_eo(results, qrels, labels)

    output = {
        "relevance_threshold": REL_THRESHOLD,
        "qrels_source": str(QRELS_FILE.name),
        "note": (
            "Judged pool from build_qrels_llm.py: Gemini-judged relevance "
            "(not human labels), candidates pooled from baseline and MMR "
            "top-30. Institution labels joined live from Qdrant."
        ),
        "runs": runs,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    print("\nEqualized Odds (retrieval as binary classifier over judged papers)")
    print("===================================================================")
    for name, run in runs.items():
        print_run(name, run)

    if "baseline" in runs and "mmr" in runs:
        delta = round(
            runs["mmr"]["equalized_odds_difference"]
            - runs["baseline"]["equalized_odds_difference"], 4
        )
        print(f"\n  EO difference change (mmr - baseline): {delta:+.4f}")

    print("\n  TPR gap > 0 → relevant privileged papers are retrieved more reliably")
    print("  FPR gap > 0 → non-relevant privileged papers still get retrieved more")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
