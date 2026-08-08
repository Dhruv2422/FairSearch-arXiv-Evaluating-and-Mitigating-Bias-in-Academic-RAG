import json
import math
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

BASELINE_FILE = PROJECT_ROOT / "data" / "results" / "experiment_a_results.json"
MMR_FILE = PROJECT_ROOT / "data" / "results" / "experiment_a_mmr_results.json"
QRELS_FILE = PROJECT_ROOT / "data" / "eval" / "qrels_auto.json"
QUERIES_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results" / "ndcg_mrr_results.json"

TOP_K = 10


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip()


def normalize_query(text: str) -> str:
    return normalize_text(text).lower()


def load_queries(path: Path) -> list[str]:
    raw = load_json(path)

    if not isinstance(raw, list):
        raise ValueError("retrieval_eval_queries.json must contain a JSON list.")

    queries = []
    for item in raw:
        if isinstance(item, str):
            q = normalize_text(item)
        elif isinstance(item, dict) and "query" in item:
            q = normalize_text(item["query"])
        else:
            raise ValueError("Each query must be a string or an object with a 'query' field.")

        if q:
            queries.append(q)

    return queries


def load_qrels(path: Path):
    raw = load_json(path)

    if not isinstance(raw, list):
        raise ValueError("qrels.json must contain a JSON list.")

    qrels = {}

    if not raw:
        return qrels

    first = raw[0]

    # Flat format:
    # [{"query": "...", "paper_id": "...", "relevance": 3}, ...]
    if isinstance(first, dict) and "paper_id" in first and "relevance" in first:
        for row in raw:
            query = normalize_query(row["query"])
            paper_id = row.get("paper_id")
            relevance = row.get("relevance")

            if relevance is None:
                continue

            if paper_id:
                key = str(paper_id).strip()
            else:
                title = row.get("title")
                if not title:
                    continue
                key = normalize_text(title).lower()

            qrels.setdefault(query, {})[key] = int(relevance)

        return qrels

    # Nested judgments format:
    # [{"query": "...", "judgments": [{"paper_id": "...", "doc_id": "...", "relevance": 2}, ...]}, ...]
    if isinstance(first, dict) and "query" in first and "judgments" in first:
        for row in raw:
            query = normalize_query(row["query"])
            judgments = row.get("judgments", [])

            for cand in judgments:
                relevance = cand.get("relevance")
                if relevance is None:
                    continue

                paper_id = cand.get("paper_id")
                doc_id = cand.get("doc_id")
                title = cand.get("title")

                if paper_id:
                    key = str(paper_id).strip()
                elif doc_id:
                    key = str(doc_id).strip()
                elif title:
                    key = normalize_text(title).lower()
                else:
                    continue

                qrels.setdefault(query, {})[key] = int(relevance)

        return qrels

    # Older nested candidates format:
    # [{"query": "...", "candidates": [{"paper_id": "...", "doc_id": "...", "relevance": 2}, ...]}, ...]
    if isinstance(first, dict) and "query" in first and "candidates" in first:
        for row in raw:
            query = normalize_query(row["query"])
            candidates = row.get("candidates", [])

            for cand in candidates:
                relevance = cand.get("relevance")
                if relevance is None:
                    continue

                paper_id = cand.get("paper_id")
                doc_id = cand.get("doc_id")
                title = cand.get("title")

                if paper_id:
                    key = str(paper_id).strip()
                elif doc_id:
                    key = str(doc_id).strip()
                elif title:
                    key = normalize_text(title).lower()
                else:
                    continue

                qrels.setdefault(query, {})[key] = int(relevance)

        return qrels

    raise ValueError("Unsupported qrels format.")


def dcg_at_k(relevances, k=10):
    score = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        score += rel / math.log2(rank + 1)
    return score


def ndcg_at_k(relevances, qrel_values, k=10):
    if not relevances:
        return None

    dcg = dcg_at_k(relevances, k)
    ideal = sorted(qrel_values, reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return None

    return dcg / idcg


def reciprocal_rank(relevances):
    for rank, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def doc_id_from_hit(hit: dict) -> str | None:
    if hit.get("paper_id"):
        return str(hit["paper_id"]).strip()

    if hit.get("doc_id"):
        return str(hit["doc_id"]).strip()

    if hit.get("title"):
        return normalize_text(hit["title"]).lower()

    return None


def build_run_lookup(data: dict) -> dict:
    lookup = {}
    for item in data.get("per_query", []):
        query = item.get("query")
        if not query:
            continue
        lookup[normalize_query(query)] = item
    return lookup


def evaluate_run(results_path: Path, qrels: dict, canonical_queries: list[str]):
    data = load_json(results_path)
    run_lookup = build_run_lookup(data)

    scored = []
    ndcgs = []
    mrrs = []
    missing_queries = []
    queries_with_no_judgments = []

    for query in canonical_queries:
        qkey = normalize_query(query)

        if qkey not in qrels:
            missing_queries.append(query)
            continue

        query_qrels = qrels[qkey]
        if not query_qrels:
            queries_with_no_judgments.append(query)
            continue

        run_item = run_lookup.get(qkey)
        if run_item is None:
            missing_queries.append(query)
            continue

        retrieved = run_item.get("retrieved", [])[:TOP_K]

        relevances = []
        for hit in retrieved:
            key = doc_id_from_hit(hit)
            relevances.append(query_qrels.get(key, 0) if key is not None else 0)

        qrel_values = list(query_qrels.values())

        ndcg = ndcg_at_k(relevances, qrel_values, TOP_K)
        mrr = reciprocal_rank(relevances)

        scored.append({
            "query": query,
            "relevances": relevances,
            "num_judged_docs_for_query": len(query_qrels),
            "ndcg@10": round(ndcg, 4) if ndcg is not None else None,
            "mrr": round(mrr, 4),
        })

        if ndcg is not None:
            ndcgs.append(ndcg)
        mrrs.append(mrr)

    return {
        "input_file": str(results_path),
        "queries_total_in_canonical_list": len(canonical_queries),
        "queries_scored": len(scored),
        "queries_missing_from_run_or_qrels": len(missing_queries),
        "queries_with_no_judgments": len(queries_with_no_judgments),
        "mean_ndcg@10": round(mean(ndcgs), 4) if ndcgs else None,
        "mean_mrr": round(mean(mrrs), 4) if mrrs else None,
        "per_query": scored,
        "debug": {
            "missing_queries_sample": missing_queries[:10],
            "no_judgments_sample": queries_with_no_judgments[:10],
        },
    }


def main():
    canonical_queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    baseline = evaluate_run(BASELINE_FILE, qrels, canonical_queries)
    mmr = evaluate_run(MMR_FILE, qrels, canonical_queries)

    comparison = {
        "ndcg@10_delta": (
            round(mmr["mean_ndcg@10"] - baseline["mean_ndcg@10"], 4)
            if baseline["mean_ndcg@10"] is not None and mmr["mean_ndcg@10"] is not None
            else None
        ),
        "mrr_delta": (
            round(mmr["mean_mrr"] - baseline["mean_mrr"], 4)
            if baseline["mean_mrr"] is not None and mmr["mean_mrr"] is not None
            else None
        ),
    }

    output = {
        "queries_file": str(QUERIES_FILE),
        "qrels_file": str(QRELS_FILE),
        "baseline": baseline,
        "mmr": mmr,
        "comparison": comparison,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nUtility evaluation")
    print("------------------")
    print(f"Canonical queries loaded: {len(canonical_queries)}")
    print(f"Qrels queries loaded:     {len(qrels)}")
    print()
    print(f"Baseline queries scored:  {baseline['queries_scored']}")
    print(f"Baseline mean NDCG@10:    {baseline['mean_ndcg@10']}")
    print(f"Baseline mean MRR:        {baseline['mean_mrr']}")
    print()
    print(f"MMR queries scored:       {mmr['queries_scored']}")
    print(f"MMR mean NDCG@10:         {mmr['mean_ndcg@10']}")
    print(f"MMR mean MRR:             {mmr['mean_mrr']}")
    print()
    print(f"Delta NDCG@10:            {comparison['ndcg@10_delta']}")
    print(f"Delta MRR:                {comparison['mrr_delta']}")
    print()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()