import json
from pathlib import Path
from statistics import mean

from retriever import load_model, connect_qdrant, search

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_FILE = PROJECT_ROOT / 'data' / 'eval' / 'retrieval_eval_queries.json'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
OUTPUT_JSON = RESULTS_DIR / 'retrieval_metrics.json'
TOP_K = 10


def get_doc_id(hit):
    payload = getattr(hit, 'payload', {}) or {}
    for key in ('paper_id', 'id', 'doc_id', 'arxiv_id'):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    if getattr(hit, 'id', None) is not None:
        return str(hit.id)
    raise KeyError(
        'Could not find a document id. Add paper_id/id/doc_id/arxiv_id to the payload during indexing, or use Qdrant point id.'
    )


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    tp = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return tp / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    tp = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return tp / len(relevant_ids)


def evaluate_query(query_text, relevant_ids, model, client, k):
    hits = search(query_text, model, client, k=k)
    retrieved_ids = [get_doc_id(hit) for hit in hits]
    p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
    r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)

    return {
        'query': query_text,
        'relevant_ids': sorted(relevant_ids),
        'retrieved_ids': retrieved_ids,
        'num_relevant': len(relevant_ids),
        'num_retrieved': len(retrieved_ids[:k]),
        'num_relevant_retrieved': sum(
            1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids
        ),
        'precision_at_k': round(p_at_k, 4),
        'recall_at_k': round(r_at_k, 4),
    }


def load_eval_queries(eval_file):
    if not eval_file.exists():
        raise FileNotFoundError(
            f'Missing evaluation file: {eval_file}\n'
            'Create retrieval_eval_queries.json with entries like:\n'
            '[\n'
            '  {"query": "bias in academic search", "relevant_ids": ["1234.5678", "2345.6789"]}\n'
            ']'
        )

    data = json.loads(eval_file.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('Evaluation file must contain a non-empty JSON list.')

    normalized = []
    for i, item in enumerate(data, start=1):
        if 'query' not in item or 'relevant_ids' not in item:
            raise ValueError(f'Entry {i} must contain both "query" and "relevant_ids".')

        query_text = str(item['query']).strip()
        relevant_ids = {str(x) for x in item['relevant_ids']}

        if not query_text:
            raise ValueError(f'Entry {i} has an empty query.')
        if not relevant_ids:
            raise ValueError(f'Entry {i} has no relevant_ids.')

        normalized.append({
            'query': query_text,
            'relevant_ids': relevant_ids,
        })

    return normalized


def print_table(summary):
    per_query = summary['per_query']
    k = summary['top_k']

    # Simple truncated query label for readability
    def short(q, max_len=40):
        return q if len(q) <= max_len else q[: max_len - 3] + "..."

    header = (
        f"{'Query':40}  {'k':>2}  "
        f"{'#rel':>4}  {'#ret':>4}  {'#rel@k':>6}  "
        f"{'P@k':>6}  {'R@k':>6}"
    )
    sep = "-" * len(header)
    print("\nPer-query metrics table:")
    print(header)
    print(sep)
    for item in per_query:
        row = (
            f"{short(item['query']):40}  "
            f"{k:>2}  "
            f"{item['num_relevant']:>4}  "
            f"{item['num_retrieved']:>4}  "
            f"{item['num_relevant_retrieved']:>6}  "
            f"{item['precision_at_k']:>6.4f}  "
            f"{item['recall_at_k']:>6.4f}"
        )
        print(row)
    print(sep)
    print(
        f"{'MEAN':40}  {k:>2}      "
        f"      "
        f"      "
        f"{summary['mean_precision_at_k']:>6.4f}  "
        f"{summary['mean_recall_at_k']:>6.4f}"
    )


def main():
    eval_queries = load_eval_queries(EVAL_FILE)
    model = load_model()
    client = connect_qdrant()

    per_query = [
        evaluate_query(item['query'], item['relevant_ids'], model, client, TOP_K)
        for item in eval_queries
    ]

    summary = {
        'top_k': TOP_K,
        'num_queries': len(per_query),
        'mean_precision_at_k': round(mean(item['precision_at_k'] for item in per_query), 4),
        'mean_recall_at_k': round(mean(item['recall_at_k'] for item in per_query), 4),
        'per_query': per_query,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))
    print(f'\nSaved JSON: {OUTPUT_JSON}')
    print_table(summary)


if __name__ == '__main__':
    main()