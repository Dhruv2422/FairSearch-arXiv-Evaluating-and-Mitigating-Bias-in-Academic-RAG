import json
from pathlib import Path

from retriever import load_model, connect_qdrant, search

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_FILE = PROJECT_ROOT / 'data' / 'eval' / 'retrieval_eval_queries.json'
TOP_K = 10


def get_doc_id(hit):
    payload = getattr(hit, 'payload', {}) or {}
    for key in ('paper_id', 'id', 'doc_id', 'arxiv_id'):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    if getattr(hit, 'id', None) is not None:
        return str(hit.id)
    raise KeyError(
        'Could not find a stable document id. Add paper_id/id/doc_id/arxiv_id to the payload during indexing, or use Qdrant point id.'
    )


def load_existing_eval_file(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Existing eval file must contain a JSON list.')
    return data


def save_eval_file(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding='utf-8')


def print_hits(hits):
    print('\nTop results:\n')
    for rank, hit in enumerate(hits, start=1):
        payload = getattr(hit, 'payload', {}) or {}
        doc_id = get_doc_id(hit)
        title = payload.get('title', 'N/A')
        category = payload.get('category', 'N/A')
        year = payload.get('year', 'N/A')
        score = getattr(hit, 'score', None)
        score_text = f'{score:.4f}' if isinstance(score, (int, float)) else 'N/A'
        print(f'[{rank}] id={doc_id}')
        print(f'    score={score_text}')
        print(f'    title={title}')
        print(f'    category={category} | year={year}')
        print('-' * 70)


def parse_relevance_input(user_text, hits):
    user_text = user_text.strip()
    if not user_text:
        return []

    tokens = [token.strip() for token in user_text.split(',') if token.strip()]
    selected_ids = []

    hit_ids = [get_doc_id(hit) for hit in hits]

    for token in tokens:
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(hits):
                selected_ids.append(hit_ids[idx - 1])
                continue
        if token in hit_ids:
            selected_ids.append(token)
            continue
        raise ValueError(
            f'Invalid selection: {token}. Use ranks like 1,3,5 or explicit document ids shown above.'
        )

    return sorted(set(selected_ids))


def query_exists(entries, query):
    return any(str(item.get('query', '')).strip().lower() == query.strip().lower() for item in entries)


def main():
    print('Interactive helper for building retrieval_eval_queries.json')
    print(f'Eval file: {EVAL_FILE}')
    print(f'Default top-k: {TOP_K}\n')

    entries = load_existing_eval_file(EVAL_FILE)
    model = load_model()
    client = connect_qdrant()

    while True:
        query = input('Enter a query (or press Enter to stop): ').strip()
        if not query:
            break

        if query_exists(entries, query):
            overwrite = input('This query already exists in the eval file. Overwrite it? [y/N]: ').strip().lower()
            if overwrite != 'y':
                print('Skipping existing query.\n')
                continue
            entries = [item for item in entries if str(item.get('query', '')).strip().lower() != query.lower()]

        k_text = input(f'How many results to inspect? [default {TOP_K}]: ').strip()
        k = int(k_text) if k_text else TOP_K

        hits = search(query, model, client, k=k)
        if not hits:
            print('No results returned for this query.\n')
            continue

        print_hits(hits)
        print('Mark relevant results by rank numbers or doc ids, separated by commas.')
        print('Example: 1,3,5  OR  1234.5678,2345.6789')
        selected = input('Relevant selections: ')

        try:
            relevant_ids = parse_relevance_input(selected, hits)
        except ValueError as exc:
            print(f'Error: {exc}\n')
            continue

        if not relevant_ids:
            confirm = input('No relevant ids selected. Save this query with an empty relevant_ids list? [y/N]: ').strip().lower()
            if confirm != 'y':
                print('Query not saved.\n')
                continue

        entry = {
            'query': query,
            'relevant_ids': relevant_ids,
        }
        entries.append(entry)
        save_eval_file(EVAL_FILE, entries)

        print('\nSaved entry:')
        print(json.dumps(entry, indent=2))
        print(f'Updated eval file: {EVAL_FILE}\n')

    print('\nDone.')
    print(f'Total queries saved: {len(entries)}')


if __name__ == '__main__':
    main()