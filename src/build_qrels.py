import json
from pathlib import Path

from retriever import load_model, connect_qdrant, search


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

QUERIES_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "eval" / "qrels_auto.json"

TOP_K = 10
OVERSAMPLE_K = 30

HIGH_REL_THRESHOLD = 0.60
MED_REL_THRESHOLD = 0.50
LOW_REL_THRESHOLD = 0.40


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path) -> list[str]:
    raw = load_json(path)

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


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip()


def tokenize(text: str) -> set[str]:
    cleaned = normalize_text(text).lower()
    for ch in ",.:;!?()[]{}\"'`/\\|-_":
        cleaned = cleaned.replace(ch, " ")
    return {tok for tok in cleaned.split() if len(tok) > 2}


def lexical_overlap_ratio(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def assign_relevance(score: float, overlap: float) -> int | None:
    if score >= HIGH_REL_THRESHOLD and overlap >= 0.20:
        return 3
    if score >= MED_REL_THRESHOLD and overlap >= 0.10:
        return 2
    if score >= LOW_REL_THRESHOLD:
        return 1
    return None


def build_judgment(hit, query: str) -> dict | None:
    payload = hit.payload or {}
    title = normalize_text(payload.get("title", ""))
    abstract = normalize_text(payload.get("abstract", ""))
    combined_text = f"{title} {abstract}".strip()

    score = float(hit.score)
    overlap = lexical_overlap_ratio(query, combined_text)
    relevance = assign_relevance(score, overlap)

    if relevance is None:
        return None

    return {
        "paper_id": payload.get("paper_id"),
        "title": title,
        "relevance": relevance,
        "score": round(score, 4),
        "lexical_overlap": round(overlap, 4),
        "category": payload.get("category"),
        "year": payload.get("year"),
        "institution_label": payload.get("institution_label"),
        "judgment_source": "auto"
    }


def build_auto_qrels(queries: list[str], model, client) -> list[dict]:
    qrels = []

    for i, query in enumerate(queries, start=1):
        print(f"[{i:>3}/{len(queries)}] {query[:80]}")

        hits = search(query, model, client, k=OVERSAMPLE_K)

        judgments = []
        seen_ids = set()

        for hit in hits:
            payload = hit.payload or {}
            paper_id = payload.get("paper_id")
            title = normalize_text(payload.get("title", ""))

            unique_key = paper_id or title
            if not unique_key or unique_key in seen_ids:
                continue

            judgment = build_judgment(hit, query)
            if judgment is None:
                continue

            judgments.append(judgment)
            seen_ids.add(unique_key)

            if len(judgments) >= TOP_K:
                break

        qrels.append({
            "query": query,
            "judgments": judgments
        })

    return qrels


def main():
    print("Loading queries...")
    queries = load_queries(QUERIES_FILE)

    print("Loading model and connecting to Qdrant...")
    model = load_model()
    client = connect_qdrant()

    print("Building auto-qrels from corpus retrieval...")
    qrels = build_auto_qrels(queries, model, client)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(qrels, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    num_queries = len(qrels)
    num_judgments = sum(len(item["judgments"]) for item in qrels)

    print(f"\nSaved auto qrels file to: {OUTPUT_FILE}")
    print(f"Queries: {num_queries}")
    print(f"Total judgments: {num_judgments}")

    if qrels:
        print("\nExample entry:")
        print(json.dumps(qrels[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()