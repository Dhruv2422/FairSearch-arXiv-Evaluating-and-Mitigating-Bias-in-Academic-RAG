import json
import random
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = (BASE_DIR / "../data/processed/papers.parquet").resolve()
EVAL_FILE = (BASE_DIR / "../data/eval/queries.json").resolve()

TARGET_QUERIES = 100
RANDOM_SEED = 42

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'into', 'is', 'it', 'of', 'on',
    'or', 'that', 'the', 'their', 'this', 'to', 'using', 'with', 'via', 'based', 'toward', 'towards', 'study',
    'approach', 'method', 'methods', 'analysis', 'framework', 'system', 'systems', 'model', 'models', 'learning',
    'deep', 'neural', 'paper', 'new', 'efficient', 'such', 'without', 'within', 'across', 'among', 'our', 'we',
    'show', 'shows', 'showing', 'propose', 'proposes', 'proposed', 'introduce', 'introduces', 'novel'
}

BAD_EDGE_WORDS = {
    'and', 'or', 'of', 'for', 'with', 'without', 'via', 'based', 'toward', 'towards',
    'such', 'than', 'from', 'into', 'using', 'method', 'methods', 'approach', 'analysis'
}

GENERIC_QUERY_PATTERNS = {
    'research topics',
    'applications',
    'methods',
    'research'
}

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    df = pd.read_parquet(path)
    required = {"paper_id", "title", "abstract", "category", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.fillna("")

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def clean_phrase(text: str) -> str:
    text = normalize_space(text.lower())
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return [t for t in clean_phrase(text).split() if len(t) > 2 and t not in STOPWORDS]

def title_to_query(title: str) -> str:
    title = normalize_space(title)
    title = re.sub(r"[:;|]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.rstrip("?.!")

def sentence_split(text: str):
    text = normalize_space(text)
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def is_good_query(query: str) -> bool:
    q = normalize_space(query)
    q_clean = clean_phrase(q)
    tokens = q_clean.split()

    if len(tokens) < 3:
        return False
    if len(q) < 12 or len(q) > 120:
        return False
    if tokens[0] in BAD_EDGE_WORDS or tokens[-1] in BAD_EDGE_WORDS:
        return False
    if q_clean in GENERIC_QUERY_PATTERNS:
        return False

    content_tokens = [t for t in tokens if t not in STOPWORDS]
    return len(content_tokens) >= 3

def extract_strong_phrases(text: str, max_phrases: int = 4):
    tokens = tokenize(text)
    phrases = []

    for n in (4, 3):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            if phrase not in phrases and is_good_query(phrase):
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases

    return phrases

def abstract_to_query_candidates(abstract: str, max_candidates: int = 2):
    candidates = []
    for sent in sentence_split(abstract)[:3]:
        cleaned = clean_phrase(sent)
        tokens = cleaned.split()
        if len(tokens) < 6:
            continue

        phrase = " ".join(tokens[:8])
        if is_good_query(phrase):
            candidates.append(phrase)

        for strong_phrase in extract_strong_phrases(sent, max_phrases=2):
            if strong_phrase not in candidates:
                candidates.append(strong_phrase)

        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]

def generate_candidate_queries(row):
    title = normalize_space(row["title"])
    abstract = normalize_space(row["abstract"])
    candidates = []

    if title:
        title_query = title_to_query(title)
        if is_good_query(title_query):
            candidates.append(title_query)
        candidates.extend(extract_strong_phrases(title, max_phrases=2))

    if abstract:
        candidates.extend(abstract_to_query_candidates(abstract, max_candidates=2))

    cleaned = []
    seen = set()

    for q in candidates:
        q = normalize_space(q)
        q_norm = q.lower()
        if not is_good_query(q):
            continue
        if q_norm in seen:
            continue
        seen.add(q_norm)
        cleaned.append(q)

    return cleaned

def dedupe_queries(queries):
    seen = set()
    deduped = []
    for q in queries:
        q_norm = q.strip().lower()
        if q_norm in seen:
            continue
        seen.add(q_norm)
        deduped.append(q)
    return deduped

def build_queries(df: pd.DataFrame, target_queries: int = TARGET_QUERIES):
    random.seed(RANDOM_SEED)
    rows = df.sample(frac=1, random_state=RANDOM_SEED).to_dict(orient="records")
    queries = []

    for row in rows:
        generated = generate_candidate_queries(row)
        if not generated:
            continue

        queries.extend(generated)

        if len(queries) >= target_queries * 2:
            break

    queries = dedupe_queries(queries)

    if len(queries) < target_queries:
        raise ValueError(
            f"Only generated {len(queries)} unique high-quality queries; need at least {target_queries}."
        )

    return queries[:target_queries]

def save_queries(path: Path, queries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    print(f"Loading processed papers from: {DATA_FILE}")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} papers")

    print("Generating queries only...")
    queries = build_queries(df, target_queries=TARGET_QUERIES)

    save_queries(EVAL_FILE, queries)

    print(f"Saved {len(queries)} queries to: {EVAL_FILE}")
    print("Example query:")
    print(queries[0])

if __name__ == "__main__":
    main()