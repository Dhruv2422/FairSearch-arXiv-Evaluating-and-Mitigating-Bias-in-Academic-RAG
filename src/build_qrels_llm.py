"""
build_qrels_llm.py — LLM-judged relevance labels.

Gemini grades each (query, paper) pair 0-3 from the title and abstract
only, over a TREC-style pool of the baseline and MMR retrievers' top-30.
Grade 0 judgments are kept as explicit negatives for downstream metrics.

Judgments are cached (data/processed/llm_judgment_cache.json) after every
API call, so the script is safe to interrupt and re-run.

Output: data/eval/qrels_auto.json

Run from inside src/:
    python build_qrels_llm.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retriever import load_model, connect_qdrant, search
from retriever_mmr import search_mmr

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

QUERIES_FILE = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "eval" / "qrels_auto.json"
CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "llm_judgment_cache.json"

MODEL_NAME = "gemini-2.5-flash-lite"
POOL_K = 30          # judged pool depth per retriever
JUDGE_BATCH = 50     # papers judged per API call (whole pool -> 1 call/query)
REQUEST_DELAY = 1.0  # seconds between API calls (paid tier)
RETRY_LIMIT = 8
ABSTRACT_CHARS = 800

JUDGE_SYSTEM_PROMPT = (
    "You are a relevance assessor for an academic search evaluation, judging "
    "arXiv computer science papers against search queries. For each paper, "
    "assign one relevance grade based only on the title and abstract:\n"
    "  3 = directly answers or addresses the query's core topic\n"
    "  2 = substantially relevant; a searcher would find it useful\n"
    "  1 = marginally related; touches the topic only tangentially\n"
    "  0 = not relevant to the query\n"
    "Judge topical relevance only. Ignore paper quality, recency, or venue. "
    "Respond with JSON only: a list of objects with integer fields "
    '"id" and "grade".'
)


def normalize_query(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip().lower()


def load_queries(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for item in raw:
        q = item["query"].strip() if isinstance(item, dict) else str(item).strip()
        if q:
            queries.append(q)
    return queries


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def cache_key(query: str, paper_id: str) -> str:
    return f"{normalize_query(query)}||{paper_id}"


# ---------------------------------------------------------------------------
# Candidate pooling
# ---------------------------------------------------------------------------

def build_pool(query: str, model, client) -> list[dict]:
    """Union of baseline top-30 and MMR top-30, deduped by paper_id."""
    pool = {}
    baseline_hits = search(query, model, client, k=POOL_K)
    mmr_hits = search_mmr(query, model, client, k=POOL_K, fetch_k=100)

    for hit in list(baseline_hits) + list(mmr_hits):
        payload = hit.payload or {}
        pid = payload.get("paper_id")
        if not pid or pid in pool:
            continue
        pool[pid] = {
            "paper_id": pid,
            "title": " ".join(str(payload.get("title", "")).split()),
            "abstract": " ".join(str(payload.get("abstract", "")).split()),
            "category": payload.get("category"),
            "year": payload.get("year"),
            "institution_label": payload.get("institution_label"),
        }
    return list(pool.values())


# ---------------------------------------------------------------------------
# Gemini judging
# ---------------------------------------------------------------------------

def judge_batch(gemini, query: str, papers: list[dict]) -> dict[str, int]:
    """Ask Gemini to grade a batch of papers. Returns {paper_id: grade}."""
    lines = [f"Query: {query}\n\nPapers to judge:"]
    for i, paper in enumerate(papers, start=1):
        abstract = paper["abstract"][:ABSTRACT_CHARS]
        lines.append(f"\n[{i}] Title: {paper['title']}\nAbstract: {abstract}")
    prompt = "\n".join(lines)

    for attempt in range(RETRY_LIMIT):
        try:
            response = gemini.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text)
            grades = {}
            for row in parsed:
                idx = int(row["id"])
                grade = max(0, min(3, int(row["grade"])))
                if 1 <= idx <= len(papers):
                    grades[papers[idx - 1]["paper_id"]] = grade
            if len(grades) == len(papers):
                return grades
            raise ValueError(
                f"judge returned {len(grades)}/{len(papers)} grades"
            )
        except Exception as e:
            # 429 asks for ~50s waits; 503 means the model is overloaded and
            # needs real time to recover — short backoff just burns retries
            msg = str(e)
            if "429" in msg:
                wait = 60
            elif "503" in msg:
                wait = min(30 * (attempt + 1), 120)
            else:
                wait = 2 ** (attempt + 1)
            print(f"    judge error ({e}) — retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Judging failed after {RETRY_LIMIT} attempts: {query[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set — add it to the repo-root .env")

    queries = load_queries(QUERIES_FILE)
    cache = load_cache()
    print(f"Loaded {len(queries)} queries; cache has {len(cache)} judgments")

    print("Loading embedding model and connecting to Qdrant...")
    model = load_model()
    client = connect_qdrant()
    gemini = genai.Client(api_key=api_key)

    qrels = []
    api_calls = 0

    for qi, query in enumerate(queries, start=1):
        pool = build_pool(query, model, client)
        to_judge = [p for p in pool if cache_key(query, p["paper_id"]) not in cache]
        print(f"[{qi:>3}/{len(queries)}] pool={len(pool)} unjudged={len(to_judge)} | {query[:60]}")

        for start in range(0, len(to_judge), JUDGE_BATCH):
            batch = to_judge[start:start + JUDGE_BATCH]
            grades = judge_batch(gemini, query, batch)
            api_calls += 1
            for pid, grade in grades.items():
                cache[cache_key(query, pid)] = grade
            save_cache(cache)
            time.sleep(REQUEST_DELAY)

        judgments = []
        for paper in pool:
            grade = cache.get(cache_key(query, paper["paper_id"]))
            if grade is None:
                continue
            judgments.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "relevance": grade,
                "category": paper["category"],
                "year": paper["year"],
                "institution_label": paper["institution_label"],
                "judgment_source": "llm-gemini-2.5-flash-lite",
            })

        qrels.append({"query": query, "judgments": judgments})

    OUTPUT_FILE.write_text(
        json.dumps(qrels, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(len(q["judgments"]) for q in qrels)
    from collections import Counter
    dist = Counter(j["relevance"] for q in qrels for j in q["judgments"])
    print(f"\nSaved {total} judgments across {len(qrels)} queries to {OUTPUT_FILE}")
    print(f"API calls this run: {api_calls}")
    print(f"Grade distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
