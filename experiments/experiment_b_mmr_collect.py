"""
Experiment B: Generative Faithfulness - MMR Data Collection

Purpose:
This script runs the MMR-based RAG pipeline on contradictory research
queries to collect data for synthesis neutrality and RAG quality analysis.

Pipeline:
1. Load contradictory query set.
2. Retrieve papers using MMR re-ranking from Qdrant.
3. Generate Gemini summaries using retrieved abstracts.
4. Save retrieved documents and generated summaries.

Output:
experiment_b_raw_mmr_results.json
"""

import json
import time

import sys
from pathlib import Path

# Resolve paths and imports from the repo root so these run from any working
# directory. Without this the module imports below need cwd=repo root while the
# data paths need cwd=experiments/, which only lined up under PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import load_model, connect_qdrant
from src.retriever_mmr import search_mmr
from src.generator import generate

QUERY_FILE = PROJECT_ROOT / "data" / "eval" / "contradictory_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results" / "experiment_b_raw_mmr_results.json"


def load_queries(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main():

    print("Loading queries...")
    queries = load_queries(QUERY_FILE)
    print(f"Loaded {len(queries)} queries")

    print("Loading embedding model...")
    embedding_model = load_model()

    print("Connecting to Qdrant...")
    client = connect_qdrant()

    experiment_results = []

    for q in queries:

        print("\n==============================")
        print(f"Query {q['id']}: {q['query']}")

        try:

            # 1. Retrieve top-10 papers using MMR
            hits = search_mmr(
                q["query"],
                embedding_model,
                client,
                k=10,
                fetch_k=50,
                lambda_mult=0.5
            )

            print(f"Retrieved {len(hits)} MMR-ranked papers")

            # 2. Generate Gemini summary
            print("Generating Gemini response...")

            answer = generate(
                q["query"],
                hits
            )

            # 3. Store results
            result = {
                "retrieval_method": "mmr",
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],

                "retrieved_documents": [
                    {
                        "title": hit.payload["title"],
                        "abstract": hit.payload["abstract"],
                        "score": hit.score
                    }
                    for hit in hits
                ],

                "generated_summary": answer
            }

            experiment_results.append(result)

            # Save after every successful query
            save_results(
                experiment_results,
                OUTPUT_FILE
            )

            print("Saved result successfully")

            # Gemini rate limit protection
            time.sleep(5)


        except Exception as e:

            print(f"ERROR processing query {q['id']}: {e}")

            # Save progress after failure
            save_results(
                experiment_results,
                OUTPUT_FILE
            )

            time.sleep(30)


    print("\n==============================")
    print("Experiment B MMR retrieval + generation complete!")
    print(f"Saved {len(experiment_results)} successful queries")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()