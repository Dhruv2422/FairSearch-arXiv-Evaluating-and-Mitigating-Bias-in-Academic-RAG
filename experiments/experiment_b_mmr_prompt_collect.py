"""
Experiment B: Generative Faithfulness - MMR + Prompt Data Collection

Purpose:
Runs the MMR-based RAG pipeline using the perspective-balancing
system prompt.

Pipeline:
1. Load contradictory query set.
2. Retrieve papers using MMR re-ranking from Qdrant.
3. Generate Gemini summaries using the balanced system prompt.
4. Save retrieved documents and generated summaries.

Output:
experiment_b_raw_mmr_prompt_results.json
"""

import json
import time

from src.retriever import load_model, connect_qdrant, search_mmr
from src.generator import generate, BALANCED_SYSTEM_PROMPT

QUERY_FILE = "../data/eval/contradictory_queries.json"
OUTPUT_FILE = "../data/results/experiment_b_raw_mmr_prompt_results.json"

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
            # Retrieve top-10 papers using the same MMR settings
            hits = search_mmr(
                q["query"], embedding_model, client,
                k=10, fetch_k=50, lambda_mult=0.5
            )

            print(f"Retrieved {len(hits)} MMR-ranked papers")
            print("Generating Gemini response with balanced prompt...")

            answer = generate(
                q["query"],
                hits,
                system_prompt=BALANCED_SYSTEM_PROMPT
            )

            result = {
                "retrieval_method": "mmr",
                "prompt_method": "balanced",
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
            save_results(experiment_results, OUTPUT_FILE)

            print("Saved result successfully")
            time.sleep(5)

        except Exception as e:
            print(f"ERROR processing query {q['id']}: {e}")
            save_results(experiment_results, OUTPUT_FILE)
            time.sleep(30)

    print("\n==============================")
    print("Experiment B MMR + Prompt complete!")
    print(f"Saved {len(experiment_results)} successful queries")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()