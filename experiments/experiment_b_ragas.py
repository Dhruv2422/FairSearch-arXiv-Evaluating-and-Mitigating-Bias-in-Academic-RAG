import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

# Input files
INPUT_FILE_1 = "../data/results/experiment_b_raw_results.json"
INPUT_FILE_2 = "../data/results/experiment_b_raw_mmr_results.json"
INPUT_FILE_3 = "../data/results/experiment_b_raw_mmr_prompt_results.json"

# Combined output
OUTPUT_FILE = "../data/results/experiment_b_ragas_comparison.json"

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

llm = llm_factory("gpt-4o-mini", client=client, max_tokens=4000)
embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=client
)

answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
faithfulness = Faithfulness(llm=llm)


async def score(result):
    contexts = [doc["abstract"] for doc in result["retrieved_documents"]]

    relevancy = await answer_relevancy.ascore(
        user_input=result["query"],
        response=result["generated_summary"]
    )

    faithful = await faithfulness.ascore(
        user_input=result["query"],
        response=result["generated_summary"],
        retrieved_contexts=contexts
    )

    return {
        "answer_relevancy": relevancy.value,
        "faithfulness": faithful.value
    }


def print_aggregate_results(results):
    n = len(results)
    if n == 0:
        return

    metrics = {
        "Answer Relevancy": "answer_relevancy",
        "Faithfulness": "faithfulness"
    }

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(
        f"{'Metric':<25}"
        f"{'Baseline':>15}"
        f"{'MMR':>15}"
        f"{'MMR + Prompt':>18}"
    )
    print("-" * 80)

    for name, key in metrics.items():
        baseline = sum(r["baseline"][key] for r in results) / n
        mmr = sum(r["mmr"][key] for r in results) / n
        prompt = sum(r["mmr_prompt"][key] for r in results) / n

        print(
            f"{name:<25}"
            f"{baseline:>15.4f}"
            f"{mmr:>15.4f}"
            f"{prompt:>18.4f}"
        )

    print("-" * 80)
    print(f"Queries scored: {n}")
    print("=" * 80)


async def main():
    with open(INPUT_FILE_1, encoding="utf-8") as f:
        baseline = json.load(f)

    with open(INPUT_FILE_2, encoding="utf-8") as f:
        mmr = json.load(f)

    with open(INPUT_FILE_3, encoding="utf-8") as f:
        mmr_prompt = json.load(f)

    results = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)

    print(f"Baseline:      {len(baseline)} queries")
    print(f"MMR:           {len(mmr)} queries")
    print(f"MMR + Prompt:  {len(mmr_prompt)} queries")
    print(f"Existing RAGAS results: {len(results)}")

    # Index existing results by ID so we can update them.
    existing = {r["id"]: r for r in results}

    for i, (base, mmr_result, prompt_result) in enumerate(
        zip(baseline, mmr, mmr_prompt), 1
    ):
        if not (
            base["id"] == mmr_result["id"] == prompt_result["id"]
        ):
            raise ValueError(f"ID mismatch at query {i}")

        query_id = base["id"]

        if query_id in existing:
            result = existing[query_id]
        else:
            result = {
                "id": query_id,
                "category": base["category"],
                "query": base["query"]
            }
            existing[query_id] = result
            results.append(result)

        print(f"\nScoring {i}/{len(baseline)}: {base['query']}")

        # Baseline and MMR are preserved if already scored.
        if "baseline" not in result:
            print("  Scoring Baseline...")
            result["baseline"] = await score(base)
        else:
            print("  Baseline already scored — skipping")

        if "mmr" not in result:
            print("  Scoring MMR...")
            result["mmr"] = await score(mmr_result)
        else:
            print("  MMR already scored — skipping")

        # Only the new MMR + Prompt condition should normally be scored.
        if "mmr_prompt" not in result:
            print("  Scoring MMR + Prompt...")
            result["mmr_prompt"] = await score(prompt_result)
        else:
            print("  MMR + Prompt already scored — skipping")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print(
            f"  Baseline:     "
            f"AR={result['baseline']['answer_relevancy']:.4f}, "
            f"F={result['baseline']['faithfulness']:.4f}"
        )
        print(
            f"  MMR:          "
            f"AR={result['mmr']['answer_relevancy']:.4f}, "
            f"F={result['mmr']['faithfulness']:.4f}"
        )
        print(
            f"  MMR + Prompt: "
            f"AR={result['mmr_prompt']['answer_relevancy']:.4f}, "
            f"F={result['mmr_prompt']['faithfulness']:.4f}"
        )

    print_aggregate_results(results)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())