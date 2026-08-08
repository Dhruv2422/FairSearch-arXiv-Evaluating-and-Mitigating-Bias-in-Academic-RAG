import json
import math
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_FILE = PROJECT_ROOT / "data" / "results" / "experiment_a_results.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results" / "fairness_metrics_from_results.json"


def exposure_at_rank(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def compute_query_metrics(query_result: dict) -> dict | None:
    retrieved = query_result.get("retrieved", [])
    labeled = [
        hit for hit in retrieved
        if hit.get("institution_label") in ("privileged", "underrepresented")
    ]

    if not labeled:
        return None

    n = len(labeled)
    priv_count = sum(1 for hit in labeled if hit["institution_label"] == "privileged")
    under_count = sum(1 for hit in labeled if hit["institution_label"] == "underrepresented")

    priv_rate = priv_count / n
    under_rate = under_count / n

    dp_ratio = round(under_rate / priv_rate, 4) if priv_rate > 0 else None
    spd = round(priv_rate - under_rate, 4)

    priv_exposure = 0.0
    under_exposure = 0.0

    for rank, hit in enumerate(labeled, start=1):
        exp = exposure_at_rank(rank)
        if hit["institution_label"] == "privileged":
            priv_exposure += exp
        else:
            under_exposure += exp

    total_exposure = priv_exposure + under_exposure
    priv_exposure_share = priv_exposure / total_exposure if total_exposure > 0 else 0.0
    under_exposure_share = under_exposure / total_exposure if total_exposure > 0 else 0.0
    exposure_ratio = round(under_exposure / priv_exposure, 4) if priv_exposure > 0 else None
    exposure_difference = round(priv_exposure_share - under_exposure_share, 4)

    return {
        "query": query_result["query"],
        "top_k": query_result.get("top_k", n),
        "labeled_results": n,
        "counts": {
            "privileged": priv_count,
            "underrepresented": under_count,
        },
        "demographic_parity": {
            "privileged_selection_rate": round(priv_rate, 4),
            "underrepresented_selection_rate": round(under_rate, 4),
            "selection_rate_ratio": dp_ratio,
            "statistical_parity_difference": spd,
        },
        "exposure_diversity": {
            "privileged_exposure": round(priv_exposure, 4),
            "underrepresented_exposure": round(under_exposure, 4),
            "privileged_exposure_share": round(priv_exposure_share, 4),
            "underrepresented_exposure_share": round(under_exposure_share, 4),
            "exposure_ratio": exposure_ratio,
            "exposure_difference": exposure_difference,
        },
    }


def main():
    data = json.loads(INPUT_FILE.read_text())
    per_query_input = data.get("per_query", [])

    per_query_output = []
    for item in per_query_input:
        result = compute_query_metrics(item)
        if result is not None:
            per_query_output.append(result)

    if not per_query_output:
        raise ValueError("No labeled retrieved results found in input file.")

    dp_ratios = [
        q["demographic_parity"]["selection_rate_ratio"]
        for q in per_query_output
        if q["demographic_parity"]["selection_rate_ratio"] is not None
    ]
    exposure_ratios = [
        q["exposure_diversity"]["exposure_ratio"]
        for q in per_query_output
        if q["exposure_diversity"]["exposure_ratio"] is not None
    ]

    summary = {
        "num_queries_scored": len(per_query_output),
        "mean_demographic_parity": {
            "privileged_selection_rate": round(mean(
                q["demographic_parity"]["privileged_selection_rate"] for q in per_query_output
            ), 4),
            "underrepresented_selection_rate": round(mean(
                q["demographic_parity"]["underrepresented_selection_rate"] for q in per_query_output
            ), 4),
            "selection_rate_ratio": round(mean(dp_ratios), 4) if dp_ratios else None,
            "statistical_parity_difference": round(mean(
                q["demographic_parity"]["statistical_parity_difference"] for q in per_query_output
            ), 4),
        },
        "mean_exposure_diversity": {
            "privileged_exposure": round(mean(
                q["exposure_diversity"]["privileged_exposure"] for q in per_query_output
            ), 4),
            "underrepresented_exposure": round(mean(
                q["exposure_diversity"]["underrepresented_exposure"] for q in per_query_output
            ), 4),
            "privileged_exposure_share": round(mean(
                q["exposure_diversity"]["privileged_exposure_share"] for q in per_query_output
            ), 4),
            "underrepresented_exposure_share": round(mean(
                q["exposure_diversity"]["underrepresented_exposure_share"] for q in per_query_output
            ), 4),
            "exposure_ratio": round(mean(exposure_ratios), 4) if exposure_ratios else None,
            "exposure_difference": round(mean(
                q["exposure_diversity"]["exposure_difference"] for q in per_query_output
            ), 4),
        },
    }

    output = {
        "input_file": str(INPUT_FILE),
        "summary": summary,
        "per_query": per_query_output,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    print("\nFairness metrics from ranked results")
    print("-----------------------------------")
    print(f"Queries scored: {summary['num_queries_scored']}")
    print()
    print("Demographic parity")
    print(f"  Mean privileged selection rate:       {summary['mean_demographic_parity']['privileged_selection_rate']:.4f}")
    print(f"  Mean underrepresented selection rate: {summary['mean_demographic_parity']['underrepresented_selection_rate']:.4f}")
    print(f"  Mean selection rate ratio:            {summary['mean_demographic_parity']['selection_rate_ratio']}")
    print(f"  Mean SPD:                             {summary['mean_demographic_parity']['statistical_parity_difference']:+.4f}")
    print()
    print("Exposure diversity")
    print(f"  Mean privileged exposure:             {summary['mean_exposure_diversity']['privileged_exposure']:.4f}")
    print(f"  Mean underrepresented exposure:       {summary['mean_exposure_diversity']['underrepresented_exposure']:.4f}")
    print(f"  Mean privileged exposure share:       {summary['mean_exposure_diversity']['privileged_exposure_share']:.4f}")
    print(f"  Mean underrepresented exposure share: {summary['mean_exposure_diversity']['underrepresented_exposure_share']:.4f}")
    print(f"  Mean exposure ratio:                  {summary['mean_exposure_diversity']['exposure_ratio']}")
    print(f"  Mean exposure difference:             {summary['mean_exposure_diversity']['exposure_difference']:+.4f}")
    print()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()