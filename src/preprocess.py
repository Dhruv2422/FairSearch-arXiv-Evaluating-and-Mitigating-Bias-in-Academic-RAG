import json
import random
import pandas as pd

def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def filter_cs(papers):
    """
    Keep only cs.* category papers.
    """
    return [p for p in papers if "cs." in p.get("categories", "")]

def sample_papers(papers, n=50000, seed=42):
    """
    Randomly sample 50,000 papers
    """
    random.seed(seed)

    if len(papers) < n:
        raise ValueError(f"Not enough papers: {len(papers)} available")

    return random.sample(papers, n)

def clean_papers(papers):
    """
    Keep only necessary metadata.
    """
    cleaned = []

    for p in papers:
        cleaned.append({
            "paper_id": p.get("id", ""),
            "title": p.get("title", "").strip(),
            "abstract": p.get("abstract", "").strip(),
            "authors": p.get("authors", ""),
            "category": p.get("categories", ""),
            "year": int(p.get("update_date", "1900")[:4])
        })

    return cleaned

def save_parquet(papers, output_path):
    """
    Save processed dataset for fast loading later.
    """
    df = pd.DataFrame(papers)
    df.to_parquet(output_path, index=False)

def build_dataset(input_path, output_path, sample_size=50000):
    print("Loading dataset...")
    data = load_data(input_path)

    print(f"Total papers loaded: {len(data)}")

    print("Filtering cs.* papers...")
    data = filter_cs(data)

    print(f"CS papers: {len(data)}")

    print(f"Sampling {sample_size} papers...")
    data = sample_papers(data, n=sample_size)

    print("Cleaning dataset...")
    data = clean_papers(data)

    print("Saving processed dataset...")
    save_parquet(data, output_path)

    print(f"Done! Saved to: {output_path}")

if __name__ == "__main__":
    build_dataset(
        input_path="../data/raw/arxiv-metadata-oai-snapshot.json",
        output_path="../data/processed/papers.parquet",
        sample_size=50000
    )