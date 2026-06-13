# FairSearch-arXiv

**Evaluating and Mitigating Bias in Academic RAG**

FairSearch-arXiv is a retrieval-augmented search system built over a sample of the
arXiv computer science (`cs.*`) corpus. It preprocesses raw arXiv metadata, embeds
papers with a sentence-transformer model, indexes them in a local
[Qdrant](https://qdrant.tech/) vector database, and supports semantic retrieval —
forming the basis for studying and mitigating bias in academic search.

## Project structure

```
.
├── data/
│   ├── raw/          # Raw arXiv JSON snapshot (downloaded from Kaggle — see below)
│   ├── processed/    # Cleaned/sampled dataset (papers.parquet, generated)
│   └── indices/      # Generated Qdrant index files (qdrant/, generated)
├── src/
│   ├── preprocess.py     # Filter cs.* papers, sample, clean → papers.parquet
│   ├── index_builder.py  # Embed papers and build the Qdrant index
│   └── retriever.py       # Run a semantic search query against the index
├── app/              # (Reserved) Streamlit application
├── experiments/      # (Reserved) Experiment outputs and evaluation results
└── requirements.txt
```

## Prerequisites

- Python 3.9+
- A [Kaggle](https://www.kaggle.com/) account (to download the dataset)

## Setup

1. **Clone the repository and enter it**

   ```bash
   git clone <repo-url>
   cd FairSearch-arXiv-Evaluating-and-Mitigating-Bias-in-Academic-RAG
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**

   Download the raw arXiv metadata JSON from the
   [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
   Unzip it and place the file in the `data/raw/` folder. The preprocessing script
   expects it at:

   ```
   data/raw/arxiv-metadata-oai-snapshot.json
   ```

   > If your downloaded file has a different name, either rename it to
   > `arxiv-metadata-oai-snapshot.json` or update `input_path` at the bottom of
   > `src/preprocess.py`.

## Running the pipeline

The scripts use paths relative to the `src/` directory, so **run them from inside
`src/`** and in the following order:

```bash
cd src

# 1. Preprocess: filter cs.* papers, sample 50,000, clean, and save as parquet
python preprocess.py
#    → writes data/processed/papers.parquet

# 2. Build the index: embed papers and store them in a local Qdrant database
python index_builder.py
#    → writes data/indices/qdrant/

# 3. Retrieve: run a semantic search query against the index
python retriever.py
```

### What each step does

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `preprocess.py` | `data/raw/arxiv-metadata-oai-snapshot.json` | `data/processed/papers.parquet` |
| 2 | `index_builder.py` | `data/processed/papers.parquet` | `data/indices/qdrant/` (Qdrant collection `fairsearch_arxiv`) |
| 3 | `retriever.py` | `data/indices/qdrant/` | Top-k search results printed to the console |

### Customizing the search query

`retriever.py` runs a hardcoded example query. To search for something else, edit the
`query` variable in the `__main__` block at the bottom of `src/retriever.py`:

```python
query = "Recent advances in graph neural networks"
```

## Notes

- **Embedding model:** Both indexing and retrieval use
  `sentence-transformers/all-MiniLM-L6-v2`. The same model must be used for both,
  which the scripts handle automatically.
- **GPU acceleration:** `index_builder.py` automatically uses CUDA if a compatible
  GPU is available; otherwise it falls back to CPU. Building the index over 50,000
  papers on CPU can take a while.
- **Sampling is deterministic:** `preprocess.py` samples 50,000 papers with a fixed
  seed (`42`), so the dataset is reproducible across runs.
