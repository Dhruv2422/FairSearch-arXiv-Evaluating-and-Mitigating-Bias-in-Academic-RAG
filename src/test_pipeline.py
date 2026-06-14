import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from retriever import load_model, connect_qdrant, search
from generator import generate

QUERIES = [
    "What are the main approaches to natural language processing with transformers?",
    "How do convolutional neural networks work for image classification?",
    "What methods exist for federated learning and privacy preservation?",
    "How is reinforcement learning applied to robotics?",
    "What are recent advances in generative adversarial networks?",
]

def run_tests():
    print("Loading model and connecting to Qdrant...")
    model = load_model()
    client = connect_qdrant()
    print(f"Ready. Running {len(QUERIES)} queries.\n")
    print("=" * 70)

    for i, query in enumerate(QUERIES, start=1):
        print(f"[{i}/{len(QUERIES)}] {query}")
        print("-" * 70)

        hits = search(query, model, client, k=5)
        print("Retrieved papers:")
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. {hit.payload['title']} (score {hit.score:.3f})")

        print("\nAnswer:")
        answer = generate(query, hits)
        print(answer)
        print("=" * 70)

if __name__ == "__main__":
    run_tests()
