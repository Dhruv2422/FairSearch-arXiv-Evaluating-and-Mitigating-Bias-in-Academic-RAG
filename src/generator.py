import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = (
    "You are a research assistant synthesizing findings from retrieved academic papers. "
    "Answer the user's question by drawing on the provided abstracts. "
    "Cite each paper you use by its title in brackets. "
    "Synthesize what the abstracts say — do not refuse to answer simply because the abstracts "
    "are incomplete. Only flag missing information if the context is genuinely irrelevant to "
    "the question."
)


def build_context(hits: list) -> str:
    """Format Qdrant result points into a numbered context block."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        p = hit.payload
        parts.append(
            f"[{i}] Title: {p.get('title', 'N/A')}\n"
            f"    Abstract: {p.get('abstract', '')}"
        )
    return "\n\n".join(parts)


def generate(query: str, hits: list, api_key: str | None = None) -> str:
    """
    Call Gemini 1.5 Flash with RAG context and return the response text.

    Args:
        query:   The user's research question.
        hits:    List of ScoredPoint objects from retriever.search().
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key required — set GEMINI_API_KEY or pass api_key=")

    client = genai.Client(api_key=key)

    context = build_context(hits)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from retriever import load_model, connect_qdrant, search

    query = "Recent advances in graph neural networks"

    print(f"Query: {query}\n")

    embedding_model = load_model()
    client = connect_qdrant()
    hits = search(query, embedding_model, client, k=5)

    print(f"Retrieved {len(hits)} papers. Generating answer...\n")
    answer = generate(query, hits)
    print("Answer:\n")
    print(answer)
