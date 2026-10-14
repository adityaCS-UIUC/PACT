"""
RAG environment tools.

Provides a document corpus and search tools that demonstrate the
speed-vs-accuracy tradeoff: accuracy comes from making the *right* call
(correct tool, correct query), not from calling more.
"""

from __future__ import annotations

from agents import function_tool

from pact.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# In-memory document corpus
# ---------------------------------------------------------------------------

RAG_DOCUMENTS: list[str] = [
    "The Roman Empire fell in 476 CE when Romulus Augustulus was deposed by Odoacer.",
    "Sharks have been on Earth for over 400 million years; trees evolved around 350 million years ago.",
    "The first president of the United States was George Washington, inaugurated in 1789.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Photosynthesis occurs in chloroplasts; plants use sunlight to convert CO2 and water into glucose.",
    "The Treaty of Versailles was signed in 1919, ending World War I.",
    "Machine learning is a subset of artificial intelligence focused on learning from data.",
    "DNA was first identified by Friedrich Miescher in 1869; its double-helix structure was described by Watson and Crick in 1953.",
    "The Great Wall of China was built over many centuries; most of the existing wall dates from the Ming Dynasty (1368–1644).",
    "Albert Einstein published the theory of general relativity in 1915.",
    "The Amazon rainforest produces roughly 20 percent of the world's oxygen.",
    "The Pythagorean theorem states that a^2 + b^2 = c^2 for right triangles.",
    "Quantum computing uses qubits that can exist in superposition of 0 and 1 simultaneously.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
]


def _retrieve(query: str, max_results: int = 3, skip: int = 0) -> list[str]:
    """Keyword-overlap retrieval. Replace with embeddings for real use."""
    tokens = query.lower().split()
    scored: list[tuple[float, str]] = []
    for doc in RAG_DOCUMENTS:
        d_lower = doc.lower()
        score = sum(1 for w in tokens if w in d_lower)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[skip : skip + max_results]]


# ---------------------------------------------------------------------------
# Plain-function versions (for ToolRegistry / Agent B)
# ---------------------------------------------------------------------------

def search_corpus(query: str, max_results: int = 3) -> str:
    """Search the document corpus for passages matching the query."""
    max_results = max(1, min(5, max_results))
    hits = _retrieve(query, max_results=max_results)
    if not hits:
        return "No matching passages found."
    return "\n\n---\n\n".join(f"[{i+1}] {h}" for i, h in enumerate(hits))


def refine_search(query: str, skip: int = 0) -> str:
    """Retrieve more passages for the same query, skipping already-seen ones."""
    skip = max(0, skip)
    hits = _retrieve(query, max_results=3, skip=skip)
    if not hits:
        return "No additional passages found."
    return "\n\n---\n\n".join(f"[{i+1}] {h}" for i, h in enumerate(hits))


# ---------------------------------------------------------------------------
# openai-agents SDK @function_tool versions (for Agent A via Runner)
# ---------------------------------------------------------------------------

@function_tool
def search(query: str, max_results: int = 3) -> str:
    """Search the document corpus for passages matching the query.
    Call when the answer likely lives in the corpus.
    Accuracy = choosing to search when needed, with a query that matches how the info is expressed.

    Args:
        query: Search query (keywords or natural language).
        max_results: Number of passages to return (1-5). Default 3.
    """
    return search_corpus(query, max_results)


@function_tool
def get_more_passages(query: str, skip: int = 0) -> str:
    """Retrieve more passages for a query, skipping already-seen results.
    Use only when the first search was on the right track but missed relevant docs.

    Args:
        query: Same or refined search query.
        skip: Number of already-seen results to skip.
    """
    return refine_search(query, skip)


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

SEARCH_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {"type": "integer", "description": "1-5 results.", "default": 3},
    },
    "required": ["query"],
}

REFINE_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "skip": {"type": "integer", "description": "Results to skip.", "default": 0},
    },
    "required": ["query"],
}


def build_rag_registry() -> ToolRegistry:
    """Return a ToolRegistry pre-loaded with RAG tools."""
    reg = ToolRegistry()
    reg.register(
        name="search",
        description="Search the document corpus for passages matching the query.",
        parameters=SEARCH_PARAMS_SCHEMA,
        fn=search_corpus,
    )
    reg.register(
        name="get_more_passages",
        description="Retrieve more passages, skipping already-seen results.",
        parameters=REFINE_PARAMS_SCHEMA,
        fn=refine_search,
    )
    return reg
