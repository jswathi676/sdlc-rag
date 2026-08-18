import os
import re
import math
from collections import Counter

import streamlit as st
from google import genai

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDLC KT Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 SDLC Knowledge Transfer Assistant")
st.caption(
    "Ask any question about the SDLC — Agile, DevOps, Testing, Security, "
    "Roles, Tools, and more."
)

# ── API client ───────────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "⚠️ GOOGLE_API_KEY environment variable is not set. "
        "Add it in Render → Environment → Add Environment Variable."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ── Load & chunk KT document ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading knowledge base…")
def load_and_chunk(path: str = "sdlc_kt.txt") -> list[dict]:
    """
    Split the KT document into semantically meaningful chunks.

    Strategy:
    1. Split on the section/sub-section divider lines (dashes or equal signs).
    2. Further split any chunk that is still > MAX_CHARS using a
       sliding window so we never lose context at boundaries.
    """
    MAX_CHARS = 1_200
    OVERLAP = 200

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on lines of 4+ dashes or equal signs (section dividers in the KT)
    sections = re.split(r"\n[-=]{4,}\n", raw)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= MAX_CHARS:
            chunks.append(section)
        else:
            # Sliding-window split for oversized sections
            start = 0
            while start < len(section):
                end = start + MAX_CHARS
                chunks.append(section[start:end])
                start += MAX_CHARS - OVERLAP

    # Return as dicts so we can attach metadata later if needed
    return [{"id": i, "text": c} for i, c in enumerate(chunks)]


KT_CHUNKS = load_and_chunk()

# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


@st.cache_data(show_spinner=False)
def _build_idf(chunks: tuple) -> dict[str, float]:
    """Pre-compute IDF weights for all terms across all chunks."""
    N = len(chunks)
    df: Counter = Counter()
    for chunk in chunks:
        terms = set(_tokenize(chunk["text"]))
        for t in terms:
            df[t] += 1
    return {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}


# Pass chunks as a tuple so it's hashable for st.cache_data
IDF = _build_idf(tuple(KT_CHUNKS))


def _tfidf_vector(tokens: list[str], idf: dict) -> dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: (count / total) * idf.get(t, 0) for t, count in tf.items()}


def _cosine(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Return the top_k most relevant chunks using TF-IDF cosine similarity."""
    q_tokens = _tokenize(question)
    q_vec = _tfidf_vector(q_tokens, IDF)

    scored = []
    for chunk in KT_CHUNKS:
        c_tokens = _tokenize(chunk["text"])
        c_vec = _tfidf_vector(c_tokens, IDF)
        score = _cosine(q_vec, c_vec)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Only return chunks with a non-zero similarity score
    return [chunk for score, chunk in scored[:top_k] if score > 0]


# ── Gemini generation ─────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    prompt = f"""You are an expert SDLC Knowledge Transfer Assistant.

Answer the user's question using ONLY the information provided in the
SDLC Knowledge Base below. Be thorough, structured, and clear.

If the answer is not found in the knowledge base, say:
"I couldn't find this information in the SDLC Knowledge Base."

Do NOT invent information. Do NOT use prior training knowledge beyond
what is in the knowledge base.

════════════════════════════════
SDLC KNOWLEDGE BASE (retrieved):
════════════════════════════════
{context}

════════════════════════════════
USER QUESTION:
{question}
════════════════════════════════

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask about SDLC, Agile, DevOps, Testing, Security…")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            relevant_chunks = retrieve(question, top_k=5)

        if not relevant_chunks:
            answer = (
                "I couldn't find relevant information about that in the "
                "SDLC Knowledge Base. Please try rephrasing your question."
            )
            st.markdown(answer)
        else:
            with st.spinner("Generating answer…"):
                answer = generate_answer(question, relevant_chunks)
            st.markdown(answer)

            # Show retrieved sources in an expander
            with st.expander(
                f"📄 {len(relevant_chunks)} source(s) used from the KT document"
            ):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Source {i}**")
                    st.caption(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar: knowledge base info ──────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Knowledge Base")
    st.metric("Total chunks indexed", len(KT_CHUNKS))
    st.markdown(
        """
        **Topics covered:**
        - SDLC Models (Waterfall, Agile, DevOps…)
        - Planning & Requirements
        - System Design
        - Implementation & Code Review
        - Testing & QA
        - Deployment & Release Management
        - Maintenance & Support
        - Risk Management
        - Security (Secure SDLC / OWASP)
        - Tools & Technology Stack
        - Team Roles & Responsibilities
        - Metrics & KPIs
        - Governance & Compliance
        - Case Study
        """
    )
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
