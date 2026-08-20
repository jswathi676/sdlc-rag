import os
import re
from collections import Counter

import streamlit as st
from google import genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDLC KT Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 SDLC Knowledge Transfer Assistant")
st.caption("Ask any question about the SDLC — Agile, DevOps, Testing, Security, Roles, Tools, and more.")

# ── API client ────────────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY environment variable is not set.")
    st.stop()

client = genai.Client(api_key=api_key)

# ── Load & chunk KT document ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading knowledge base...")
def load_and_chunk(path="sdlc_kt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on section dividers
    sections = re.split(r"\n[-=]{4,}\n", raw)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 50:  # skip very short fragments
            chunks.append(section)
    return chunks


KT_CHUNKS = load_and_chunk()

# ── Simple keyword retrieval ──────────────────────────────────────────────────

def retrieve(question, top_k=4):
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    # Remove very common stop words
    stop_words = {'the','a','an','is','are','was','were','what','how','why',
                  'when','where','who','which','in','of','to','and','or','for',
                  'it','this','that','with','on','at','by','from','be','do'}
    question_words -= stop_words

    scored = []
    for chunk in KT_CHUNKS:
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        score = len(question_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

# ── Gemini generation ─────────────────────────────────────────────────────────

def generate_answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are an expert SDLC Knowledge Transfer Assistant.

Answer the question using ONLY the SDLC Knowledge Base below.
Be clear and structured. If not found, say "I couldn't find this in the SDLC Knowledge Base."

SDLC KNOWLEDGE BASE:
{context}

QUESTION: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    return response.text

# ── Chat UI ───────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about SDLC, Agile, DevOps, Testing, Security...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        relevant_chunks = retrieve(question, top_k=4)

        if not relevant_chunks:
            answer = "I couldn't find relevant information. Please try rephrasing."
            st.markdown(answer)
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, relevant_chunks)
            st.markdown(answer)

            with st.expander(f"📄 {len(relevant_chunks)} source(s) from KT document"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Source {i}**")
                    st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ Knowledge Base")
    st.metric("Sections indexed", len(KT_CHUNKS))
    st.markdown("""
**Topics covered:**
- SDLC Models
- Planning & Requirements
- System Design
- Implementation & Code Review
- Testing & QA
- Deployment & Release
- Maintenance & Support
- Risk Management
- Security (OWASP)
- Tools & Tech Stack
- Team Roles
- Metrics & KPIs
- Governance & Compliance
- Case Study
    """)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
