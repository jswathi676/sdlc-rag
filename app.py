import os
import streamlit as st
from google import genai


st.set_page_config(
    page_title="SDLC RAG Assistant",
    page_icon="📚"
)

st.title("📚 SDLC RAG Assistant")
st.write("Ask questions about the Software Development Life Cycle.")


# Get Google API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()

client = genai.Client(api_key=api_key)


# Load KT document
@st.cache_data
def load_kt():
    with open("sdlc_kt.txt", "r", encoding="utf-8") as file:
        return file.read()


# Split KT into chunks
@st.cache_data
def create_chunks(text):
    chunk_size = 1500
    overlap = 250

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


kt_text = load_kt()
chunks = create_chunks(kt_text)


# Ask Gemini to find relevant information
def retrieve_context(question):

    scored_chunks = []

    question_words = set(
        question.lower().split()
    )

    for chunk in chunks:

        chunk_words = set(
            chunk.lower().split()
        )

        score = len(
            question_words.intersection(chunk_words)
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:4]
        if score > 0
    ]


question = st.text_input(
    "Ask your SDLC question:"
)


if question:

    with st.spinner("Searching the SDLC knowledge base..."):

        relevant_chunks = retrieve_context(
            question
        )

        if not relevant_chunks:

            st.warning(
                "I could not find relevant information "
                "in the SDLC knowledge base."
            )

        else:

            context = "\n\n---\n\n".join(
                relevant_chunks
            )

            prompt = f"""
You are an SDLC Knowledge Assistant.

Answer the user's question ONLY using the
information provided in the SDLC knowledge base.

If the information is not available in the
knowledge base, say:

"I could not find this information in the
SDLC knowledge base."

Do not invent information.

SDLC KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("Answer")

            st.write(
                response.text
            )

            st.subheader(
                "Retrieved Knowledge"
            )

            for i, chunk in enumerate(
                relevant_chunks,
                1
            ):

                with st.expander(
                    f"Source {i}"
                ):
                    st.write(chunk)        model="gemini-2.0-flash",