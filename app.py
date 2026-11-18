"""
Store Support Assistant: chat-style document Q&A with a calculator, powered by Groq.
Run with: streamlit run app.py
"""

from pathlib import Path

import chromadb
import streamlit as st
from groq import APIError

from agent import Agent
from config import load_api_key
from retriever import build_index, load_docs_folder, load_model

DOCS_DIR = Path(__file__).parent / "docs"

st.set_page_config(page_title="Store Support Assistant", page_icon="💬")
st.title("Store Support Assistant")
st.caption("Ask questions about account help, refunds, and shipping. Also does quick calculations.")


@st.cache_resource(show_spinner="Preparing the assistant. The first start can take a minute...")
def get_resources():
    """Load the model and index the documents once, then reuse them for every visitor."""
    embed_model = load_model()
    docs = load_docs_folder(DOCS_DIR)
    collection = build_index(embed_model, chromadb.EphemeralClient(), docs)
    return embed_model, collection, list(docs)


def render_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['source']}** (similarity {s['similarity']:.2f})")
            st.caption(s["text"])


# ---------- Setup checks ----------
if not load_api_key():
    st.error(
        "The app is not configured yet. The administrator must set the "
        "GROQ_API_KEY. See the README for setup steps."
    )
    st.stop()

try:
    embed_model, collection, doc_names = get_resources()
except Exception as e:
    st.error(f"Could not load the documents: {e}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    for name in doc_names:
        st.write(f"- {name}")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- Keep the whole conversation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show every earlier question and answer
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ---------- New question (chat box stays at the bottom) ----------
question = st.chat_input("Ask another question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            agent = Agent(embed_model, collection)
            with st.spinner("Thinking..."):
                result = agent.answer(question)

            st.write(result["answer"])
            if result["sources"]:
                render_sources(result["sources"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except RuntimeError as e:
            st.error(str(e))
        except APIError as e:
            st.error(f"The language model request failed: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")