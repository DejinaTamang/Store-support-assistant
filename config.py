"""
Loads the Groq API key from outside the code.
1. GROQ_API_KEY environment variable (local use, or set on the host)
2. Streamlit secrets (.streamlit/secrets.toml, or app settings on Streamlit Community Cloud)
"""

import os

import streamlit as st


LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "capstone_docs"


def load_api_key() -> bool:
    """Make sure GROQ_API_KEY is available. Returns True if it is."""
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return False
    os.environ["GROQ_API_KEY"] = key
    return True