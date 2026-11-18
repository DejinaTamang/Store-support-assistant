import os

from groq import Groq

from config import LLM_MODEL
from retriever import retrieve


NO_ANSWER = "I could not find that in the documents."
MIN_SIMILARITY = 0.30

SYSTEM_PROMPT = """Answer using only the provided context passages.
After each fact, cite the passage with [1], [2], or [3].
If the context does not contain the answer, reply exactly:
"I could not find that in the documents."
"""


def make_llm_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def answer_question(llm: Groq, embed_model, collection, question: str, top_k: int = 3) -> dict:
    retrieved = retrieve(embed_model, collection, question, top_k)
    relevant = [chunk for chunk in retrieved if chunk["similarity"] >= MIN_SIMILARITY]

    if not relevant:
        return {"answer": NO_ANSWER, "sources": []}

    context = "\n\n".join(
        f"[{i}] (source: {chunk['source']})\n{chunk['text']}"
        for i, chunk in enumerate(relevant, start=1)
    )

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=500,
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content or NO_ANSWER,
        "sources": relevant,
    }
