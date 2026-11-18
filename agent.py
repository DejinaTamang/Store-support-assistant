"""
An agent that can search the documents AND use a calculator, using Groq.
The model decides which tool to call. The loop repeats until the model
writes a final answer without requesting a tool.
"""

import json

from rag import LLM_MODEL, MIN_SIMILARITY, NO_ANSWER, make_llm_client
from retriever import retrieve
from tools import calculate

MAX_STEPS = 5

SYSTEM_PROMPT = """You are a customer help assistant for an online store.
- For policy questions, call search_docs first, then answer using only those passages.
- Cite the source file name for each policy fact, e.g. (refund_policy.txt).
- For any arithmetic, call calculate instead of calculating yourself.
- If the documents do not cover the question, say you could not find that information."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the store's help documents. Returns relevant passages with their source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression exactly, e.g. '(120 * 0.8) + 5'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The arithmetic expression"}
                },
                "required": ["expression"],
            },
        },
    },
]


class Agent:
    def __init__(self, embed_model, collection):
        self.llm = make_llm_client()
        self.embed_model = embed_model
        self.collection = collection
        self.sources: list[dict] = []

    def run_tool(self, name: str, args: dict) -> str:
        if name == "search_docs":
            hits = retrieve(self.embed_model, self.collection, args.get("query", ""), top_k=3)
            relevant = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]
            if not relevant:
                return "No relevant passages found."
            self.sources.extend(relevant)
            return "\n\n".join(f"(source: {h['source']})\n{h['text']}" for h in relevant)
        if name == "calculate":
            return calculate(args.get("expression", ""))
        return f"Error: unknown tool '{name}'"

    def answer(self, question: str) -> dict:
        self.sources = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(MAX_STEPS):
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=600,
                temperature=0.2,
                tools=TOOLS,
                messages=messages,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = (msg.content or "").strip()
                unique = {(s["source"], s["text"]): s for s in self.sources}
                return {"answer": text or NO_ANSWER, "sources": list(unique.values())}

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": self.run_tool(tc.function.name, args),
                })

        return {
            "answer": "I could not finish answering. Please try a simpler question.",
            "sources": self.sources,
        }