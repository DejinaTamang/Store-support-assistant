from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, EMBEDDING_MODEL


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def load_docs_folder(folder: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(folder.glob("*.txt"))
    }


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break

    return chunks


def build_index(model: SentenceTransformer, client, docs: dict[str, str]):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for source, text in docs.items():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}-{i}")
            texts.append(chunk)
            metadatas.append({"source": source})

    if texts:
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return collection


def retrieve(model: SentenceTransformer, collection, question: str, top_k: int = 3) -> list[dict]:
    if not question.strip() or collection.count() == 0:
        return []

    query_vector = model.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_vector,
        n_results=min(top_k, collection.count()),
    )

    return [
        {
            "text": document,
            "source": metadata["source"],
            "similarity": 1 - distance,
        }
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
