from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policy"


def load_documents():
    documents = []
    ids = []
    metadatas = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(text)
            ids.append(file_path.stem)
            metadatas.append({
                "document_id": file_path.stem,
                "source": file_path.name,
            })

    return documents, ids, metadatas


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    return collection


def build_index():
    documents, ids, metadatas = load_documents()

    if not documents:
        raise RuntimeError("No policy documents found.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def ensure_index():
    collection = get_collection()

    if collection.count() == 0:
        collection = build_index()

    return collection


if __name__ == "__main__":
    collection = build_index()

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents indexed: {collection.count()}")
    print("Embedding model: all-MiniLM-L6-v2")
    print("ChromaDB index created successfully.")