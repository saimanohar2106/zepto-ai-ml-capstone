import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from graph import ask_question
from ingestion import get_collection, load_documents
from main import app


client = TestClient(app)


def test_eight_policy_documents_exist():
    documents, ids, metadatas = load_documents()

    assert len(documents) == 8
    assert len(ids) == 8
    assert len(metadatas) == 8


def test_chroma_collection_has_eight_documents():
    collection = get_collection()

    assert collection.count() == 8


def test_policy_question_uses_retrieval():
    response = ask_question("What is the delivery fee?")

    assert response.answer.startswith(
        "Based on the retrieved context:"
    )
    assert len(response.sources) == 3
    assert response.confidence == 1.0


def test_general_question_uses_direct_answer():
    response = ask_question("What is your favorite food?")

    assert response.answer == (
        "I can only answer questions about Zepto policies right now."
    )
    assert response.sources == []
    assert response.confidence == 1.0


def test_fastapi_policy_endpoint():
    response = client.post(
        "/ask",
        json={"query": "How long do refunds take?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_fastapi_general_endpoint():
    response = client.post(
        "/ask",
        json={"query": "Tell me a joke."}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sources"] == []
    assert data["confidence"] == 1.0