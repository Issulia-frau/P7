import pytest
from fastapi.testclient import TestClient
import main


@pytest.fixture
def client(monkeypatch):

    # RAG
    def fake_run_rag(question):
        return "Fake answer"

    #vectorstore builder
    def fake_build_vectorstore():
        return None

   
    monkeypatch.setattr(main, "run_rag", fake_run_rag)
    monkeypatch.setattr(main, "build_vectorstore", fake_build_vectorstore)

    return TestClient(main.app)


def test_ask_basic(client):
    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Fake answer"


def test_rebuild(client):
    response = client.post("/rebuild")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"