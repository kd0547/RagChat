import pytest
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from sqlalchemy.testing.suite.test_reflection import metadata

from repository.vector_store import VectorStore

@pytest.fixture
def test_vectorstore(tmp_path):
    model = "qwen3-embedding:latest"
    embeddings = OllamaEmbeddings(model=model)
    vs = VectorStore(
        collection_name="pytest_test",
        embeddings=embeddings,
        save_path=str(tmp_path / "chroma_db")
    )
    return vs

def test_add_and_find(test_vectorstore):
    vs = test_vectorstore

    doc = Document(page_content="pytest 테스트 문서", metadata={"test": True})

    vs.add(doc)
    results = vs.find("pytest", k=1)
    assert len(results) > 0
    assert "pytest" in results[0].page_content