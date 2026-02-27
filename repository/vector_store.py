from huggingface_hub.utils.insecure_hashlib import sha256
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def get_documents_id(document:Document) -> str:
    content = document.page_content
    source = document.metadata.get("source", "")

    key = f"{source}:{content}"
    return sha256(key.encode()).hexdigest()[:32]


class VectorStore:
    def __init__(
            self,
            collection_name:str,
            embeddings:Embeddings,
            save_path:str):

        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = save_path


        self._vectorstore: Chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=save_path,
        )

    def find(self,
             query:str,
             k:int = 5
             ):
        docs = self._vectorstore.similarity_search(
            query=query,
            k=k
        )
        return docs

    def add(self,document:Document):
        """단일 문서 추가"""
        ids = get_documents_id(document)

        document.id = id

        self._vectorstore.add_documents(
            documents=[document],
            ids=[ids]
        )

    def save_all(self,documents:list[Document]):
        """여러 문서 일괄 추가"""
        ids = [get_documents_id(document) for document in documents]
        self._vectorstore.add_documents(
            documents,
            ids=ids
        )
    def delete(self,
               ):
        pass



