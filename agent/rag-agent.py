from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage, add_messages
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from model.ollama_model import OllamaModel
from repository.vector_store import VectorStore


class RagState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # 대화 히스토리 자동 누적
    query: str  # 현재 질문
    retrieved_docs: List[str]  # 검색된 문서들 (또는 Document 리스트)
    answer: str  # 최종 답변
    # 필요하면 더 추가: context: str, tool_calls: list 등

class RagAgent:
    def __init__(self,
                 llm:OllamaModel,
                 vector_store:VectorStore):
        self.llm= llm
        self.vector_store = vector_store


    def retrieve(self,
                 state:RagState) -> RagState:
        self.vector_store.find()