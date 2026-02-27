import json
import os

import requests
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from serpapi import GoogleSearch



@tool
def search(query:str) -> str:
    """인터넷에서 정보를 검색합니다."""
    params = {
        "engine": "google_light",
        "q": query,
        "location": "South Korea",
        "google_domain": "google.com",
        "hl": "ko",
        "gl": "kr",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }

    searchs = GoogleSearch(params)
    results = searchs.get_dict()
    organic_results = results["organic_results"]

    return organic_results

# 2. 모델 및 도구 설정
#llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = ChatOllama(model="qwen3-vl:8b",num_ctx=8096)

# 도구 설정
tools = [search]


#agent 설정
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "엔비디아의 지난 분기 영업이익과 최근 출시된 아이폰16의 한국 출시가를 합치면 얼마야?"}]}
)
print(result)

# 전체 메시지 흐름 확인
for message in result['messages']:
    role = message.__class__.__name__
    content = message.content

    print(f"[{role}]")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"🛠️ 도구 호출: {message.tool_calls}")
    else:
        print(content)
    print("-" * 30)