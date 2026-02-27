import streamlit as st
import os
import tempfile
import time

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. 페이지 설정
st.set_page_config(
    page_title="AI Analyst (Dark)",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. [디자인] CSS (잘림 방지 및 중앙 정렬 밸런스)
st.markdown("""
<style>
    /* 1. 앱 배경 및 폰트 설정 */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* 2. 메인 컨테이너 여백 조정 (잘림 방지 핵심) */
    .block-container {
        padding-top: 3rem !important; /* 위쪽 여백을 넉넉히 주어 잘림 방지 */
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* 3. 채팅창 높이 자동 계산 (화면 꽉 채우기) */
    /* 100vh(전체화면) - 170px(상단 여백 + 입력창 높이 + 하단 여백) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: calc(100vh - 170px) !important;
        background-color: #161920; 
        border: 1px solid #303030;
        border-radius: 12px;
        overflow-y: auto; 
        display: flex;
        flex-direction: column;
    }

    /* 스크롤바 디자인 */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
    ::-webkit-scrollbar-track { background: #161920; }

    /* 4. 파일 업로더 디자인 */
    .stFileUploader {
        background-color: transparent !important;
        border: 1px dashed #555;
        border-radius: 8px;
        padding: 5px;
    }
    .stFileUploader div { color: #ccc !important; }
    .stFileUploader small { display: none; }

    /* 5. 버튼 스타일 */
    .stButton button {
        width: 100%;
        background-color: #262730;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        height: 3em;
    }
    .stButton button:hover { background-color: #363945; }

    /* 6. 입력창 스타일 */
    .stChatInputContainer { background-color: #0E1117 !important; padding-bottom: 1rem !important; }
    .stChatInput input { color: white !important; }

    /* 7. 헤더 숨기기 (공간 확보) */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# 3. 모델 로딩
@st.cache_resource
def load_llm():
    return ChatOllama(model="qwen3-vl:8b", num_ctx=8096, temperature=0.1)


@st.cache_resource
def load_embedding_model():
    return OllamaEmbeddings(model='qwen3-embedding:latest')


try:
    llm = load_llm()
    embeddings = load_embedding_model()
except Exception:
    st.error("⚠️ Ollama 모델 로딩 실패.")
    st.stop()

# 4. 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "process_complete" not in st.session_state:
    st.session_state.process_complete = False

# ==========================================
# [레이아웃] 3:7 분할
# ==========================================
col_left, col_right = st.columns([3, 7], gap="medium")

# ---------------------------------------------------------
# [LEFT] 설정 패널
# ---------------------------------------------------------
with col_left:
    st.markdown("### 🌙 AI Analyst")

    # 1. RAG 모드 토글
    use_rag = st.toggle("📄 문서 분석 모드 (RAG)", value=True)

    # 2. 탭으로 프롬프트 분리
    tab1, tab2 = st.tabs(["📄 RAG 설정", "💬 일반 설정"])

    with tab1:
        rag_prompt = st.text_area(
            "RAG 프롬프트",
            value="당신은 냉철한 데이터 분석가입니다. 문서 내용에 기반하여 사실만 답변하세요.",
            height=120,
            key="rag_prompt_input"
        )
        if use_rag and not st.session_state.vectorstore:
            st.warning("👇 아래에서 문서를 업로드하세요.")

    with tab2:
        general_prompt = st.text_area(
            "일반 대화 프롬프트",
            value="당신은 유능한 AI 비서입니다. 자유롭고 친절하게 답변하세요.",
            height=120,
            key="general_prompt_input"
        )

    st.markdown("---")

    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 업로드", type="pdf", label_visibility="collapsed")

    status_area = st.empty()

    if uploaded_file:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            status_area.info("⏳ 분석 중...")
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                #pdf 읽기
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()


                # 문맥 나누기
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(docs)


                #백터 저장
                vectorstore = Chroma.from_documents(split_docs, embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.process_complete = True

                os.unlink(tmp_file_path)
                status_area.success("✅ 완료")
                time.sleep(1)
                status_area.empty()

            except Exception as e:
                status_area.error(f"Error: {e}")
        else:
            if st.session_state.process_complete:
                status_area.caption(f"📑 {uploaded_file.name}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------
# [RIGHT] 채팅창 (상하좌우 꽉 참)
# ---------------------------------------------------------
with col_right:
    # 높이 계산 로직 수정으로 인해 상하가 꽉 차게 렌더링됨
    chat_container = st.container(height=500, border=True)

    with chat_container:
        if not st.session_state.messages:
            msg = "문서를 분석할 준비가 되었습니다." if use_rag else "무엇이든 물어보세요."
            st.markdown(
                f"""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #666;'>
                    <h3>🌑 {msg}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # HTML 말풍선 렌더링
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='background-color: #2b5c8a; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px; max-width: 75%; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='background-color: #333333; color: #e0e0e0; padding: 10px 15px; border-radius: 15px 15px 15px 0; max-width: 75%; border: 1px solid #444;'>
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 입력창
    input_text = "질문을 입력하세요... (📄 RAG)" if (use_rag and st.session_state.vectorstore) else "질문을 입력하세요... (💬 일반)"

    if prompt := st.chat_input(input_text):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: #2b5c8a; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px; max-width: 75%; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                    {prompt}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 프롬프트 선택 로직
        if use_rag and st.session_state.vectorstore:
            retrieved_docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = f"[지시사항]\n{rag_prompt}\n\n[문서내용]\n{context_text}\n\n[질문]\n{prompt}"
        else:
            final_prompt = f"[지시사항]\n{general_prompt}\n\n[질문]\n{prompt}"

        # AI 답변 생성
        with chat_container:
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in llm.stream(final_prompt):
                    full_response += chunk.content
                    message_placeholder.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                        <div style='background-color: #333333; color: #e0e0e0; padding: 10px 15px; border-radius: 15px 15px 15px 0; max-width: 75%; border: 1px solid #444;'>
                            {full_response}▌
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                message_placeholder.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='background-color: #333333; color: #e0e0e0; padding: 10px 15px; border-radius: 15px 15px 15px 0; max-width: 75%; border: 1px solid #444;'>
                        {full_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")