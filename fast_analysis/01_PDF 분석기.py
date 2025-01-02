import os; import warnings; warnings.filterwarnings("ignore")           # 경고 메시지 무시
from dotenv import load_dotenv; load_dotenv()
import os; import pandas as pd; import numpy as np; import streamlit as st    
import matplotlib as plt; import seaborn as sns; import plotly   
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_teddynote import logging           
logging.langsmith("[Project] PDF 멀티턴 RAG 챗봇")

# 멀티턴
from operator import itemgetter
from retriever import create_retriever
from langchain_core.runnables import RunnableLambda

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

########################################
# PDF RAG + 멀티턴 챗봇
########################################

st.title("PDF 기반 멀티턴 챗봇📑")

# 세션 스테이트 초기화
if "messages_pdf" not in st.session_state:
    st.session_state["messages_pdf"] = []

if "store_pdf" not in st.session_state:
    st.session_state["store_pdf"] = {}

if "chain_pdf" not in st.session_state:
    st.session_state["chain_pdf"] = None

################## 사이드바 ##################
with st.sidebar:
    # 초기화 버튼
    clr_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=['pdf'])

    # 모델 선택
    selected_model = st.selectbox(
        "모델 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )

    # 세션 ID 입력
    session_id = st.text_input("세션 ID를 입력하세요", "abc123")

    # 시스템 프롬프트 입력받기
    system_prompt = st.text_area("시스템 프롬프트", 
    """당신은 질문 답변 작업의 보조자입니다.\
검색된 컨텍스트와 사용자 입력의 다음 부분을 사용하여 질문에 답하십시오.\
답을 모른다면 그냥 모른다고 말하십시오. 한국어로 답변하세요.""",
height=200,) 
    
    # 데이터 분석을 시작하는 버튼
    apply_btn = st.button("데이터 분석 시작")  
    
################## /사이드바 ##################

################################ 기능함수 정의 ################################

# 이전 대화 출력 함수
def print_messages():
    for chat_message in st.session_state["messages_pdf"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 세션 스테이트에 메시지 추가
def add_message(role, message):
    st.session_state["messages_pdf"].append(ChatMessage(role=role, content=message))

# 세션 ID별 대화 기록 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state["store_pdf"]:
        st.session_state["store_pdf"][session_ids] = ChatMessageHistory()
    return st.session_state["store_pdf"][session_ids]

# 파일 업로드 시 retriever 생성
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    return create_retriever(file_path)

# 문서를 포맷팅하는 함수
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# 멀티턴 RAG 기반 체인 생성 함수
def create_chain(retriever, model_name='gpt-4o'):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ""
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "# Context:\n {context} \n\n # Question:\n{question} # Answer:"),  # 사용자 입력을 변수로 사용
        ]
    )

    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)

    chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt 
    | llm 
    | StrOutputParser()
    )
    
    # RunnableWithMessageHistory를 사용하여 세션별 대화 기록 관리
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션별 대화 기록 관리 함수
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history

################################ /기능함수 정의 ################################

# 초기화 버튼 동작
if clr_btn:
    st.session_state["messages_pdf"] = []

# 파일 업로드 후 대화 시작 버튼 누르기
if apply_btn and uploaded_file:
    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")
elif apply_btn:
    st.warning("먼저 파일을 업로드 해주세요.")

# 이전 대화 기록 출력
print_messages()

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 파일 업로드가 되면 retriever 생성, chain 생성, 세션 스테이트에 저장
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever)
    st.session_state["chain_pdf"] = chain


# 사용자 입력
user_input = st.chat_input("무엇이 궁금하신가요?")

# 사용자 입력이 들어오면
if user_input:
    chain = st.session_state["chain_pdf"]

    # 체인이 생성되어 있는지 확인
    if chain is not None:
        # 사용자 메시지 출력
        st.chat_message("user").write(user_input)
        
        # 답변 생성 (세션 ID를 통한 대화 관리)
        response = chain.stream(    # stream 아웃풋은 ["안", "녕"] 리스트처럼 개별 단위로 제공됩니다.
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                # print(token)
                ai_answer += token
                # print(ai_answer)
                container.markdown(ai_answer)   
            print(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
