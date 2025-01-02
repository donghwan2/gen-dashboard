import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import warnings; warnings.filterwarnings("ignore")           # 경고 메시지 무시
from dotenv import load_dotenv; load_dotenv()
from langchain_teddynote import logging           
logging.langsmith("이미지 인식 챗봇")

# 멀티턴
from operator import itemgetter
from retriever import create_retriever
from langchain_core.runnables import RunnableLambda

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 멀티모달
from langchain_teddynote.models import MultiModal   # 이미지 인식

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

################################ 채팅 준비 #################################

st.title("IMAGE 인식 챗봇🖼️")

# 세션 스테이트 초기화
if "messages_img" not in st.session_state:
    st.session_state["messages_img"] = []

if "store_img" not in st.session_state:
    st.session_state["store_img"] = {}

if "chain_img" not in st.session_state:
    st.session_state["chain_img"] = None

# 탭을 생성
tab1, tab2 = st.tabs(["이미지", "대화내용"])

################## 사이드바 ##################
with st.sidebar:
    # 초기화 버튼
    clr_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    # 모델 선택
    selected_model = st.selectbox(
        "모델 선택", ["gpt-4o", "gpt-4o-mini"], index=0
    )

    # 세션 ID 입력
    session_id = st.text_input("세션 ID를 입력하세요", "abc123")

    # 시스템 프롬프트 입력받기  
    system_prompt = st.text_area(
        "시스템 프롬프트", 
        """당신은 표를 분석하는 AI 어시스턴트입니다. 당신의 임무는 주어진 테이블을 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다. 한국어로 대답하세요.""",
        height=200) 
    
    # 데이터 분석을 시작하는 버튼
    apply_btn = st.button("데이터 분석 시작")  
    
################## /사이드바 ##################

# 이전까지의 대화 출력 함수
def print_messages():
    for chat_message in st.session_state["messages_img"]:
        tab2.chat_message(chat_message.role).write(chat_message.content)

# 세션 스테이트에 새로운 메시지 추가 함수
def add_message(role, message):
    st.session_state["messages_img"].append(ChatMessage(role=role, content=message))

# 이미지가 업로드 되었을 때, 이미지를 캐시 저장(시간이 오래 걸리는 작업에 사용)
# "01-MyProject/.cache/files" 디렉토리 안에 파일이 캐시 저장됨.
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")   # 업로드 파일을 캐싱해서 가지고 있는다.
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# multimodal chain 생성 함수
def multimodal_answer(image_filepath, system_prompt, user_prompt, model_name='gpt-4o'):
    
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,          # 창의성 (0.0 ~ 2.0)
        model_name=model_name,  # 모델명
    )

    # 멀티모달 객체 생성
    multimodal = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    # 이미지 파일로 부터 질의(스트림 방식)
    response = multimodal.stream(image_filepath)

    return response


################################# 채팅 #################################

# 초기화 버튼이 눌리면 세션 messages 초기화
if clr_btn:
    st.session_state["messages_img"] = []

# 파일 업로드 후 대화 시작 버튼 누르기
if apply_btn and uploaded_file:
    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")
elif apply_btn:
    st.warning("먼저 파일을 업로드 해주세요.")

# 이전까지 세션 스테이트의 대화내용 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("무엇이 궁금하신가요?")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = tab2.empty()

# 이미지가 업르드되면 이미지를 출력
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    tab1.image(image_filepath)

# 사용자 입력이 들어오면,
if user_input:
    # 파일이 업로드 되었는지 확인
    if uploaded_file:
        # 새로운 사용자 입력 출력
        tab2.chat_message("user").write(user_input)
        
        image_filepath = process_imagefile(uploaded_file)
        # 답변 요청
        response = multimodal_answer(image_filepath, system_prompt, user_input, selected_model)

        with tab2.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서 여기에 토큰을 스트리밍 출력.
            container = st.empty()
            answer = ""
            for token in response:  # 답변의 토큰을 하나씩 스트리밍 출력
                answer += token.content
                container.markdown(answer, unsafe_allow_html=True)
        # 세션 스테이트 messages에 새로운 메시지 추가
        add_message("user", user_input)
        add_message("assistant", answer)   
    else:
        # 이미지가 없으면 업로드 하라는 경고 메시지 출력
        warning_msg.error("이미지를 업로드 해주세요.")

