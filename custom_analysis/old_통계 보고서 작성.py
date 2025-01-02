import os; import pandas as pd; import numpy as np; import streamlit as st    
import matplotlib as plt; import seaborn as sns; import plotly   
import warnings; warnings.filterwarnings('ignore')
from dotenv import load_dotenv; load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# agent
from file_llm import pdf_chain, process_imagefile, multimodal_answer
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool  # PythonREPL
from typing import List, Dict, Union, Annotated              # 데이터 타입
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent # Pandas
from langchain_teddynote.messages import AgentCallbacks      # Agent callback 함수
from langchain_teddynote.messages import AgentStreamParser   # Agent 중간단계 스트리밍

print(os.getcwd())    # 현재 작업 디렉토리

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")


# 세션 스테이트 초기화
session_state = st.session_state
if "messages" not in session_state:
    session_state["messages"] = []

# if "store" not in session_state:
#     session_state["store"] = {}

# if "chain" not in session_state:
#     session_state["chain"] = None

if "df" not in session_state:
    session_state["df"] = None



############################ 채팅 ############################

st.title("개인 맞춤형 보고서 작성")
st.markdown("###### 자세히 입력할수록 보고서의 퀄리티는 높아집니다.")

st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기

############################ tab 분류 ############################

# 탭을 생성
list_of_tabs = ["인적사항", "분석상황", "데이터 업로드", "제한사항", "보고서 작성"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(list_of_tabs)

# tab1 : 인적사항
with tab1:
    tab1.markdown("<br>", unsafe_allow_html=True)    # 줄 띄어쓰기
    # 빈 열을 포함한 2단 레이아웃
    col1, spacer, col2 = tab1.columns([1, 0.2, 1])    # 열 비율 설정 (spacer가 간격 역할)

    # 첫번째 열 (왼쪽)
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        age = st.selectbox(
            "나이대", 
            [None, "10대", "20대", "30대", "40대", "50대", "60대 이상"], index=0
        )
        
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        gender = st.radio(
        "성별",
        ["남성", "여성"], index=None,
        )
    
    # 두번째 열(오른쪽)
    with col2:    
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        industry = st.selectbox(
            "업종*", [None, "소비재", "제조업", "서비스업", "금융", "교육", "엔터테인먼트", "유통", "미디어"], index=0
        )
        
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        job = st.selectbox(
            "직무*", [None, "마케터", "기획자", "개발자", "데이터 분석가", "데이터 엔지니어", "디자이너"], index=0
        )

########################## tab2 : 분석상황 ##########################
# tab2 : 분석상황
with tab2:
    st.markdown("<br><br>", unsafe_allow_html=True)    # 줄 띄어쓰기
    analysis_skils = st.multiselect(
        "사용하고 싶은 분석 기술은?(복수 선택 가능)*", [None, "집계 및 시각화", "통계분석", "머신러닝 예측"]
    )
    st.markdown("<br>", unsafe_allow_html=True)    # 줄 띄어쓰기

    analysis_purpose = st.multiselect(
        "분석 목적은?(복수 선택 가능)*", [None, "현황 파악", "변수 인과관계 설명", "미래 예측", "그룹 간 비교", "유의미한 변수 파악"]
    )

########################## tab3 : 데이터 업로드 ##########################
# tab3 : 데이터 업로드
with tab3:
    st.markdown("<br><br>", unsafe_allow_html=True)   # 줄 띄어쓰기
    # 파일 업로드
    uploaded_file = st.file_uploader("csv 파일을 업로드 해주세요", type=['csv'],   # 'pdf', 'jpg', 'jpeg', 'png'
                                    accept_multiple_files=False)

    # 파일이 업로드 되었을 때, pdf, image를 구분
    data = None
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            data = "pdf"
            chain = pdf_chain(uploaded_file)
            
        elif uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]:
            data = "image"
            file_path = process_imagefile(uploaded_file)
            # multimodal_answer(file_path, user_input)
            
        elif uploaded_file.type == "text/csv":
            data = "csv"
            st.info("상위 5개 행을 출력합니다")
            df = pd.read_csv(uploaded_file)  # CSV 파일을 읽어서 데이터프레임으로 변환
            session_state["df"] = df
            st.dataframe(df.head())

    st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
    analysis_comment = st.text_area(
            "어떤 분석을 하고 싶은지 자유롭게 설명해주세요", 
            placeholder="집계와 시각화를 통해 EDA를 하고 싶어. 그리고 charges 변수를 선형회귀 예측하고 유의미한 변수가 무엇인지 파악해줘.",
            height=50) 
    

########################## # tab4 : 제한사항 ##########################
# tab4 : 제한사항
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)    # 줄 띄어쓰기
    # 빈 열을 포함한 2단 레이아웃
    col1, spacer, col2 = tab4.columns([1, 0.2, 1])    # 열 비율 설정 (spacer가 간격 역할)

    # 첫번째 열(왼쪽)
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        Difficulty = st.selectbox(
            "용어 난이도", [None, 
                    "Very Easy", 
                    "Beginner", 
                    "Junior", 
                    "Senior", 
                    "Expert"], index=0
        )

        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        page = st.number_input("보고서 페이지 수", 0, 10, step=1)
        
    # 두번째 열(오른쪽) : Tone/Language
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        Tone = st.selectbox(
            "어조", [None, 
                    "친근한",
                    "비즈니스적인", 
                    "전문가같은", 
                    "논리적인", 
                    "이모티콘이 추가된", 
                    ], index=0
        )

        st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기
        language = st.radio(
        "언어*",
        ["Korean", "English"], index=0,
        )


########################## # tab5 : 보고서 작성 ##########################

# tab5 : 보고서 작성
with tab5:
    strt_btn = st.button("보고서 작성")
    if strt_btn:
        if uploaded_file and strt_btn:
            st.dataframe(df.head())
            st.warning("준비중입니다")
        else:
            st.warning("파일을 업로드 해주세요")

    

# query 예시
# 나는 30대 남성이고 제조업에서 기획자로 일하고 하고 있어.
# 집계와 시각화, 그리고 통계 분석을 수행하고자 해.
# 분석 목적은 charges에 대한 선형회귀 예측과 통계적으로 유의한 변수 파악이야.
# 먼저 charges 분석을 하고싶어. 히스토그램으로 분포를 파악해줘.
# 성별 차이에 따른 charges 분포 비교를 해줘.
# charges와 다른 변수들 간의 상관계수 히트맵을 그리고 해석해줘.
# charges 선형회귀를 한 후에 유의한 변수가 무엇이 있는지 찾고 통계적 이유를 들어줘.
# 용어 난이도는 very easy, 보고서는 A4용지 2page 정도, 어조는 친근하게, 한국어로 답변해줘.
# 양식은 회사 보고서 양식으로 해줘

