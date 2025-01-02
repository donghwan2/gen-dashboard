import os; import warnings; warnings.filterwarnings("ignore")           # 경고 메시지 무시
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
import os; import pandas as pd; import numpy as np; import streamlit as st    
import matplotlib as plt; import seaborn as sns; import plotly   

from langchain_core.messages.chat import ChatMessage         # Streamlit에서 ChatMessage 저장
from langchain_core.prompts import PromptTemplate            # 프롬프트 템플릿
from langchain_core.prompts import ChatPromptTemplate        # AI와 대화용 프롬프트 템플릿
from langchain_core.prompts import load_prompt
from langchain import hub                  
from langchain_core.output_parsers import StrOutputParser

import glob
from retriever import create_retriever
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent # Pandas

# rag chain 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 멀티턴
from langchain_core.prompts import MessagesPlaceholder       # 임시 공간_대화 기록이 쌓이게 됨
from langchain_community.chat_message_histories import ChatMessageHistory      # 세션 ID 별 대화 기록을 관리, in memory(휘발성)
from langchain_core.runnables.history import RunnableWithMessageHistory        # 저장된 대화 기록을 가져오는 체인 구성

# 에이전트
from typing import List, Dict, Union, Annotated              # 데이터 타입
from langchain_teddynote.messages import AgentCallbacks      # Agent callback 함수
from langchain_teddynote.messages import AgentStreamParser   # Agent 중간단계 스트리밍
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool  # PythonREPL

from langchain_teddynote import logging      # 랭스미스 로그 추적   
logging.langsmith("CSV Agent 챗봇 ")          # 프로젝트 이름,   set_enable=False : 로그 추적 끄기

# 캐시 디렉토리 생성(.폴더 : 숨김 폴더)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더(파일 캐싱 임시 저장)
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")


st.title("CSV 데이터 분석 챗봇📊")

# 처음에 세션 스테이트 초기화 
session_state = st.session_state
# messages : 현재 세션에서 대화 기록 저장(세션 ID 별 대화 구분 X)
if "messages_csv" not in session_state:
    session_state["messages_csv"] = []

################################ 기능함수 정의 ################################

class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """
    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할

class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """
    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지


# 이전까지 session_state["messages"]에 저장된 대화기록 출력
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in session_state["messages_csv"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):    # 예상치 못한 데이터 타입 입력 방지
                    message_type, message_content = content
                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)          # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)            # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )                                 # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)         # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")

# 세션 스테이트에 새로운 content 추가 함수
def add_message(role: MessageRole, content: List[Union[MessageType, str]]):  # content: (ex) ["dataframe", "df.head()"]
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole) : 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]) : 메시지 내용
    """
    messages = session_state["messages_csv"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다


################################ 채팅 #################################


################################ 사이드바 ################################
with st.sidebar:
    clr_btn = st.button("대화 초기화")  # 대화 내용을 초기화하는 버튼

    # CSV 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드 해주세요.", type=['csv'], accept_multiple_files=False)
    print("/n", "업로드된 파일:", uploaded_file, "/n")

    # 데이터 분석을 시작하는 버튼
    apply_btn = st.button("데이터 분석 시작")  


################################ /사이드바 ################################

################################ 에이전트 ################################

# 콜백 함수
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict): 실행된 도구 정보
    """
    print("<<<<<<< 도구 호출 >>>>>>")
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:         # ex) "len(df)", "남녀 비율 바차트를 그려줘"
                df_in_result = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in session_state:
                        # python code 실행 후 결과를 result에 담는다.
                        result = session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    status.update(label="코드 출력", state="complete", expanded=False)

                # dataframe이라면 출력 후 add_message 함수로 저장
                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                # code에 "plt.show"가 들어있으면 시각화 후 add_message 함수로 저장
                if "plt.show" in query:
                    fig = plt.pyplot.gcf()    # raw_code 에러(plt.gcf)
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])
                print("<<<<<<< ~도구 호출 >>>>>>", '\n')
                
                return result
                
            else:
                st.error(
                    "데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요."
                )
                print("<<<<<<< ~도구 호출 >>>>>>", '\n')
                return


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    print("<<<<<<< 관찰 내용 >>>>>>")
    if "observation" in observation:
        obs = observation["observation"]
        # 에러 발생 시 에러 출력 후 다시 
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            session_state["messages_csv"][-1][1].clear()  # 에러 발생 시 마지막 메시지 삭제
    print("<<<<<<< ~관찰 내용 >>>>>>", '\n')

def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    print("<<<<<<< 최종 답변 >>>>>>")
    pass  # 현재는 아무 동작도 하지 않습니다
    print("<<<<<<< ~최종 답변 >>>>>>", '\n')

# 에이전트 생성 함수
def create_csv_agent(dataframe):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4o"

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    return create_pandas_dataframe_agent(
        ChatOpenAI(model="gpt-4o", temperature=0),
        dataframe,
        verbose = False,
        agent_type = "tool-calling",
        allow_dangerous_code = True,
        prefix ="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request. "
        "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "The language of final answer should be written in Korean. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n",
    )
    # "당신은 전문적인 데이터 분석가이자 pandas 전문가입니다."
    # "사용자의 요청에 응답하려면 pandas DataFrame(`df`)을 사용해야 합니다."
    # "[중요] 코드에서 `df` 변수를 만들거나 덮어쓰지 마십시오. "
    # "시각화 코드를 생성할 의향이 있다면 코드 끝에 `plt.show()`를 사용하십시오."
    # "시각화에는 plotly 코드를 선호하지만 matplotlib도 사용할 수 있습니다."
    # "<시각화 환경 설정>"
    # "- [중요] 시각화 제목과 레이블에는 `영어`를 사용하십시오."
    # "- 시각화에는 `muted` cmap, white background, 그리드 없음."
    # "seaborn plot에 cmap, 팔레트 매개변수를 설정하는 것이 좋습니다."
    # "최종 답변의 언어는 한국어로 작성해야 합니다."
    # "<Column 가이드라인>"
    # "사용자가 `df.columns`에 나열되지 않은 열로 질문하는 경우 아래에 나열된 가장 유사한 열을 참조할 수 있습니다.

# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.

    Args:
        query (str): 사용자의 질문
    """
    if "agent" in session_state:
        st.chat_message("user").write(query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = session_state["agent"]
        response = agent.stream({"input": query})

        ai_answer = ""
        parser_callback = AgentCallbacks(tool_callback, observation_callback, result_callback)
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message("assistant"):
            for step in response:
                stream_parser.process_agent_steps(step)
                if "output" in step:
                    ai_answer += step["output"]
            st.write(ai_answer)

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])

################################ ~에이전트 ################################

#################################### 채팅 ####################################

# 메인 로직
if clr_btn:
    session_state["messages_csv"] = []  # 대화 내용 초기화

if apply_btn and uploaded_file:
    # if isinstance(uploaded_file, list):
    #     uploaded_file = uploaded_file[0]
    df = pd.read_csv(uploaded_file)  # CSV 파일 로드
    session_state["df"] = df  # 데이터프레임 저장
    session_state["python_tool"] = PythonAstREPLTool()  # Python 실행 도구 생성
    session_state["python_tool"].locals["df"] = df  # 데이터프레임을 Python 실행 환경에 추가
    session_state["agent"] = create_csv_agent(df)   # 랭체인에서 제공하는 "데이터프레임 에이전트" 생성 후 세션 스테이트에 저장

    st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")
elif apply_btn:
    st.warning("파일을 업로드 해주세요.")

print_messages()  # 이전까지 session_state["messages"]에 저장된 메시지 출력

user_input = st.chat_input("무엇이 궁금하신가요?")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리

# titanic.csv 파일 업로드
# 행의 개수를 출력해줘
# 남녀 각각 생존율을 bar chart로 그려줘
# 상위 10개의 행을 출력해줘
# 남녀 구분해서 charges 상자그림 그리고 차이에 대해 해석해줘

