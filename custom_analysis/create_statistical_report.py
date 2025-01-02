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

# if "input_query" not in session_state:
#     session_state["input_query"] = {}

# if "store" not in session_state:
#     session_state["store"] = {}

# if "chain" not in session_state:
#     session_state["chain"] = None

# if "df" not in session_state:
#     session_state["df"] = None

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


# 이전까지의 대화 출력 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):                 # 예상치 못한 데이터 타입 입력 방지
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
    messages = session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다

    
################## 에이전트 ##################

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
            if query:    # ex) len(df)
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
            # st.error(obs)   # 에러 출력 
            session_state["messages"][-1][1].clear()  # 에러 발생 시 마지막 메시지 삭제
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

############################ 채팅 ############################

st.title("생성형 AI 데이터 분석")
# st.markdown("###### 자세히 입력할수록 보고서의 퀄리티는 높아집니다.")
st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기

############################ tab 분류 ############################

# 탭을 생성
list_of_tabs = ["데이터 업로드", "보고서 작성"]
tab1, tab2 = st.tabs(list_of_tabs)

with tab1:
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

with tab2:              
    st.markdown("<br><br>", unsafe_allow_html=True)   # 줄 띄어쓰기
    analysis_comment = st.text_area(
            "어떤 분석을 하고 싶은지 자유롭게 설명해주세요", 
            placeholder="집계와 시각화를 통해 EDA를 하고 싶어. 그리고 charges 변수를 선형회귀 예측하고 유의미한 변수가 무엇인지 파악해줘.",
            height=50) 
    
    strt_btn = st.button("보고서 작성")
    if strt_btn:
        if uploaded_file:
            session_state["df"] = df  # 데이터프레임 저장
            session_state["python_tool"] = PythonAstREPLTool()  # Python 실행 도구 생성
            session_state["python_tool"].locals["df"] = df  # 데이터프레임을 Python 실행 환경에 추가
            session_state["agent"] = create_csv_agent(df)   # 랭체인에서 제공하는 "데이터프레임 에이전트" 생성 후 세션 스테이트에 저장

            st.write("데이터 상위 5개 행을 출력합니다")
            st.dataframe(df.head())
            # st.warning("준비중입니다")

            # 사용자 입력(input_data)에 의한 프롬프트 작성
            # st.write(session_state["input_data"])
            prompt = f"""너는 친절한 어시스턴트 챗봇이야.
{analysis_comment}. 한국어로 작성해줘.
"""
            # st.write(prompt)

            st.chat_message("user").write(prompt)
            add_message(MessageRole.USER, [MessageType.TEXT, prompt])

            agent = session_state["agent"]
            response = agent.stream({"input": prompt})
            # response = agent.invoke({"input": query})
            # st.markdown(session_state)

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
                
        else:
            st.warning("파일을 업로드 해주세요")
       

# 선형회귀 분석과 유의한 변수 파악

# query 예시
# 나는 남성이고 30대살의 제조업 산업에서 기획자로 일하고 있어. ['시각화', '통계 검정'] sikll은 꼭 사용하면 좋겠고, 분석 목적은 ['변수 인과관계 설명']이야. 
# 다음에 언급된 내용들에 대해 중점적으로 분석 및 해석해줘 : 용어 수준은 Very Easy하게, 보고서 페이지 수는 2장 작성해주고, 어조는 "비즈니스적인" 으로 해줘. Korean로 작성해줘.

# 나는 30대 남성이고 제조업에서 기획자로 일하고 하고 있어.
# 집계와 시각화, 그리고 통계 분석을 수행하고자 해.
# 분석 목적은 charges에 대한 선형회귀 예측과 통계적으로 유의한 변수 파악이야.
# 먼저 charges 분석을 하고싶어. 히스토그램으로 분포를 파악해줘.
# 성별 차이에 따른 charges 분포 비교를 해줘.
# charges와 다른 변수들 간의 상관계수 히트맵을 그리고 해석해줘.
# charges 선형회귀를 한 후에 유의한 변수가 무엇이 있는지 찾고 통계적 이유를 들어줘.
# 용어 난이도는 very easy, 보고서는 A4용지 2page 정도, 어조는 친근하게, 한국어로 답변해줘.
# 양식은 회사 보고서 양식으로 해줘

