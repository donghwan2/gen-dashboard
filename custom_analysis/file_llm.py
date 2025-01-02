import streamlit as st
from retriever import create_retriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_teddynote.models import MultiModal   # 이미지 인식

llm_model = "gpt-4o"    # "gpt-4o-mini"

# pdf 파일 업로드 시 retriever 생성
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    return create_retriever(file_path)

# llm_chain 생성 함수
def create_chain(retriever):
    # prompt | llm | output_parser
    # 프롬프트 적용
    # prompt = load_prompt(selected_prompt, encoding="utf-8")

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    # Context: 
    {context}

    # Question:
    {question}

    # Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=llm_model, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# pdf가 업로드되면 rag chain을 생성하는 함수
def pdf_chain(uploaded_file):
    # 파일 업로드 후 retriever 생성(작업 시간이 오래 걸릴 예정), chain 생성
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever)
    st.session_state["chain_pdf"] = chain
    return chain

# 이미지가 업로드 되었을 때, 이미지를 캐시 저장(시간이 오래 걸리는 작업에 사용)
# /.cache/files" 디렉토리 안에 파일이 캐시 저장됨.
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")   # 업로드 파일을 캐싱해서 가지고 있는다.
def process_imagefile(uploaded_file):
    print("Processing image file...")
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = uploaded_file.read()
    file_path = f"./.cache/files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

# multimodal chain 생성 함수
def multimodal_answer(image_filepath, user_prompt):
    
    # 시스템 프롬프트 입력받기  
    system_prompt = """
    당신은 표를 분석하는 AI 어시스턴트입니다. 
    당신의 임무는 주어진 테이블을 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다. 한국어로 대답하세요."""
    
    # 객체 생성
    llm = ChatOpenAI(
        model_name=llm_model,
        temperature=0          # 창의성 (0.0 ~ 2.0)
    ) 

    # 멀티모달 객체 생성
    multimodal = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    # 이미지 파일로 부터 질의(스트림 방식)
    response = multimodal.stream(image_filepath)

    return response

