import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# # Streamlit Page Configuration
# st.set_page_config(
#     page_title="ReportGenie - An ai auto report agent",
#     page_icon="resource/리포트지니_로고2_rbg.png",
#     layout="wide",
#     initial_sidebar_state="expanded",   # collapsed, expanded
# )

# # 로고 이미지를 표시 (이미지 파일 경로 또는 URL)
# logo_path = "resource/리포트지니_로고_세줄_원_blue_rbg.png"  # 로컬 파일 경로 또는 URL
# st.sidebar.image(logo_path, use_column_width=True)
# st.markdown("<br><br>", unsafe_allow_html=True)   # 줄 띄어쓰기

# # 이미지 불러오기
# image = Image.open("resource/female_genie.jpg")

# # 원래 크기에서 70%로 줄이기
# width, height = image.size
# new_size = (int(width * 0.5), int(height * 0.5))
# resized_image = image.resize(new_size)

# # 조정된 이미지 표시
# st.image(resized_image, caption="보고서 자동화 솔루션, 리포트지니")
# st.markdown("<br><br>", unsafe_allow_html=True)   # 줄 띄어쓰기

############################ /사이드바 구성 ############################

############################## 본문 내용 ##############################

st.title("ReportGenie")

# 꼭지 1 제목
st.html("""
<h4> 1. 시간 절약과 효율성 극대화</h4>
""")

# 꼭지 1 내용
st.markdown("""
- <p style="font-size:20px; color:gray;">리포트지니는 데이터를 입력하면 AI가 자동으로 분석, 시각화, 및 리포트 작성을 수행하므로 데이터 분석과 문서화에 소요되는 시간을 획기적으로 줄일 수 있습니다.</p>
- <p style="font-size:20px; color:gray;">반복적이고 단순한 작업을 자동화해 데이터 전문가가 더 중요한 의사결정 및 전략 수립에 집중할 수 있도록 돕습니다.</p>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기

# 꼭지 2 제목
st.html("""
<h4> 2. 일관되고 전문적인 리포트 작성</h4>
""")

# 꼭지 2 내용
st.markdown("""
- <p style="font-size:20px; color:gray;">리포트지니는 통계, 머신러닝, 데이터 시각화 기술을 기반으로 한 맞춤형 분석 리포트를 생성하여, 분석 결과가 명확하고 직관적으로 전달되도록 합니다.</p>
- <p style="font-size:20px; color:gray;">PDF, 이미지, CSV 등 다양한 포맷의 데이터를 처리하며, 데이터의 특징에 맞는 최적의 분석 결과와 해석을 제공합니다.</p>
- <p style="font-size:20px; color:gray;">회사나 팀의 요구에 맞춘 포맷이나 톤을 유지해 전문적인 보고서를 작성합니다.</p>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)   # 줄 띄어쓰기

# 꼭지 3 제목
st.html("""
<h4> 3. 데이터 인사이트를 빠르게 도출</h4>
""")

# 꼭지 3 내용
st.markdown("""
- <p style="font-size:20px; color:gray;">리포트지니는 대규모 데이터를 신속하게 분석하여 통찰력 있는 인사이트를 제공합니다.</p>
- <p style="font-size:20px; color:gray;">사용자는 복잡한 데이터를 쉽게 이해하고 의사결정을 내릴 수 있으며, 분석에 필요한 추가 조치를 제안받을 수도 있습니다.</p>
- <p style="font-size:20px; color:gray;">데이터 기반 인사이트는 사업 전략 수립, 문제 해결, 트렌드 파악에 큰 도움을 줍니다.</p>
""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)   # 줄 띄어쓰기

############################## /본문 내용 ##############################

# # 빈 열을 활용해 오른쪽 정렬
# col1, col2 = st.columns([3, 1])  # 왼쪽에 더 많은 공간

# # 버튼을 누르면 페이지 이동
# col2.page_link("pages/00_리포트포유.py", label="보고서 작성", icon="➡️")




