import streamlit as st
from PIL import Image

st.set_page_config(page_title="ReportGenie - Auto Report Agent", page_icon="resource/리포트지니_로고_세줄_원_blue_rbg.png")

# # Home pages
# intro = st.Page("Home/intro.py", title="ReportGenie", icon=":material/house:", 
#                     default=True)

# Category2 pages
reportgenie_analysis = st.Page("custom_analysis/create_statistical_report.py", title="Gen-AI 데이터 분석", icon=":material/search:")
 
dashboard = st.Page("dashboard/wine.py", title="dashboard")

# # Category1 pages
# PDF_analysis = st.Page("fast_analysis/01_PDF 분석기.py", title="PDF_analysis", icon=":material/dashboard:")
# IMAGE_analysis = st.Page("fast_analysis/02_IMAGE 분석기.py", title="IMAGE_analysis", icon=":material/bug_report:")
# CSV_analysis = st.Page("fast_analysis/03_CSV 분석기.py", title="CSV_analysis", icon=":material/notification_important:")
 
# 여러 개의 st.Page 객체를 묶어서 내비게이션 메뉴 생성
pages_navi = st.navigation(
    {
        # "Home": [intro],
        "대시보드" : [dashboard],
        "AI 데이터 분석": [reportgenie_analysis],
        # "빠른 분석봇": [PDF_analysis, IMAGE_analysis, CSV_analysis]
    }
)

pages_navi.run()


