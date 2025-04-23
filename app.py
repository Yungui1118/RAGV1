import streamlit as st
from navigation import navigation
from login import login_page
from documents import documents_page
from qa import qa_page
from utils import load_session, save_session, local_css

# 设置页面配置
st.set_page_config(
    page_title="企业RAG系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

def main():
    # 添加CSS
    local_css()
    
    # 初始化会话状态
    if 'logged_in' not in st.session_state:
        # 尝试从文件加载会话状态
        saved_session = load_session()
        for key, value in saved_session.items():
            st.session_state[key] = value
    
    # 根据登录状态显示不同页面
    if st.session_state.get('logged_in', False):
        # 显示导航栏
        navigation()
        
        # 根据当前页面显示不同内容
        current_page = st.session_state.get('current_page', 'qa')  # 默认为智能问答
        
        if current_page == 'documents':
            documents_page()
        elif current_page == 'qa':
            qa_page()
    else:
        login_page()

if __name__ == "__main__":
    main()