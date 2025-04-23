import streamlit as st
import time
import random

def dashboard_page():
    # 欢迎信息
    st.markdown(f"""
    <h2 style="margin-bottom: 5px;">欢迎, {st.session_state['user_name']}</h2>
    <p style="color: #64748B; margin-bottom: 20px;">今天是 {time.strftime("%Y年%m月%d日")} · {st.session_state['user_department']}</p>
    """, unsafe_allow_html=True)
    
    # 系统概览
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="dashboard-title">系统概览</h3>', unsafe_allow_html=True)
    
    # 统计数据
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 15px;">
            <div style="font-size: 36px; font-weight: 600; color: #3B82F6;">124</div>
            <div style="color: #64748B;">知识库文档</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 15px;">
            <div style="font-size: 36px; font-weight: 600; color: #10B981;">568</div>
            <div style="color: #64748B;">本周查询次数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px;">
            <div style="font-size: 36px; font-weight: 600; color: #8B5CF6;">92%</div>
            <div style="color: #64748B;">用户满意度</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 知识库访问权限
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="dashboard-title">您的知识库访问权限</h3>', unsafe_allow_html=True)
    
    for dept in st.session_state['user_knowledge_access']:
        doc_count = random.randint(15, 50)
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500;">{dept}</span>
                <span>{doc_count} 文档</span>
            </div>
            <div style="height: 8px; background-color: #F1F5F9; border-radius: 4px;">
                <div style="height: 100%; width: {doc_count*2}%; background-color: #3B82F6; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 功能卡片
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="dashboard-title">文档管理</h3>', unsafe_allow_html=True)
        st.write("上传、组织和管理企业文档，支持多种格式的文件。")
        if st.button("进入文档管理", key="goto_docs"):
            st.session_state['current_page'] = 'documents'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="dashboard-title">智能问答</h3>', unsafe_allow_html=True)
        st.write("基于企业知识库的AI助手，回答您的业务问题。")
        if st.button("开始提问", key="goto_qa"):
            st.session_state['current_page'] = 'qa'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="dashboard-title">知识检索</h3>', unsafe_allow_html=True)
        st.write("使用自然语言查询企业知识库，获取精准的信息。")
        if st.button("搜索知识", key="goto_search"):
            st.session_state['current_page'] = 'search'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="dashboard-title">数据分析</h3>', unsafe_allow_html=True)
        st.write("分析知识库使用情况，了解热门话题和知识缺口。")
        if st.button("查看分析", key="goto_analytics"):
            st.session_state['current_page'] = 'analytics'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True) 