import streamlit as st
import time
import random

def analytics_page():
    st.title("📊 数据分析")
    st.write("分析知识库使用情况，了解热门话题和知识缺口")
    
    # 时间范围选择
    col1, col2 = st.columns([1, 3])
    with col1:
        time_range = st.selectbox("📅 时间范围", ["过去7天", "过去30天", "过去90天", "全部时间"], index=1)
    
    # 知识库使用情况
    st.subheader("📈 知识库使用情况")
    
    # 模拟数据
    chart_data = {
        '技术部': 45,
        '市场部': 30,
        '管理部': 15,
        '财务部': 10
    }
    
    st.bar_chart(chart_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("各部门知识库访问占比")
    with col2:
        st.caption("总访问量: 1,245次")
    
    # 用户活跃度和热门话题
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 用户活跃度")
        
        # 模拟用户活跃度数据
        active_users = [
            {"name": "张三", "department": "技术部", "queries": 78},
            {"name": "李四", "department": "市场部", "queries": 65},
            {"name": "王五", "department": "管理部", "queries": 52},
            {"name": "赵六", "department": "技术部", "queries": 45},
            {"name": "钱七", "department": "财务部", "queries": 38}
        ]
        
        for user in active_users:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{user['name']}**")
                    st.caption(f"{user['department']}")
                with col2:
                    st.info(f"{user['queries']}次")
    
    with col2:
        st.subheader("🔍 热门搜索关键词")
        
        hot_topics = [
            {"keyword": "项目管理", "count": 120},
            {"keyword": "技术文档", "count": 95},
            {"keyword": "市场分析", "count": 78},
            {"keyword": "财务报表", "count": 65},
            {"keyword": "产品规划", "count": 52}
        ]
        
        for topic in hot_topics:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{topic['keyword']}**")
                st.progress(topic['count']/120)
            with col2:
                st.write(f"{topic['count']}次")
    
    # 知识缺口分析
    st.subheader("🧩 知识缺口分析")
    st.write("以下是系统检测到的可能存在知识缺口的领域，建议补充相关文档：")
    
    gap_topics = [
        {"topic": "云原生架构", "confidence": 85, "queries": 42},
        {"topic": "数据安全合规", "confidence": 78, "queries": 36},
        {"topic": "敏捷开发流程", "confidence": 72, "queries": 29},
        {"topic": "客户满意度调研", "confidence": 65, "queries": 24}
    ]
    
    for topic in gap_topics:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{topic['topic']}**")
                st.caption(f"相关查询: {topic['queries']}次")
                st.write("用户经常查询此主题，但知识库中相关文档较少或质量不高。")
            
            with col2:
                if topic['confidence'] > 80:
                    st.success(f"置信度: {topic['confidence']}%")
                elif topic['confidence'] > 70:
                    st.warning(f"置信度: {topic['confidence']}%")
                else:
                    st.error(f"置信度: {topic['confidence']}%") 