import streamlit as st

def search_page():
    st.title("🔍 知识检索")
    st.write("使用自然语言查询企业知识库")
    
    # 搜索框
    search_query = st.text_input("🔎 输入您的搜索关键词", placeholder="例如：项目管理、技术文档...", key="search_input")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # 部门筛选
        departments = st.session_state['user_knowledge_access']
        selected_departments = st.multiselect("🏢 选择部门", departments, default=departments)
    
    with col2:
        # 文档类型筛选
        doc_types = ["PDF", "Word", "文本文件"]
        selected_types = st.multiselect("📑 文档类型", doc_types, default=doc_types)
    
    # 搜索按钮
    search_button = st.button("🚀 搜索", key="search_button")
    
    if search_button and search_query:
        st.divider()
        st.subheader("🔍 搜索结果")
        
        # 显示搜索信息
        st.info(f"""
        **搜索关键词:** {search_query}
        **选择的部门:** {', '.join(selected_departments)}
        **选择的文档类型:** {', '.join(selected_types)}
        """)
        
        # 模拟结果列表
        for i in range(3):
            relevance = 90-i*10
            
            with st.container():
                st.write(f"### 📄 示例文档 {i+1}")
                st.write(f"这是一个与\"{search_query}\"相关的文档摘要。这里显示文档的相关内容片段，包含关键词和上下文信息，帮助用户快速了解文档内容...")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"来源: 部门文档 | 类型: PDF | 上传时间: 2023-10-{15+i}")
                with col2:
                    if relevance > 80:
                        st.success(f"相关度: {relevance}%")
                    elif relevance > 60:
                        st.warning(f"相关度: {relevance}%")
                    else:
                        st.error(f"相关度: {relevance}%")
                
                st.divider() 