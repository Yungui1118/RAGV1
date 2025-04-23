import streamlit as st
import os

def navigation():
    # 创建一个行布局，用于放置用户信息和退出按钮在同一行
    cols = st.sidebar.columns([8, 2])
    
    # 用户信息和退出按钮在同一行
    with cols[0]:
        st.markdown(f"### 👤 欢迎, {st.session_state['user_name']}")
    
    with cols[1]:
        # 使用更明确的关机图标
        if st.button("❌", key="logout", help="退出系统"):
            # 清除会话状态
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # 删除保存的会话文件
            if os.path.exists('.session_data.pkl'):
                os.remove('.session_data.pkl')
            
            # 重新加载页面
            st.rerun()
    
    # 部门信息单独一行
    st.sidebar.markdown(f"#### 🏢 {st.session_state['user_department']}")
    
    # 添加分隔线
    st.sidebar.divider()
    
    # 导航菜单标题
    st.sidebar.title("🧭 导航菜单")
    
    # 导航链接 - 移除知识检索
    nav_items = {
        '💬 智能问答': 'qa',
        '📄 知识库管理': 'documents'
    }
    
    # 当前页面 - 默认为智能问答
    current_page = st.session_state.get('current_page', 'qa')
    
    # 渲染导航链接
    for label, page in nav_items.items():
        # 高亮当前页面
        if current_page == page:
            button_label = f"**{label}**"
        else:
            button_label = label
            
        if st.sidebar.button(button_label, key=f"nav_{page}"):
            st.session_state['current_page'] = page
            st.rerun() 