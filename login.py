import streamlit as st
import time
import os
from utils import load_users, save_session

def login_page():
    # 设置页面背景和文字样式 - 使用更具体的选择器并添加!important标记
    st.markdown("""
        <style>
            /* 页面背景 */
            .stApp {
                background-color: #003366 !important;
            }
          
            /* 标题文字 */
            .main h1, .main h2, .main h3, .main p {
                color: white !important;
                text-align: center !important;
            }
            
            /* 输入框样式 */
            .stTextInput > div > div > input {
                color: #333333 !important;
                border-radius: 5px !important;
                background-color: white !important;
            }
            
            /* 复选框文字颜色 */
            .stCheckbox > div > div > label,
            .stCheckbox label,
            .stCheckbox span,
            div[data-testid="stCheckbox"] label,
            div[data-testid="stCheckbox"] span {
                color: white !important;
            }
            
            /* 复选框本身的样式 */
            .stCheckbox input[type="checkbox"] {
                border-color: white !important;
            }
            
            /* 复选框勾选时的颜色 */
            .stCheckbox input[type="checkbox"]:checked {
                background-color: #4CAF50 !important;
            }
            
            /* 标签文字颜色 */
            .stTextInput > label, .stCheckbox > label {
                color: white !important;
                font-weight: 500 !important;
            }
            
            /* 按钮样式 */
            .stButton > button {
                background-color: #4CAF50 !important;
                color: white !important;
                font-weight: bold !important;
                border: none !important;
                border-radius: 5px !important;
            }
            
            /* 提示框样式 - 使用更具体的选择器 */
            div[data-baseweb="notification"] {
                background-color: white !important;
                color: #003366 !important;
                border-radius: 5px !important;
                border: none !important;
            }
            
            /* 加载中动画颜色 */
            .stSpinner > div > div > div {
                border-color: white transparent transparent transparent !important;
            }
            
            /* 加载中文字颜色 */
            .stSpinner > div > span {
                color: white !important;
            }
            
            /* Logo居中显示 */
            .logo-container {
                display: flex;
                justify-content: center;
                margin-bottom: 20px;
            }
            
            /* 全局强制所有复选框相关文本为白色 */
            .stCheckbox *,
            [data-testid="stCheckbox"] *,
            div:has(> input[type="checkbox"]) *,
            label:has(+ input[type="checkbox"]),
            input[type="checkbox"] ~ * {
                color: white !important;
            }
            
            /* 复选框本身的样式 */
            input[type="checkbox"] {
                border-color: white !important;
                accent-color: #4CAF50 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 检查SVG文件是否存在
    logo_path = "p.svg"
    if os.path.exists(logo_path):
        # 读取SVG文件内容
        with open(logo_path, "r") as f:
            svg_content = f.read()
        
        # 设置SVG大小和颜色
        svg_content = svg_content.replace('<svg', '<svg width="120" height="120" fill="white"')
        
        # 显示SVG logo
        st.markdown(f'<div class="logo-container">{svg_content}</div>', unsafe_allow_html=True)
    else:
        # 如果SVG文件不存在，显示文字logo
        st.markdown("<h1 style='color: white; text-align: center;'>🏢</h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='color: white; text-align: center;'>企业知识管理系统</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown("<h3 style='color: white; text-align: center;'>用户登录</h3>", unsafe_allow_html=True)
        
        username = st.text_input("用户名", placeholder="请输入工号或邮箱")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        
        remember_me = st.checkbox("记住我", value=True)
        
        if st.button("登录", use_container_width=True):
            if username and password:
                users = load_users()
                
                if username in users:
                    stored_password = str(users[username].get('password', ''))
                    input_password = str(password)
                    
                    if stored_password == input_password:
                        with st.spinner("登录中，请稍候..."):
                            time.sleep(1)
                        
                        session_data = {
                            'logged_in': True,
                            'username': username,
                            'user_name': users[username].get('name', username),
                            'user_department': users[username].get('department', '未知部门'),
                            'user_role': users[username].get('role', '普通用户'),
                            'user_knowledge_access': users[username].get('knowledge_access', []),
                            'current_page': 'qa'
                        }
                        
                        for key, value in session_data.items():
                            st.session_state[key] = value
                        
                        if remember_me:
                            save_session(session_data)
                        
                        # 在每次显示消息前重新应用样式
                        st.markdown("""
                            <style>
                            div[data-baseweb="notification"] {
                                background-color: white !important;
                                color: #003366 !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        st.warning("登录成功，即将跳转...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        # 在每次显示消息前重新应用样式
                        st.markdown("""
                            <style>
                            div[data-baseweb="notification"] {
                                background-color: white !important;
                                color: #003366 !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        st.error("密码错误，请重新输入。")
                else:
                    # 在每次显示消息前重新应用样式
                    st.markdown("""
                        <style>
                        div[data-baseweb="notification"] {
                            background-color: white !important;
                            color: #003366 !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    st.error("用户不存在，请检查用户名。")
            else:
                # 在每次显示消息前重新应用样式
                st.markdown("""
                    <style>
                    div[data-baseweb="notification"] {
                        background-color: white !important;
                        color: #003366 !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.warning("请完整填写用户名和密码。")

    st.markdown("<p style='color:#cccccc; text-align:center;'>© 2025 企业知识管理系统 v1.0</p>", unsafe_allow_html=True)
