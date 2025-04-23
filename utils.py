import yaml
import pickle
import os
import streamlit as st

def load_users():
    try:
        with open('users.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data.get('credentials', {}).get('usernames', {})
    except Exception as e:
        st.error(f"加载用户数据出错: {e}")
        # 返回一个默认用户，以便在出错时仍能登录
        return {
            "admin": {
                "name": "管理员",
                "password": "admin",
                "department": "管理部",
                "role": "管理员",
                "knowledge_access": ["技术部", "市场部", "管理部", "财务部"]
            }
        }

def save_session(session_data):
    with open('.session_data.pkl', 'wb') as f:
        pickle.dump(session_data, f)

def load_session():
    if os.path.exists('.session_data.pkl'):
        with open('.session_data.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

def local_css():
    css = """
    <style>
    /* 全局样式 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #1E293B;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #F8FAFC;
    }
    
    /* 卡片样式 */
    .dashboard-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #F1F5F9;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    .dashboard-title {
        font-size: 18px;
        margin-bottom: 16px;
        color: #0F172A;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 10px;
    }
    
    /* 聊天界面样式 */
    .chat-message {
        margin-bottom: 15px;
        padding: 12px;
        border-radius: 10px;
        max-width: 80%;
        animation: fadeIn 0.3s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background-color: #EFF6FF;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        border: 1px solid #DBEAFE;
    }
    
    .bot-message {
        background-color: white;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid #E2E8F0;
    }
    
    .message-header {
        font-size: 12px;
        margin-bottom: 5px;
        color: #64748B;
    }
    
    .message-content {
        font-size: 14px;
        line-height: 1.5;
        color: #1E293B;
    }
    
    /* 文件上传区域 */
    .file-upload {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #F8FAFC;
        transition: all 0.3s;
    }
    
    .file-upload:hover {
        border-color: #94A3B8;
        background-color: #F1F5F9;
    }
    
    /* 文件列表 */
    .file-item {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #E2E8F0;
        transition: all 0.2s;
    }
    
    .file-item:hover {
        background-color: #F8FAFC;
        border-color: #CBD5E1;
    }
    
    /* 登录页面 */
    .login-card {
        background-color: white;
        border-radius: 16px;
        padding: 30px;
        margin-top: 80px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #F1F5F9;
    }
    
    .logo-container {
        width: 80px;
        height: 80px;
        margin: 0 auto 20px;
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 15px rgba(59, 130, 246, 0.3);
    }
    
    .logo-text {
        color: white;
        font-size: 24px;
        font-weight: 700;
    }
    
    .main-title {
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
        color: #0F172A;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB, #1E40AF);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-2px);
    }
    
    /* 搜索结果样式 */
    .search-result {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #E2E8F0;
        transition: all 0.2s;
    }
    
    .search-result:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .search-result h4 {
        color: #1E40AF;
        margin-bottom: 10px;
    }
    
    .search-result p {
        font-size: 14px;
        color: #334155;
        margin-bottom: 10px;
    }
    
    .search-result small {
        color: #64748B;
        font-size: 12px;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background-color: #3B82F6;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True) 

def save_uploaded_file(uploaded_file, directory, filename):
    """保存上传的文件到指定目录"""
    import os
    
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 构建文件路径
    file_path = os.path.join(directory, filename)
    
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path 