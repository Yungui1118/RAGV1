#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform

def print_step(message):
    """打印带有格式的步骤信息"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def run_command(command):
    """运行命令并实时显示输出"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def check_python_version():
    """检查Python版本"""
    print_step("检查Python版本")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"错误: 需要Python 3.8或更高版本，当前版本为{major}.{minor}")
        return False
    
    print(f"Python版本检查通过: {major}.{minor}")
    return True

def create_directories():
    """创建必要的目录结构"""
    print_step("创建项目目录结构")
    
    directories = [
        "knowledge_bases",
        "logs",
        "uploads"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        else:
            print(f"目录已存在: {directory}")
    
    return True

def configure_pip_mirror():
    """配置pip使用清华源"""
    print_step("配置pip使用清华源")
    
    pip_conf_dir = os.path.expanduser("~/.pip")
    if not os.path.exists(pip_conf_dir):
        os.makedirs(pip_conf_dir)
    
    pip_conf_file = os.path.join(pip_conf_dir, "pip.conf")
    
    # 检查是否已经配置了清华源
    if os.path.exists(pip_conf_file):
        with open(pip_conf_file, "r") as f:
            content = f.read()
            if "tsinghua" in content:
                print("pip已配置清华源")
                return True
    
    # 写入清华源配置
    with open(pip_conf_file, "w") as f:
        f.write("""[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
""")
    
    print("已配置pip使用清华源")
    return True

def print_conda_mirror_instructions():
    """打印conda配置清华源的说明"""
    print_step("Conda清华源配置说明")
    
    print("""
要配置conda使用清华源，请手动执行以下命令：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

对于xinference环境，创建后可以执行：

conda activate xinference
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "xinference[all]"
""")
    
    return True

def install_dependencies():
    """安装依赖包"""
    print_step("安装依赖包")
    
    # 升级pip
    print("升级pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # 安装requirements.txt中的依赖
    if os.path.exists("requirements.txt"):
        print("从requirements.txt安装依赖...")
        result = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if result != 0:
            print("警告: 安装依赖时出现错误，尝试单独安装关键依赖...")
            
            # 安装关键依赖
            key_packages = [
                "streamlit==1.28.0",
                "pyyaml==6.0",
                "pandas==2.0.3",
                "numpy==1.24.3",
                "requests==2.31.0",
                "httpx==0.24.1",
                "openai==0.28.0"
            ]
            
            for package in key_packages:
                print(f"安装 {package}...")
                run_command(f"{sys.executable} -m pip install {package}")
            
            # 尝试安装faiss-cpu
            print("安装 faiss-cpu...")
            faiss_result = run_command(f"{sys.executable} -m pip install faiss-cpu==1.7.4")
            if faiss_result != 0:
                print("警告: 安装faiss-cpu失败，尝试安装替代版本...")
                run_command(f"{sys.executable} -m pip install faiss-cpu")
    else:
        print("错误: requirements.txt文件不存在")
        return False
    
    return True

def create_test_users():
    """创建测试用户配置"""
    print_step("创建测试用户配置")
    
    if not os.path.exists("users.yaml"):
        users_config = """credentials:
  usernames:
    admin:
      name: 管理员
      password: admin
      department: 管理部
      role: 管理员
      knowledge_access: [技术部, 市场部, 管理部, 财务部]
    user1:
      name: 技术人员
      password: user1
      department: 技术部
      role: 普通用户
      knowledge_access: [技术部]
    user2:
      name: 市场人员
      password: user2
      department: 市场部
      role: 普通用户
      knowledge_access: [市场部]
"""
        with open("users.yaml", "w", encoding="utf-8") as f:
            f.write(users_config)
        print("已创建测试用户配置文件: users.yaml")
    else:
        print("用户配置文件已存在: users.yaml")
    
    return True

def create_xinference_setup_script():
    """创建xinference环境安装脚本"""
    print_step("创建xinference环境安装脚本")
    
    script_content = """#!/bin/bash
# xinference环境安装脚本

# 配置conda使用清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# 创建xinference环境
conda create -n xinference python=3.10 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xinference

# 配置pip使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装xinference
pip install "xinference[all]"

# 验证安装
xinference --version

echo "xinference环境安装完成"
echo "使用方法:"
echo "1. 激活环境: conda activate xinference"
echo "2. 启动服务: xinference --host 0.0.0.0 --port 9997"
"""
    
    with open("setup_xinference.sh", "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("setup_xinference.sh", 0o755)
    
    print("已创建xinference环境安装脚本: setup_xinference.sh")
    print("使用方法: bash setup_xinference.sh")
    
    return True

def main():
    """主函数"""
    print("\n欢迎使用企业知识库系统安装脚本\n")
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 配置pip使用清华源
    configure_pip_mirror()
    
    # 打印conda配置清华源的说明
    print_conda_mirror_instructions()
    
    # 创建目录结构
    if not create_directories():
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies():
        print("警告: 依赖安装可能不完整，请检查错误信息")
    
    # 创建测试用户
    create_test_users()
    
    # 创建xinference环境安装脚本
    create_xinference_setup_script()
    
    print("\n" + "=" * 80)
    print("  安装完成！")
    print("=" * 80)
    print("\n启动应用程序:")
    print("1. 安装并启动xinference服务(在另一个终端):")
    print("   bash setup_xinference.sh")
    print("   conda activate xinference")
    print("   xinference --host 0.0.0.0 --port 9997")
    print("\n2. 启动Streamlit应用:")
    print("   streamlit run app.py")
    print("\n默认登录信息:")
    print("   用户名: admin")
    print("   密码: admin")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main() 