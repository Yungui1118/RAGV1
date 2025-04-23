import streamlit as st
import os
import pandas as pd
import time
import uuid
import json
import pickle
import numpy as np
from utils import save_uploaded_file
from datetime import datetime
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# 添加日志函数
def log_info(message):
    """打印带时间戳的日志信息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] INFO: {message}")

def log_error(message):
    """打印带时间戳的错误信息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] ERROR: {message}")

# 尝试导入faiss，如果不可用则设置标志
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# 尝试导入xinference，如果不可用则使用模拟实现
try:
    from xinference.client import RESTfulClient
    XINFERENCE_AVAILABLE = True
except ImportError:
    XINFERENCE_AVAILABLE = False
    # 创建一个模拟的RESTfulClient类
    class MockRESTfulClient:
        def __init__(self, url):
            self.url = url

        def list_models(self):
            return []

        def launch_model(self, model_name, model_type):
            return {"model_id": "mock-model-id"}

    RESTfulClient = MockRESTfulClient

# 向量处理函数
def process_document_vectors(kb_name, document_row, client):
    """
    处理单个文档的向量化

    参数:
        kb_name: 知识库名称
        document_row: 文档元数据行
        client: xinference客户端

    返回:
        处理成功返回True，否则返回False
    """
    try:
        # 获取文件路径和元数据
        file_path = document_row["file_path"]
        file_type = document_row["file_type"]
        department = document_row["department"]
        document_title = document_row["title"]

        # 读取文件内容
        content = read_file_content(file_path, file_type)
        if not content:
            return False

        # 使用固定的最佳参数
        chunk_size = 300  # 较大的分块大小，保持语义完整性
        chunk_overlap = 100  # 适当的重叠，确保上下文连贯
        zh_title_enhance = True  # 启用中文标题增强

        # 分块处理
        chunks = split_text_into_chunks(content, chunk_size, chunk_overlap, zh_title_enhance)

        # 为每个块生成向量
        documents = []
        for chunk in chunks:
            # 获取向量
            vector = get_embedding(chunk)

            # 创建文档对象
            doc = {
                "text": chunk,
                "vector": vector,
                "document_title": document_title,
                "department": department,
                "file_type": file_type,
                "source": file_path
            }
            documents.append(doc)

        # 保存向量到文件
        if documents:
            vectors_dir = os.path.join("knowledge_bases", kb_name, "vectors")
            os.makedirs(vectors_dir, exist_ok=True)

            # 使用UUID作为文件名
            vector_file = os.path.join(vectors_dir, f"{uuid.uuid4()}.json")

            # 保存向量文件
            with open(vector_file, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)

            log_info(f"向量文件已保存: {vector_file}，包含 {len(documents)} 个文档")

            # 构建FAISS索引
            if build_faiss_index(kb_name):
                log_info(f"成功构建FAISS索引")
            else:
                log_error(f"构建FAISS索引失败")

        return True
    except Exception as e:
        log_error(f"处理文档向量时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def split_text_into_chunks(text, chunk_size=300, chunk_overlap=100, zh_title_enhance=True):
    """
    将文本分割成更有语义意义的块

    参数:
        text: 要分割的文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        zh_title_enhance: 是否增强中文标题识别
    """
    if not text:
        return []

    # 按段落分割
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 跳过空段落
        if not para.strip():
            continue

        # 检测是否为标题（短且以特定符号结尾）
        is_title = len(para.strip()) < 40 and re.search(r'[：:。？！\?!]$', para.strip()) is None

        # 如果是标题且当前块不为空，则完成当前块并开始新块
        if zh_title_enhance and is_title and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        # 如果添加段落后超过块大小，则完成当前块
        elif len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 新块以前一块的结尾开始（重叠）
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + "\n" + para
            else:
                current_chunk = para
        else:
            # 添加段落到当前块
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def read_file_content(file_path, file_type):
    """读取不同类型文件的内容"""
    content = ""
    try:
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif file_type == "pdf":
            try:
                # 使用 PyMuPDF 提取PDF内容
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                for page in doc:
                    content += page.get_text() + "\n\n"
                doc.close()
                log_info("使用PyMuPDF成功提取PDF内容")
            except Exception as e:
                log_error(f"PDF处理失败，确保已安装PyMuPDF: {e}")
                return ""
        elif file_type == "docx":
            try:
                import docx
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n"
            except ImportError:
                log_error("未安装python-docx库，无法处理DOCX文件")
                return ""
        elif file_type == "csv":
            try:
                # 尝试不同的编码方式读取CSV文件
                df = read_csv_with_encoding(file_path)
                if df is not None:
                    content = df.to_string(index=False)
            except Exception as e:
                log_error(f"处理CSV文件时出错: {e}")
                return ""
        else:
            log_error(f"不支持的文件类型: {file_type}")
            return ""

        # 文本清理和规范化
        content = clean_text(content)
        return content
    except Exception as e:
        log_error(f"读取文件内容时出错: {e}")
        return ""

def clean_text(text):
    """清理和规范化文本"""
    if not text:
        return ""

    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 替换多个换行为双换行（保留段落结构）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 移除特殊字符但保留中文标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》、]', '', text)

    return text.strip()

def read_csv_with_encoding(file_path):
    """尝试不同编码读取CSV文件"""
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            log_info(f"成功使用 {encoding} 编码读取CSV文件")
            return df
        except UnicodeDecodeError:
            continue

    # 如果所有编码都失败，尝试检测编码
    try:
        import chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        detected_encoding = result['encoding']
        log_info(f"检测到CSV文件编码: {detected_encoding}")
        return pd.read_csv(file_path, encoding=detected_encoding)
    except ImportError:
        # 如果没有安装chardet，使用latin1作为后备编码
        return pd.read_csv(file_path, encoding='latin1')
    except Exception:
        return None

# 添加构建FAISS索引的函数
def build_faiss_index(kb_name):
    """构建FAISS索引"""
    try:
        log_info(f"开始构建知识库 '{kb_name}' 的FAISS索引")

        # 获取向量文件列表
        vectors_dir = os.path.join("knowledge_bases", kb_name, "vectors")
        if not os.path.exists(vectors_dir):
            log_error(f"向量目录不存在: {vectors_dir}")
            return False

        vector_files = [f for f in os.listdir(vectors_dir) if f.endswith(".json")]
        if not vector_files:
            log_error(f"没有找到向量文件")
            return False

        log_info(f"找到 {len(vector_files)} 个向量文件")

        # 读取所有向量
        all_vectors = []
        all_metadata = []

        for file_name in vector_files:
            file_path = os.path.join(vectors_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    vectors_data = json.load(f)

                for item in vectors_data:
                    if "vector" in item and "text" in item:
                        all_vectors.append(item["vector"])

                        # 复制元数据，但不包括向量
                        metadata = {k: v for k, v in item.items() if k != "vector"}
                        all_metadata.append(metadata)
            except Exception as e:
                log_error(f"读取向量文件 {file_name} 时出错: {e}")

        if not all_vectors:
            log_error("没有找到有效的向量")
            return False

        log_info(f"读取了 {len(all_vectors)} 个向量")

        # 创建FAISS索引
        dimension = len(all_vectors[0])
        index = faiss.IndexFlatL2(dimension)

        # 将向量添加到索引
        vectors_np = np.array(all_vectors).astype('float32')
        index.add(vectors_np)

        log_info(f"创建了包含 {index.ntotal} 个向量的FAISS索引")

        # 确保索引目录存在
        index_dir = os.path.join("knowledge_bases", kb_name, "index")
        os.makedirs(index_dir, exist_ok=True)

        # 保存索引
        index_path = os.path.join(index_dir, "faiss_index")
        faiss.write_index(index, index_path)

        # 保存元数据
        metadata_path = os.path.join(index_dir, "faiss_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        log_info(f"FAISS索引已保存到: {index_path}")
        log_info(f"元数据已保存到: {metadata_path}")

        return True
    except Exception as e:
        log_error(f"创建FAISS索引时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

# 添加构建Pickle索引的函数
def build_pickle_index(kb_name):
    """为知识库构建Pickle索引"""
    vectors_dir = os.path.join("knowledge_bases", kb_name, "vectors")
    index_dir = os.path.join("knowledge_bases", kb_name, "index")
    pickle_path = os.path.join(index_dir, "vectors_index.pkl")

    if not os.path.exists(vectors_dir):
        return False

    # 创建索引目录
    os.makedirs(index_dir, exist_ok=True)

    # 收集所有向量数据
    all_data = []

    # 遍历所有向量文件
    for filename in os.listdir(vectors_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(vectors_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                vectors_data = json.load(f)
                all_data.extend(vectors_data)

    if not all_data:
        return False

    # 保存为pickle文件
    with open(pickle_path, "wb") as f:
        pickle.dump(all_data, f)

    return True

def add_documents_to_vector_store(documents, kb_name):
    """
    将文档添加到向量库，并检测重复内容

    参数:
        documents: 文档列表，每个文档包含text和metadata
        kb_name: 知识库名称

    返回:
        添加的文档数量
    """
    # 1. 创建知识库目录
    kb_dir = os.path.join("knowledge_bases", kb_name)
    index_dir = os.path.join(kb_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    # 2. 检查是否已有向量库
    faiss_index_path = os.path.join(index_dir, "faiss_index")
    metadata_path = os.path.join(index_dir, "faiss_metadata.json")

    # 3. 加载现有向量库和元数据（如果存在）
    existing_vectors = []
    existing_metadata = []
    existing_content_hash = set()  # 用于检测重复内容

    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        try:
            # 加载现有向量库
            faiss_index = faiss.read_index(faiss_index_path)
            dimension = faiss_index.d

            # 提取现有向量
            if faiss_index.ntotal > 0:
                existing_vectors_np = faiss.rev_swig_ptr(faiss_index.get_xb(), faiss_index.ntotal * dimension)
                existing_vectors_np = existing_vectors_np.reshape(faiss_index.ntotal, dimension)
                existing_vectors = existing_vectors_np.tolist()

            # 加载现有元数据
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)

            # 计算现有内容的哈希值
            for doc in existing_metadata:
                text = doc.get('text', '')
                if not text and 'content' in doc:
                    text = doc.get('content', '')
                existing_content_hash.add(hash(text.strip()))

            log_info(f"加载了现有向量库: {len(existing_vectors)} 个向量, {len(existing_metadata)} 条元数据")
        except Exception as e:
            log_error(f"加载现有向量库时出错: {e}")
            # 如果加载失败，创建新的向量库
            existing_vectors = []
            existing_metadata = []
            existing_content_hash = set()

    # 4. 处理新文档
    new_vectors = []
    new_metadata = []
    duplicate_count = 0

    for doc in documents:
        # 获取文档文本和元数据
        text = doc.get('text', '')
        if not text and 'content' in doc:
            text = doc.get('content', '')

        # 检查是否重复
        content_hash = hash(text.strip())
        if content_hash in existing_content_hash:
            duplicate_count += 1
            continue

        # 获取文档向量
        try:
            vector = get_embedding(text)
        except Exception as e:
            log_error(f"获取文档向量时出错: {e}")
            continue

        # 添加到新向量和元数据列表
        new_vectors.append(vector)
        new_metadata.append(doc)
        existing_content_hash.add(content_hash)

    # 5. 创建或更新FAISS索引
    if existing_vectors or new_vectors:
        # 确定向量维度
        dimension = len(new_vectors[0]) if new_vectors else len(existing_vectors[0])

        # 创建新的FAISS索引
        faiss_index = faiss.IndexFlatL2(dimension)

        # 添加所有向量
        all_vectors = existing_vectors + new_vectors
        if all_vectors:
            vectors_np = np.array(all_vectors).astype('float32')
            faiss_index.add(vectors_np)

        # 合并元数据
        all_metadata = existing_metadata + new_metadata

        # 保存更新后的索引和元数据
        faiss.write_index(faiss_index, faiss_index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        log_info(f"向量库更新完成: 添加了 {len(new_vectors)} 个文档，跳过了 {duplicate_count} 个重复文档")
        return len(new_vectors)
    else:
        log_info("没有新文档添加到向量库")
        return 0

def get_embedding(text):
    """获取文本的向量表示"""
    url = "http://localhost:9997/v1/embeddings"

    payload = {
        "model": "bge-large-zh-v1.5",
        "input": text
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        log_info(f"正在请求嵌入向量，文本长度: {len(text)}")
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        request_time = time.time() - start_time
        log_info(f"嵌入向量请求完成，耗时: {request_time:.2f}秒")

        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            log_info(f"成功获取嵌入向量，维度: {len(embedding)}")
            return embedding
        else:
            log_error(f"无法获取嵌入向量: {result}")
            raise Exception("无法获取嵌入向量")

    except Exception as e:
        log_error(f"获取嵌入向量时出错: {e}")
        raise

def initialize_knowledge_base(kb_name):
    """初始化知识库目录结构"""
    kb_dir = os.path.join("knowledge_bases", kb_name)
    index_dir = os.path.join(kb_dir, "index")
    vectors_dir = os.path.join(kb_dir, "vectors")

    # 创建目录
    os.makedirs(kb_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(vectors_dir, exist_ok=True)

    # 初始化元数据文件
    metadata_path = os.path.join(index_dir, "faiss_metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    log_info(f"知识库 '{kb_name}' 初始化完成")
    return True

def filter_knowledge_base(kb_name):
    """过滤知识库中的低质量或重复内容"""
    log_info(f"开始过滤知识库 '{kb_name}' 中的低质量或重复内容")

    # 加载所有向量和元数据
    kb_dir = os.path.join("knowledge_bases", kb_name)
    vectors_dir = os.path.join(kb_dir, "vectors")

    if not os.path.exists(vectors_dir):
        log_error(f"向量目录不存在: {vectors_dir}")
        return False

    # 创建备份目录
    backup_dir = os.path.join(kb_dir, "vectors_backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 备份原始向量文件
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for file in os.listdir(vectors_dir):
        if file.endswith(".json"):
            src = os.path.join(vectors_dir, file)
            dst = os.path.join(backup_dir, f"{backup_time}_{file}")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                log_error(f"备份文件 {file} 时出错: {e}")

    log_info(f"已备份原始向量文件到 {backup_dir}")

    # 加载所有文档
    all_docs = []
    vector_files = []

    for file in os.listdir(vectors_dir):
        if file.endswith(".json"):
            file_path = os.path.join(vectors_dir, file)
            vector_files.append(file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                    all_docs.extend(docs)
            except Exception as e:
                log_error(f"读取向量文件 {file} 时出错: {e}")

    if not all_docs:
        log_error("没有找到有效的文档")
        return False

    log_info(f"加载了 {len(all_docs)} 个文档，来自 {len(vector_files)} 个文件")

    # 检测重复内容
    texts = [doc["text"] for doc in all_docs]
    duplicates = find_duplicates(texts)

    # 过滤掉重复和低质量内容
    filtered_docs = []
    for i, doc in enumerate(all_docs):
        # 跳过重复内容
        if i in duplicates:
            continue

        # 跳过过短的内容
        if len(doc["text"]) < 50:
            continue

        filtered_docs.append(doc)

    log_info(f"过滤后剩余 {len(filtered_docs)} 个文档")

    # 删除原始向量文件
    for file_path in vector_files:
        try:
            os.remove(file_path)
        except Exception as e:
            log_error(f"删除文件 {file_path} 时出错: {e}")

    # 保存过滤后的向量
    # 每个文件最多保存100个文档
    chunk_size = 100
    for i in range(0, len(filtered_docs), chunk_size):
        chunk = filtered_docs[i:i+chunk_size]
        file_name = f"{uuid.uuid4()}.json"
        file_path = os.path.join(vectors_dir, file_name)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            log_info(f"保存了 {len(chunk)} 个文档到 {file_path}")
        except Exception as e:
            log_error(f"保存文件 {file_path} 时出错: {e}")

    # 重建索引
    log_info("开始重建索引...")
    if build_faiss_index(kb_name) and build_pickle_index(kb_name):
        log_info("索引重建成功")
        return True
    else:
        log_error("索引重建失败")
        return False

def find_duplicates(texts, similarity_threshold=0.9):
    """
    查找文本列表中的重复内容

    参数:
        texts: 文本列表
        similarity_threshold: 相似度阈值，超过此值视为重复

    返回:
        重复文本的索引列表
    """
    log_info(f"开始查找重复内容，共 {len(texts)} 个文本块")

    # 如果文本数量太少，直接返回空列表
    if len(texts) < 2:
        return []

    # 使用集合快速查找完全相同的文本
    seen_texts = {}
    exact_duplicates = []

    for i, text in enumerate(texts):
        # 规范化文本（去除空白字符）
        normalized = re.sub(r'\s+', ' ', text).strip()

        # 如果文本太短，跳过
        if len(normalized) < 20:
            continue

        # 检查是否已存在
        if normalized in seen_texts:
            exact_duplicates.append(i)
        else:
            seen_texts[normalized] = i

    log_info(f"找到 {len(exact_duplicates)} 个完全相同的文本块")

    # 如果需要更精确的相似度检测，可以使用向量相似度
    # 这需要计算每个文本的向量，然后比较向量相似度
    # 以下是一个简化版本，仅检测字符重叠

    fuzzy_duplicates = []
    if similarity_threshold < 1.0:
        # 对于剩余的文本，检查内容重叠
        remaining_indices = [i for i in range(len(texts)) if i not in exact_duplicates]
        remaining_texts = [texts[i] for i in remaining_indices]

        for i in range(len(remaining_texts)):
            for j in range(i+1, len(remaining_texts)):
                # 计算文本重叠度
                text1 = remaining_texts[i].lower()
                text2 = remaining_texts[j].lower()

                # 如果两个文本长度相差太大，跳过
                if len(text1) < 0.5 * len(text2) or len(text2) < 0.5 * len(text1):
                    continue

                # 计算Jaccard相似度
                words1 = set(text1.split())
                words2 = set(text2.split())

                if not words1 or not words2:
                    continue

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                similarity = intersection / union if union > 0 else 0

                # 如果相似度超过阈值，标记为重复
                if similarity > similarity_threshold:
                    fuzzy_duplicates.append(remaining_indices[j])

    log_info(f"找到 {len(fuzzy_duplicates)} 个相似文本块")

    # 合并两种重复
    all_duplicates = list(set(exact_duplicates + fuzzy_duplicates))
    log_info(f"总共找到 {len(all_duplicates)} 个重复文本块")

    return all_duplicates

# 添加删除知识库函数
def delete_knowledge_base(kb_name):
    kb_file = "knowledge_bases.json"
    try:
        if os.path.exists(kb_file) and os.path.getsize(kb_file) > 0:
            with open(kb_file, "r", encoding="utf-8") as f:
                knowledge_bases = json.load(f)
        else:
            knowledge_bases = []

        # 从知识库列表中移除该知识库
        knowledge_bases = [kb for kb in knowledge_bases if kb["name"] != kb_name]

        # 保存更新后的知识库列表
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_bases, f, ensure_ascii=False, indent=2)

        # 删除知识库目录
        kb_dir = os.path.join("knowledge_bases", kb_name)
        if os.path.exists(kb_dir):
            shutil.rmtree(kb_dir)

        log_info(f"知识库 '{kb_name}' 已删除")
        return True
    except Exception as e:
        log_error(f"删除知识库 '{kb_name}' 时出错: {e}")
        return False


def documents_page():
    st.title("📄 知识库管理")
    st.write("创建和管理企业知识库，上传和组织文档")

    # 创建必要的目录
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("knowledge_bases"):
        os.makedirs("knowledge_bases")

    # 加载知识库列表
    kb_file = "knowledge_bases.json"
    try:
        if os.path.exists(kb_file) and os.path.getsize(kb_file) > 0:
            with open(kb_file, "r", encoding="utf-8") as f:
                knowledge_bases = json.load(f)
        else:
            # 文件不存在或为空，创建新的空列表
            knowledge_bases = []
            with open(kb_file, "w", encoding="utf-8") as f:
                json.dump(knowledge_bases, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        # JSON解析错误，文件可能损坏
        log_error(f"知识库文件 {kb_file} 格式错误，创建新文件")
        knowledge_bases = []
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_bases, f, ensure_ascii=False, indent=2)

    # 侧边栏配置
    st.sidebar.subheader("⚙️ 知识库设置")

    # # 创建新知识库
    # with st.sidebar.expander("➕ 创建新知识库", expanded=False):
    #     new_kb_name = st.text_input("知识库名称", key="new_kb_name")
    #     new_kb_desc = st.text_area("知识库描述", key="new_kb_desc", height=100)
    #
    #     if st.button("创建知识库", key="create_kb"):
    #         if not new_kb_name:
    #             st.error("请输入知识库名称")
    #         elif any(kb["name"] == new_kb_name for kb in knowledge_bases):
    #             st.error(f"知识库 '{new_kb_name}' 已存在")
    #         else:
    #             # 初始化知识库目录结构
    #             if initialize_knowledge_base(new_kb_name):
    #                 # 添加到知识库列表
    #                 knowledge_bases.append({
    #                     "name": new_kb_name,
    #                     "description": new_kb_desc,
    #                     "created_by": st.session_state['user_name'],
    #                     "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    #                 })
    #
    #                 # 保存知识库列表
    #                 with open(kb_file, "w", encoding="utf-8") as f:
    #                     json.dump(knowledge_bases, f, ensure_ascii=False, indent=2)
    #
    #                 st.success(f"知识库 '{new_kb_name}' 创建成功！")
    #
    #                 # 使用rerun()重新加载页面，而不是直接修改session_state
    #                 time.sleep(1)
    #                 st.rerun()
    #             else:
    #                 st.error(f"创建知识库 '{new_kb_name}' 失败")

    # 创建新知识库
    with st.sidebar.expander("➕ 创建新知识库", expanded=False):
        if st.session_state.get('user_role') == '管理员':
            new_kb_name = st.text_input("知识库名称", key="new_kb_name")
            new_kb_desc = st.text_area("知识库描述", key="new_kb_desc", height=100)

            if st.button("创建知识库", key="create_kb"):
                if not new_kb_name:
                    st.error("请输入知识库名称")
                elif any(kb["name"] == new_kb_name for kb in knowledge_bases):
                    st.error(f"知识库 '{new_kb_name}' 已存在")
                else:
                    # 初始化知识库目录结构
                    if initialize_knowledge_base(new_kb_name):
                        # 添加到知识库列表
                        knowledge_bases.append({
                            "name": new_kb_name,
                            "description": new_kb_desc,
                            "created_by": st.session_state['user_name'],
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # 保存知识库列表
                        with open(kb_file, "w", encoding="utf-8") as f:
                            json.dump(knowledge_bases, f, ensure_ascii=False, indent=2)

                        st.success(f"知识库 '{new_kb_name}' 创建成功！")

                        # 使用rerun()重新加载页面，而不是直接修改session_state
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"创建知识库 '{new_kb_name}' 失败")
        else:
            st.write("只有管理员可以创建知识库。")

    # 选择知识库
    kb_names = [kb["name"] for kb in knowledge_bases]
    if not kb_names:
        st.sidebar.warning("请先创建知识库")
        return

    selected_kb = st.sidebar.selectbox("📚 选择知识库", kb_names)


    # 删除知识库按钮
    if st.sidebar.button(f"🗑️ 删除知识库 '{selected_kb}'"):
        if delete_knowledge_base(selected_kb):
            st.success(f"知识库 '{selected_kb}' 已删除")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"删除知识库 '{selected_kb}' 失败")

    # 获取所有部门列表
    all_departments = st.session_state['user_knowledge_access']

    # 当前用户的部门
    user_department = st.session_state['user_department']

    # 判断用户是否为管理员
    user_role = st.session_state.get('user_role', '')
    is_admin = user_role == '管理员'

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["上传文档", "处理文档", "管理文档"])

    # with tab1:
    #     st.subheader(f"上传文档到 '{selected_kb}' 知识库")
    #
    #     # 获取用户可访问的部门列表
    #     accessible_depts = st.session_state['user_knowledge_access']
    #
    #     # 添加部门选择
    #     selected_department = st.selectbox(
    #         "选择文档所属部门",
    #         accessible_depts,
    #         help="选择文档所属的部门，这将决定哪些用户可以访问该文档"
    #     )
    #
    #     # 上传文件
    #     uploaded_files = st.file_uploader("选择要上传的文件", accept_multiple_files=True)
    #
    #     if uploaded_files:
    #         upload_button = st.button("📤 上传文件", use_container_width=True)
    #
    #         if upload_button:
    #             # 创建目录
    #             kb_dir = os.path.join("knowledge_bases", selected_kb)
    #             docs_dir = os.path.join(kb_dir, "documents")
    #             meta_dir = os.path.join(kb_dir, "metadata")
    #
    #             os.makedirs(docs_dir, exist_ok=True)
    #             os.makedirs(meta_dir, exist_ok=True)
    #
    #             # 检查或创建元数据文件
    #             meta_file = os.path.join(meta_dir, "documents.csv")
    #             if os.path.exists(meta_file):
    #                 df = pd.read_csv(meta_file)
    #             else:
    #                 df = pd.DataFrame(columns=[
    #                     "title", "file_path", "file_type", "department",
    #                     "uploaded_by", "upload_time", "processed"
    #                 ])
    #
    #             # 处理每个上传的文件
    #             for uploaded_file in uploaded_files:
    #                 # 保存文件 - 修复参数问题
    #                 file_name = uploaded_file.name
    #                 file_path = os.path.join(docs_dir, file_name)
    #                 save_uploaded_file(uploaded_file, docs_dir, file_name)
    #
    #                 # 获取文件类型
    #                 file_extension = os.path.splitext(file_name)[1].lower().replace(".", "")
    #
    #                 # 创建元数据
    #                 metadata = {
    #                     "title": file_name,
    #                     "file_path": file_path,
    #                     "file_type": file_extension,
    #                     "department": selected_department,  # 使用选择的部门
    #                     "uploaded_by": st.session_state.get("username", "未知用户"),
    #                     "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    #                     "processed": False
    #                 }
    #
    #                 # 添加到元数据
    #                 df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)
    #
    #                 st.info(f"已上传: {file_name}")
    #
    #             # 保存元数据
    #             df.to_csv(meta_file, index=False)
    #
    #             # 上传成功后，询问是否立即处理向量
    #             if st.success(f"成功上传 {len(uploaded_files)} 个文件"):
    #                 if st.button("立即处理文档向量", key="process_vectors"):
    #                     with st.spinner("正在处理文档向量..."):
    #                         try:
    #                             # 连接到xinference服务
    #                             client = RESTfulClient("http://localhost:9997")
    #
    #                             # 处理最新上传的文件
    #                             for i, row in df.tail(len(uploaded_files)).iterrows():
    #                                 if process_document_vectors(selected_kb, row, client):
    #                                     # 更新处理状态
    #                                     df.at[i, "processed"] = True
    #
    #                             # 保存更新后的元数据
    #                             df.to_csv(meta_file, index=False)
    #
    #                             st.success("文档向量处理完成！")
    #                         except Exception as e:
    #                             st.error(f"处理文档向量时出错: {e}")

    with tab1:
        user_role = st.session_state.get('user_role', '普通用户')
        if user_role != '普通用户':
            st.subheader(f"上传文档到 '{selected_kb}' 知识库")

            # 获取用户可访问的部门列表
            accessible_depts = st.session_state['user_knowledge_access']

            # 添加部门选择
            selected_department = st.selectbox(
                "选择文档所属部门",
                accessible_depts,
                help="选择文档所属的部门，这将决定哪些用户可以访问该文档"
            )

            # 上传文件
            uploaded_files = st.file_uploader("选择要上传的文件", accept_multiple_files=True)

            if uploaded_files:
                upload_button = st.button("📤 上传文件", use_container_width=True)

                # if upload_button:
                #     # 创建目录
                #     kb_dir = os.path.join("knowledge_bases", selected_kb)
                #     docs_dir = os.path.join(kb_dir, "documents")
                #     meta_dir = os.path.join(kb_dir, "metadata")
                #
                #     os.makedirs(docs_dir, exist_ok=True)
                #     os.makedirs(meta_dir, exist_ok=True)
                #
                #     # 检查或创建元数据文件
                #     meta_file = os.path.join(meta_dir, "documents.csv")
                #     if os.path.exists(meta_file):
                #         df = pd.read_csv(meta_file)
                #     else:
                #         df = pd.DataFrame(columns=[
                #             "title", "file_path", "file_type", "department",
                #             "uploaded_by", "upload_time", "processed"
                #         ])
                #
                #     # 处理每个上传的文件
                #     for uploaded_file in uploaded_files:
                #         # 保存文件 - 修复参数问题
                #         file_name = uploaded_file.name
                #         file_path = os.path.join(docs_dir, file_name)
                #         save_uploaded_file(uploaded_file, docs_dir, file_name)
                #
                #         # 获取文件类型
                #         file_extension = os.path.splitext(file_name)[1].lower().replace(".", "")
                #
                #         # 创建元数据
                #         metadata = {
                #             "title": file_name,
                #             "file_path": file_path,
                #             "file_type": file_extension,
                #             "department": selected_department,  # 使用选择的部门
                #             "uploaded_by": st.session_state.get("username", "未知用户"),
                #             "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                #             "processed": False
                #         }
                #
                #         # 添加到元数据
                #         df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)
                #
                #         st.info(f"已上传: {file_name}")
                #
                #     # 保存元数据
                #     df.to_csv(meta_file, index=False)
                #
                #     # 上传成功后，询问是否立即处理向量
                #     if st.success(f"成功上传 {len(uploaded_files)} 个文件"):
                #         if st.button("立即处理文档向量", key="process_vectors"):
                #             with st.spinner("正在处理文档向量..."):
                #                 try:
                #                     # 连接到xinference服务
                #                     client = RESTfulClient("http://localhost:9997")
                #
                #                     # 处理最新上传的文件
                #                     for i, row in df.tail(len(uploaded_files)).iterrows():
                #                         if process_document_vectors(selected_kb, row, client):
                #                             # 更新处理状态
                #                             df.at[i, "processed"] = True
                #
                #                     # 保存更新后的元数据
                #                     df.to_csv(meta_file, index=False)
                #
                #                     st.success("文档向量处理完成！")
                #                 except Exception as e:
                #                     st.error(f"处理文档向量时出错: {e}")
                if upload_button:
                    # 检查管理部权限
                    if selected_department == '管理部' and not is_admin:
                        st.error("只有管理员可以上传管理部的文档")
                        return

                    # 创建目录
                    kb_dir = os.path.join("knowledge_bases", selected_kb)
                    docs_dir = os.path.join(kb_dir, "documents")
                    meta_dir = os.path.join(kb_dir, "metadata")

                    os.makedirs(docs_dir, exist_ok=True)
                    os.makedirs(meta_dir, exist_ok=True)

                    # 检查或创建元数据文件
                    meta_file = os.path.join(meta_dir, "documents.csv")
                    if os.path.exists(meta_file):
                        df = pd.read_csv(meta_file)
                    else:
                        df = pd.DataFrame(columns=[
                            "title", "file_path", "file_type", "department",
                            "uploaded_by", "upload_time", "processed"
                        ])

                    # 处理每个上传的文件
                    for uploaded_file in uploaded_files:
                        # 保存文件 - 修复参数问题
                        file_name = uploaded_file.name
                        file_path = os.path.join(docs_dir, file_name)
                        save_uploaded_file(uploaded_file, docs_dir, file_name)

                        # 获取文件类型
                        file_extension = os.path.splitext(file_name)[1].lower().replace(".", "")

                        # 创建元数据
                        metadata = {
                            "title": file_name,
                            "file_path": file_path,
                            "file_type": file_extension,
                            "department": selected_department,  # 使用选择的部门
                            "uploaded_by": st.session_state.get("username", "未知用户"),
                            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "processed": False
                        }

                        # 添加到元数据
                        df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)

                        st.info(f"已上传: {file_name}")

                    # 保存元数据
                    df.to_csv(meta_file, index=False)

                    # 上传成功后，询问是否立即处理向量
                    if st.success(f"成功上传 {len(uploaded_files)} 个文件"):
                        if st.button("立即处理文档向量", key="process_vectors"):
                            with st.spinner("正在处理文档向量..."):
                                try:
                                    # 连接到xinference服务
                                    client = RESTfulClient("http://localhost:9997")

                                    # 处理最新上传的文件
                                    for i, row in df.tail(len(uploaded_files)).iterrows():
                                        if process_document_vectors(selected_kb, row, client):
                                            # 更新处理状态
                                            df.at[i, "processed"] = True

                                    # 保存更新后的元数据
                                    df.to_csv(meta_file, index=False)

                                    st.success("文档向量处理完成！")
                                except Exception as e:
                                    st.error(f"处理文档向量时出错: {e}")
        else:
            st.write("普通用户没有上传文档的权限。")

    with tab2:
        st.subheader(f"处理 '{selected_kb}' 知识库文档")

        # 检查元数据文件
        meta_file = os.path.join("knowledge_bases", selected_kb, "metadata", "documents.csv")
        if not os.path.exists(meta_file):
            st.warning("没有可处理的文档")
        else:
            df = pd.read_csv(meta_file)
            unprocessed = df[df["processed"] == False]

            # 显示处理状态
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总文档数", len(df))
            with col2:
                st.metric("未处理文档", len(unprocessed))

            # 处理按钮
            if len(unprocessed) > 0:
                process_button = st.button("🔄 处理并建立向量索引", use_container_width=True)

                if process_button:
                    with st.spinner(f"正在处理 {len(unprocessed)} 个文档..."):
                        if not XINFERENCE_AVAILABLE:
                            st.warning("模拟处理模式：xinference模块未安装")

                        try:
                            # 连接到xinference服务
                            client = RESTfulClient("http://localhost:9997")

                            # 检查模型是否已加载
                            if XINFERENCE_AVAILABLE:
                                try:
                                    models = client.list_models()

                                    if isinstance(models, str):
                                        try:
                                            models = json.loads(models)
                                        except:
                                            pass

                                    if isinstance(models, list):
                                        model_exists = False
                                        for model in models:
                                            if isinstance(model, dict) and model.get("model_name") == "bge-large-zh-v1.5":
                                                model_exists = True
                                                break
                                    else:
                                        model_exists = "bge-large-zh-v1.5" in str(models)

                                    if not model_exists:
                                        # 加载模型
                                        result = client.launch_model(
                                            model_name="bge-large-zh-v1.5",
                                            model_type="embedding"
                                        )
                                except Exception as e:
                                    st.error(f"检查模型时出错: {e}")

                            # 处理每个未处理的文档
                            success_count = 0
                            progress_bar = st.progress(0)
                            total_docs = len(unprocessed)

                            for i, (idx, row) in enumerate(unprocessed.iterrows()):
                                progress_bar.progress((i + 1) / total_docs)
                                success = process_document_vectors(selected_kb, row, client)

                                if success:
                                    df.at[idx, "processed"] = True
                                    success_count += 1
                                    st.info(f"已处理: {i+1}/{total_docs} - {row['title']}")
                                else:
                                    st.error(f"处理失败: {row['title']}")

                            df.to_csv(meta_file, index=False)

                            if success_count > 0:
                                with st.spinner("正在优化知识库..."):
                                    # 自动过滤重复内容
                                    filter_knowledge_base(selected_kb)

                                    # 构建索引
                                    if FAISS_AVAILABLE:
                                        try:
                                            if build_faiss_index(selected_kb):
                                                st.success("FAISS索引构建成功！")
                                            else:
                                                st.warning("FAISS索引构建失败")
                                        except Exception as e:
                                            st.error(f"构建FAISS索引时出错: {e}")

                                    try:
                                        if build_pickle_index(selected_kb):
                                            st.success("Pickle索引构建成功！")
                                        else:
                                            st.warning("Pickle索引构建失败")
                                    except Exception as e:
                                        st.error(f"构建Pickle索引时出错: {e}")

                            st.success(f"处理完成！成功: {success_count}, 失败: {len(unprocessed) - success_count}")
                        except Exception as e:
                            if XINFERENCE_AVAILABLE:
                                st.error(f"无法连接到xinference服务: {e}")
                            else:
                                st.error(f"处理文档时出错: {e}")
            else:
                st.success("所有文档已处理完成")

            if len(df) > 0:
                st.subheader("文档处理状态")
                st.dataframe(
                    df[["title", "department", "processed", "upload_time"]],
                    use_container_width=True,
                    column_config={
                        "title": "文件名",
                        "department": "所属部门",
                        "processed": "已处理",
                        "upload_time": "上传时间"
                    }
                )

    # with tab3:
    #     st.subheader(f"'{selected_kb}' 知识库文档管理")
    #
    #     meta_file = os.path.join("knowledge_bases", selected_kb, "metadata", "documents.csv")
    #     if os.path.exists(meta_file):
    #         df = pd.read_csv(meta_file)
    #
    #         accessible_depts = st.session_state['user_knowledge_access']
    #         filtered_df = df[df["department"].isin(accessible_depts)]
    #
    #         if len(filtered_df) > 0:
    #             st.dataframe(
    #                 filtered_df[["title", "department", "uploaded_by", "upload_time", "processed"]],
    #                 use_container_width=True,
    #                 column_config={
    #                     "title": "文件名",
    #                     "department": "所属部门",
    #                     "uploaded_by": "上传者",
    #                     "upload_time": "上传时间",
    #                     "processed": "已处理"
    #                 }
    #             )
    #
    #             selected_doc = st.selectbox("选择文档进行操作", filtered_df["title"].tolist())
    #
    #             if selected_doc:
    #                 doc_data = filtered_df[filtered_df["title"] == selected_doc].iloc[0]
    #
    #                 st.write("### 文档详情")
    #                 st.write(f"**文件名:** {doc_data['title']}")
    #                 st.write(f"**部门:** {doc_data['department']}")
    #                 st.write(f"**上传者:** {doc_data['uploaded_by']}")
    #                 st.write(f"**上传时间:** {doc_data['upload_time']}")
    #                 st.write(f"**文件类型:** {doc_data['file_type']}")
    #                 st.write(f"**处理状态:** {'已处理' if doc_data['processed'] else '未处理'}")
    #
    #                 col1, col2 = st.columns(2)
    #
    #                 with col1:
    #                     if st.button("🗑️ 删除文档", use_container_width=True):
    #                         try:
    #                             os.remove(doc_data["file_path"])
    #                         except:
    #                             st.warning("文件不存在或无法删除")
    #
    #                         df = df[df["title"] != selected_doc]
    #                         df.to_csv(meta_file, index=False)
    #
    #                         st.success(f"文档 '{selected_doc}' 已删除")
    #                         time.sleep(1)
    #                         st.rerun()
    #
    #                 with col2:
    #                     if not doc_data["processed"]:
    #                         if st.button("🔄 处理文档", use_container_width=True):
    #                             try:
    #                                 client = RESTfulClient("http://localhost:9997")
    #                                 success = process_document_vectors(selected_kb, doc_data, client)
    #
    #                                 if success:
    #                                     idx = df[df["title"] == selected_doc].index[0]
    #                                     df.at[idx, "processed"] = True
    #                                     df.to_csv(meta_file, index=False)
    #
    #                                     st.info("正在更新向量索引...")
    #
    #                                     if FAISS_AVAILABLE:
    #                                         try:
    #                                             if build_faiss_index(selected_kb):
    #                                                 st.success("FAISS索引更新成功！")
    #                                             else:
    #                                                 st.warning("FAISS索引更新失败")
    #                                         except Exception as e:
    #                                             st.error(f"更新FAISS索引时出错: {e}")
    #
    #                                     try:
    #                                         if build_pickle_index(selected_kb):
    #                                             st.success("Pickle索引更新成功！")
    #                                         else:
    #                                             st.warning("Pickle索引更新失败")
    #                                     except Exception as e:
    #                                         st.error(f"更新Pickle索引时出错: {e}")
    #
    #                                     st.success(f"文档 '{selected_doc}' 已处理")
    #                                 else:
    #                                     st.error(f"处理文档 '{selected_doc}' 失败")
    #                             except Exception as e:
    #                                 st.error(f"处理文档时出错: {e}")
    #
    #                             time.sleep(1)
    #                             st.rerun()
    #         else:
    #             st.info("没有可访问的文档")
    #     else:
    #         st.info("知识库中还没有文档")

    with tab3:
        st.subheader(f"'{selected_kb}' 知识库文档管理")

        meta_file = os.path.join("knowledge_bases", selected_kb, "metadata", "documents.csv")
        if os.path.exists(meta_file):
            df = pd.read_csv(meta_file)

            accessible_depts = st.session_state['user_knowledge_access']
            filtered_df = df[df["department"].isin(accessible_depts)]

            if len(filtered_df) > 0:
                st.dataframe(
                    filtered_df[["title", "department", "uploaded_by", "upload_time", "processed"]],
                    use_container_width=True,
                    column_config={
                        "title": "文件名",
                        "department": "所属部门",
                        "uploaded_by": "上传者",
                        "upload_time": "上传时间",
                        "processed": "已处理"
                    }
                )

                selected_doc = st.selectbox("选择文档进行操作", filtered_df["title"].tolist())

                if selected_doc:
                    doc_data = filtered_df[filtered_df["title"] == selected_doc].iloc[0]

                    st.write("### 文档详情")
                    st.write(f"**文件名:** {doc_data['title']}")
                    st.write(f"**部门:** {doc_data['department']}")
                    st.write(f"**上传者:** {doc_data['uploaded_by']}")
                    st.write(f"**上传时间:** {doc_data['upload_time']}")
                    st.write(f"**文件类型:** {doc_data['file_type']}")
                    st.write(f"**处理状态:** {'已处理' if doc_data['processed'] else '未处理'}")

                    col1, col2 = st.columns(2)

                    with col1:
                        # 检查管理部权限
                        if doc_data['department'] == '管理部' and not is_admin:
                            st.write("只有管理员可以删除管理部的文档")
                        else:
                            if st.button("🗑️ 删除文档", use_container_width=True):
                                try:
                                    os.remove(doc_data["file_path"])
                                except:
                                    st.warning("文件不存在或无法删除")

                                df = df[df["title"] != selected_doc]
                                df.to_csv(meta_file, index=False)

                                st.success(f"文档 '{selected_doc}' 已删除")
                                time.sleep(1)
                                st.rerun()

                    with col2:
                        if not doc_data["processed"]:
                            if st.button("🔄 处理文档", use_container_width=True):
                                try:
                                    client = RESTfulClient("http://localhost:9997")
                                    success = process_document_vectors(selected_kb, doc_data, client)

                                    if success:
                                        idx = df[df["title"] == selected_doc].index[0]
                                        df.at[idx, "processed"] = True
                                        df.to_csv(meta_file, index=False)

                                        st.info("正在更新向量索引...")

                                        if FAISS_AVAILABLE:
                                            try:
                                                if build_faiss_index(selected_kb):
                                                    st.success("FAISS索引更新成功！")
                                                else:
                                                    st.warning("FAISS索引更新失败")
                                            except Exception as e:
                                                st.error(f"更新FAISS索引时出错: {e}")

                                        try:
                                            if build_pickle_index(selected_kb):
                                                st.success("Pickle索引更新成功！")
                                            else:
                                                st.warning("Pickle索引更新失败")
                                        except Exception as e:
                                            st.error(f"更新Pickle索引时出错: {e}")

                                        st.success(f"文档 '{selected_doc}' 已处理")
                                    else:
                                        st.error(f"处理文档 '{selected_doc}' 失败")
                                except Exception as e:
                                    st.error(f"处理文档时出错: {e}")

                                time.sleep(1)
                                st.rerun()
            else:
                st.info("没有可访问的文档")
        else:
            st.info("知识库中还没有文档")