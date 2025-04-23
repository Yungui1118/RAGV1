import os
import sys
import json
import faiss
import numpy as np
import pickle
import requests
from typing import List, Dict, Any

# 模型服务配置
MODEL_API_URL = "http://localhost:9997"  # 使用9997端口

def test_vector_store_creation(kb_name: str, test_docs: List[Dict[str, Any]]):
    """
    测试向量库的创建
    
    参数:
        kb_name: 知识库名称
        test_docs: 测试文档列表，每个文档是一个字典，包含text和metadata
    """
    print(f"=== 测试向量库创建: {kb_name} ===")
    
    # 1. 创建知识库目录
    kb_dir = os.path.join("knowledge_bases", kb_name)
    indexes_dir = os.path.join(kb_dir, "indexes")
    os.makedirs(indexes_dir, exist_ok=True)
    
    # 2. 为测试文档生成向量
    docs_vectors = []
    docs_metadata = []
    
    for doc in test_docs:
        # 获取文档向量
        vector = get_embedding(doc["text"])
        docs_vectors.append(vector)
        
        # 保存元数据
        metadata = doc.get("metadata", {})
        metadata["text"] = doc["text"]
        docs_metadata.append(metadata)
    
    # 3. 创建FAISS索引
    dimension = len(docs_vectors[0])  # 向量维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离的平面索引
    
    # 将向量添加到索引
    vectors_np = np.array(docs_vectors).astype('float32')
    index.add(vectors_np)
    
    # 4. 保存FAISS索引
    faiss_path = os.path.join(indexes_dir, "faiss")
    faiss.write_index(index, faiss_path)
    print(f"FAISS索引已保存到: {faiss_path}")
    
    # 5. 保存向量元数据
    metadata_path = os.path.join(indexes_dir, "vector_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(docs_metadata, f)
    print(f"向量元数据已保存到: {metadata_path}")
    
    return True

def get_embedding(text: str) -> List[float]:
    """获取文本的向量表示"""
    url = f"{MODEL_API_URL}/v1/embeddings"
    
    payload = {
        "model": "bge-large-zh-v1.5",  # 使用BGE模型
        "input": text
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "data" in result and len(result["data"]) > 0:
            return result["data"][0]["embedding"]
        else:
            raise Exception(f"无法获取文本向量: {result}")
    
    except Exception as e:
        print(f"获取向量时出错: {e}")
        # 如果API不可用，返回随机向量用于测试
        print("使用随机向量进行测试...")
        return np.random.rand(768).tolist()  # BGE模型通常是768维

def test_rag_query(kb_name: str, query: str):
    """
    测试RAG查询
    
    参数:
        kb_name: 知识库名称
        query: 查询文本
    """
    print(f"\n=== 测试RAG查询: '{query}' ===")
    
    # 1. 获取查询向量
    query_vector = get_embedding(query)
    
    # 2. 加载FAISS索引
    index_path = os.path.join("knowledge_bases", kb_name, "indexes", "faiss")
    if not os.path.exists(index_path):
        print(f"错误: FAISS索引不存在: {index_path}")
        return False
    
    faiss_index = faiss.read_index(index_path)
    
    # 3. 加载向量元数据
    metadata_path = os.path.join("knowledge_bases", kb_name, "indexes", "vector_metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"错误: 向量元数据不存在: {metadata_path}")
        return False
    
    with open(metadata_path, "rb") as f:
        vector_metadata = pickle.load(f)
    
    # 4. 执行向量搜索
    top_k = 3
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), top_k)
    
    # 5. 处理搜索结果
    results = []
    for i in range(len(I[0])):
        idx = I[0][i]
        score = float(D[0][i])
        
        if idx < len(vector_metadata):
            doc_data = vector_metadata[idx]
            doc_data["score"] = score
            results.append(doc_data)
            print(f"匹配文档 {i+1} (相似度: {score:.4f}):")
            print(f"  内容: {doc_data['text']}")
            if "source" in doc_data:
                print(f"  来源: {doc_data['source']}")
            print()
    
    # 6. 生成回答
    if results:
        answer = generate_answer(query, results)
        print(f"生成的回答: {answer}")
        return True
    else:
        print("未找到相关文档")
        return False

def generate_answer(query: str, docs: List[Dict]):
    """使用检索到的文档生成回答"""
    url = f"{MODEL_API_URL}/v1/chat/completions"
    
    # 构建提示词
    prompt = "请根据以下参考资料回答问题。如果参考资料中没有相关信息，请说明无法从参考资料中找到答案。\n\n"
    
    # 添加参考文档
    prompt += "参考资料：\n"
    for i, doc in enumerate(docs):
        prompt += f"[{i+1}] {doc.get('text', '')}\n"
    
    prompt += f"\n问题：{query}\n\n回答："
    
    payload = {
        "model": "qwen2.5-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "模型没有返回有效回答。"
    
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return f"生成回答失败: {e}"

if __name__ == "__main__":
    # 测试知识库名称
    test_kb_name = "test_kb"
    
    # 测试文档
    test_documents = [
        {
            "text": "人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "metadata": {"source": "AI简介.txt", "page": 1}
        },
        {
            "text": "机器学习是人工智能的一个子领域，它使用统计方法让计算机系统能够从数据中学习。",
            "metadata": {"source": "机器学习基础.pdf", "page": 5}
        },
        {
            "text": "深度学习是机器学习的一种方法，它使用多层神经网络来处理复杂的模式识别任务。",
            "metadata": {"source": "深度学习入门.docx", "page": 12}
        },
        {
            "text": "自然语言处理(NLP)是AI的一个分支，专注于让计算机理解和生成人类语言。",
            "metadata": {"source": "NLP技术.pdf", "page": 3}
        },
        {
            "text": "计算机视觉是AI的一个领域，它使计算机能够从图像或视频中获取信息并理解视觉世界。",
            "metadata": {"source": "计算机视觉.txt", "page": 7}
        }
    ]
    
    # 1. 创建并构建向量库
    if test_vector_store_creation(test_kb_name, test_documents):
        # 2. 测试RAG查询
        test_rag_query(test_kb_name, "什么是人工智能？")
        test_rag_query(test_kb_name, "机器学习和深度学习有什么关系？")
        test_rag_query(test_kb_name, "自然语言处理的应用有哪些？") 