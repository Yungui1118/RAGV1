import os
import json
import numpy as np
import faiss
import random
import time
import pandas as pd

def test_knowledge_base_retrieval(kb_name="TEST", top_k=3):
    """
    测试知识库的向量检索功能
    
    参数:
    kb_name (str): 知识库名称
    top_k (int): 返回的最相似文档数量
    """
    print(f"开始测试知识库 '{kb_name}' 的向量检索功能")
    
    # 检查知识库是否存在
    kb_dir = os.path.join("knowledge_bases", kb_name)
    if not os.path.exists(kb_dir):
        print(f"错误: 知识库 '{kb_name}' 不存在")
        return False
    
    # 检查向量文件是否存在
    vectors_dir = os.path.join(kb_dir, "vectors")
    if not os.path.exists(vectors_dir):
        print(f"错误: 向量目录不存在 - {vectors_dir}")
        return False
    
    vector_files = [f for f in os.listdir(vectors_dir) if f.endswith(".json")]
    if not vector_files:
        print(f"错误: 没有找到向量文件")
        return False
    
    print(f"找到 {len(vector_files)} 个向量文件:")
    for file in vector_files:
        print(f"  - {file}")
    
    # 检查FAISS索引是否存在
    index_dir = os.path.join(kb_dir, "index")
    faiss_index_path = os.path.join(index_dir, "faiss_index")
    faiss_metadata_path = os.path.join(index_dir, "faiss_metadata.json")
    
    if not os.path.exists(faiss_index_path):
        print(f"错误: FAISS索引不存在 - {faiss_index_path}")
        return False
    
    if not os.path.exists(faiss_metadata_path):
        print(f"错误: FAISS元数据不存在 - {faiss_metadata_path}")
        return False
    
    # 加载FAISS索引和元数据
    try:
        index = faiss.read_index(faiss_index_path)
        print(f"成功加载FAISS索引，包含 {index.ntotal} 个向量")
        
        with open(faiss_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"成功加载元数据，包含 {len(metadata)} 条记录")
    except Exception as e:
        print(f"加载FAISS索引或元数据时出错: {e}")
        return False
    
    # 随机选择一个向量文件作为测试数据
    test_file = os.path.join(vectors_dir, random.choice(vector_files))
    print(f"\n使用文件 {os.path.basename(test_file)} 进行测试")
    
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_vectors = json.load(f)
        
        print(f"文件包含 {len(test_vectors)} 个向量")
        
        # 随机选择一个向量作为查询向量
        query_item = random.choice(test_vectors)
        query_text = query_item["text"]
        query_vector = np.array([query_item["vector"]], dtype=np.float32)
        
        print(f"\n查询文本: '{query_text}'")
        print(f"文档标题: {query_item['document_title']}")
        
        # 执行向量检索
        start_time = time.time()
        distances, indices = index.search(query_vector, top_k + 1)  # +1 是为了排除查询向量本身
        end_time = time.time()
        
        print(f"\n检索耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
        
        # 显示检索结果
        print(f"\n检索结果:")
        results_shown = 0
        
        for i, idx in enumerate(indices[0]):
            if idx >= len(metadata):
                continue
                
            result = metadata[idx]
            
            # 跳过与查询向量完全相同的结果
            if result["text"] == query_text:
                continue
            
            results_shown += 1
            similarity = 1 - distances[0][i]  # 转换为相似度
            
            print(f"\n结果 {results_shown}:")
            print(f"相似度: {similarity:.4f}")
            print(f"文档标题: {result['document_title']}")
            print(f"文本内容: {result['text']}")
            print(f"部门: {result.get('department', '未知')}")
            
            if results_shown >= top_k:
                break
        
        if results_shown == 0:
            print("没有找到相似的文档")
        
        return True
    except Exception as e:
        print(f"执行向量检索时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_document_info(kb_name="TEST"):
    """显示知识库中的文档信息"""
    print(f"\n知识库 '{kb_name}' 的文档信息:")
    
    # 检查元数据文件
    meta_file = os.path.join("knowledge_bases", kb_name, "metadata", "documents.csv")
    if not os.path.exists(meta_file):
        print(f"元数据文件不存在: {meta_file}")
        return
    
    try:
        df = pd.read_csv(meta_file)
        print(f"共有 {len(df)} 个文档:")
        
        for _, row in df.iterrows():
            print(f"  - 标题: {row['title']}")
            print(f"    部门: {row['department']}")
            print(f"    类型: {row['file_type']}")
            print(f"    处理状态: {'已处理' if row['processed'] else '未处理'}")
            print(f"    上传时间: {row['upload_time']}")
            print()
    except Exception as e:
        print(f"读取元数据文件时出错: {e}")

def test_with_specific_file(kb_name="TEST", file_name=None, top_k=3):
    """
    使用指定的向量文件进行测试
    
    参数:
    kb_name (str): 知识库名称
    file_name (str): 向量文件名，如果为None则随机选择
    top_k (int): 返回的最相似文档数量
    """
    vectors_dir = os.path.join("knowledge_bases", kb_name, "vectors")
    
    if file_name is None:
        vector_files = [f for f in os.listdir(vectors_dir) if f.endswith(".json")]
        if not vector_files:
            print("没有找到向量文件")
            return False
        file_name = random.choice(vector_files)
    
    test_file = os.path.join(vectors_dir, file_name)
    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        return False
    
    print(f"使用文件: {file_name}")
    
    # 加载FAISS索引和元数据
    index_dir = os.path.join("knowledge_bases", kb_name, "index")
    faiss_index_path = os.path.join(index_dir, "faiss_index")
    faiss_metadata_path = os.path.join(index_dir, "faiss_metadata.json")
    
    index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 加载向量文件
    with open(test_file, "r", encoding="utf-8") as f:
        test_vectors = json.load(f)
    
    print(f"文件包含 {len(test_vectors)} 个向量")
    
    # 测试每个向量
    for i, query_item in enumerate(test_vectors):
        query_text = query_item["text"]
        query_vector = np.array([query_item["vector"]], dtype=np.float32)
        
        print(f"\n测试向量 {i+1}/{len(test_vectors)}")
        print(f"查询文本: '{query_text}'")
        
        # 执行向量检索
        distances, indices = index.search(query_vector, top_k + 1)
        
        # 显示检索结果
        results_shown = 0
        
        for j, idx in enumerate(indices[0]):
            if idx >= len(metadata):
                continue
                
            result = metadata[idx]
            
            # 跳过与查询向量完全相同的结果
            if result["text"] == query_text:
                continue
            
            results_shown += 1
            similarity = 1 - distances[0][j]
            
            print(f"  结果 {results_shown}: 相似度 {similarity:.4f} - {result['text'][:50]}...")
            
            if results_shown >= top_k:
                break
        
        if results_shown == 0:
            print("  没有找到相似的文档")
        
        # 只测试前3个向量，避免输出过多
        if i >= 2:
            print(f"\n已测试3个向量，剩余 {len(test_vectors) - 3} 个向量未测试")
            break
    
    return True

def test_specific_query(kb_name="TEST", query_text="20-20-20", top_k=5):
    """
    使用特定查询文本进行测试
    
    参数:
    kb_name (str): 知识库名称
    query_text (str): 查询文本
    top_k (int): 返回的最相似文档数量
    """
    print(f"使用特定查询文本进行测试: '{query_text}'")
    
    # 检查FAISS索引是否存在
    index_dir = os.path.join("knowledge_bases", kb_name, "index")
    faiss_index_path = os.path.join(index_dir, "faiss_index")
    faiss_metadata_path = os.path.join(index_dir, "faiss_metadata.json")
    
    if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_metadata_path):
        print(f"错误: FAISS索引或元数据文件不存在")
        return False
    
    # 加载元数据
    try:
        with open(faiss_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"成功加载元数据，包含 {len(metadata)} 条记录")
    except Exception as e:
        print(f"加载元数据时出错: {e}")
        return False
    
    # 直接在元数据中搜索包含查询文本的记录
    print("在元数据中搜索包含查询文本的记录...")
    matching_records = []
    
    for i, record in enumerate(metadata):
        # 使用更宽松的匹配方式
        normalized_text = record["text"].replace(""", "\"").replace(""", "\"")
        if query_text in normalized_text:
            matching_records.append((i, record))
    
    if matching_records:
        print(f"找到 {len(matching_records)} 条包含查询文本的记录")
        
        # 显示找到的记录
        for i, (idx, record) in enumerate(matching_records):
            print(f"\n匹配记录 {i+1}:")
            print(f"索引: {idx}")
            print(f"文档标题: {record['document_title']}")
            print(f"文本内容: {record['text']}")
            print(f"部门: {record.get('department', '未知')}")
            
            # 如果找到了足够多的记录，就停止显示
            if i >= 2:  # 只显示前3条
                print(f"还有 {len(matching_records) - 3} 条匹配记录未显示")
                break
        
        # 使用第一个匹配记录的索引进行相似度搜索
        print("\n使用第一个匹配记录进行相似度搜索...")
        match_idx = matching_records[0][0]
        
        # 加载FAISS索引
        index = faiss.read_index(faiss_index_path)
        
        # 从元数据中获取向量
        vectors_dir = os.path.join("knowledge_bases", kb_name, "vectors")
        vector_files = [f for f in os.listdir(vectors_dir) if f.endswith(".json")]
        
        # 找到包含匹配记录的向量文件
        match_vector = None
        match_text = matching_records[0][1]["text"]
        
        for file_name in vector_files:
            file_path = os.path.join(vectors_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                vectors_data = json.load(f)
                
                for item in vectors_data:
                    if item["text"] == match_text:
                        match_vector = item["vector"]
                        break
                
                if match_vector:
                    break
        
        if match_vector:
            # 使用匹配向量进行检索
            query_vector = np.array([match_vector], dtype=np.float32)
            
            # 执行向量检索
            start_time = time.time()
            distances, indices = index.search(query_vector, top_k + 1)
            end_time = time.time()
            
            print(f"检索耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
            
            # 显示检索结果
            print(f"\n检索结果:")
            results_shown = 0
            
            for i, idx in enumerate(indices[0]):
                if idx >= len(metadata):
                    continue
                    
                result = metadata[idx]
                
                # 跳过与查询向量完全相同的结果
                if result["text"] == match_text:
                    continue
                
                results_shown += 1
                similarity = 1 - distances[0][i]
                
                print(f"\n结果 {results_shown}:")
                print(f"相似度: {similarity:.4f}")
                print(f"文档标题: {result['document_title']}")
                print(f"文本内容: {result['text']}")
                print(f"部门: {result.get('department', '未知')}")
                
                if results_shown >= top_k:
                    break
            
            if results_shown == 0:
                print("没有找到相似的文档")
        else:
            print("无法找到匹配记录的向量")
    else:
        print(f"在元数据中没有找到包含 '{query_text}' 的记录")
        
        # 尝试使用词汇重叠
        print("尝试使用词汇重叠查找相关记录...")
        best_match = None
        best_score = 0
        best_idx = -1
        
        for i, record in enumerate(metadata):
            text = record["text"].lower()
            query_words = set(query_text.lower().split())
            text_words = set(text.split())
            
            if not query_words or not text_words:
                continue
            
            overlap = len(query_words.intersection(text_words))
            score = overlap / len(query_words) if len(query_words) > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = record
                best_idx = i
        
        if best_match and best_score > 0:
            print(f"找到最相关的记录 (相关度: {best_score:.2f}):")
            print(f"文档标题: {best_match['document_title']}")
            print(f"文本内容: {best_match['text']}")
            
            # 使用这个记录进行相似度搜索
            # (后续代码与上面类似)
        else:
            print("无法找到相关记录")
    
    return True

if __name__ == "__main__":
    # 显示知识库中的文档信息
    display_document_info("TEST")
    
    # 测试向量检索
    print("\n" + "=" * 80)
    print("测试向量检索功能")
    print("=" * 80)
    test_knowledge_base_retrieval("TEST", top_k=3)
    
    # 测试特定文件中的向量
    print("\n" + "=" * 80)
    print("测试特定文件中的向量")
    print("=" * 80)
    test_with_specific_file("TEST", file_name="7db0955e-1cf5-40cc-86cb-ff967d382053.json", top_k=3)
    
    # 测试特定查询
    print("\n" + "=" * 80)
    print("测试特定查询")
    print("=" * 80)
    test_specific_query("TEST", query_text="早晨喝温开水有", top_k=5)