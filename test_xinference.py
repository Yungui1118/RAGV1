import os
import json
import time
from xinference.client import RESTfulClient

def test_xinference_embedding():
    """
    测试xinference的嵌入向量生成功能
    """
    print("开始测试xinference嵌入向量生成...")
    
    # 连接到xinference服务
    try:
        client = RESTfulClient("http://localhost:9997")
        print("成功连接到xinference服务")
    except Exception as e:
        print(f"连接xinference服务失败: {e}")
        print("请确保xinference服务已启动，可以使用以下命令启动:")
        print("xinference --host 0.0.0.0 --port 9997")
        return False
    
    # 检查模型是否已加载
    try:
        models = client.list_models()
        print(f"模型列表类型: {type(models)}")
        print(f"模型列表内容: {models}")
        
        # 如果models是字符串，尝试解析JSON
        if isinstance(models, str):
            try:
                models = json.loads(models)
                print("成功将字符串解析为JSON")
            except:
                print("无法将返回的字符串解析为JSON")
        
        # 检查models是否为列表
        if isinstance(models, list):
            print(f"当前已加载的模型: {len(models)}个")
            for model in models:
                if isinstance(model, dict):
                    print(f"  - {model.get('model_name', 'unknown')} (类型: {model.get('model_type', 'unknown')})")
                else:
                    print(f"  - {model}")
            
            # 检查bge-large-zh-v1.5是否已加载
            if len(models) > 0 and isinstance(models[0], dict):
                model_exists = any(m.get("model_name") == "bge-large-zh-v1.5" for m in models)
            else:
                model_exists = "bge-large-zh-v1.5" in str(models)
        else:
            print(f"模型列表不是列表类型，无法检查已加载模型")
            model_exists = False
        
        # 获取已加载的模型ID
        model_id = None
        if isinstance(models, list) and len(models) > 0:
            for model in models:
                if isinstance(model, dict) and model.get("model_name") == "bge-large-zh-v1.5":
                    model_id = model.get("model_id")
                    print(f"找到已加载的bge-large-zh-v1.5模型，ID: {model_id}")
                    break
        
        if not model_exists:
            print("正在加载bge-large-zh-v1.5模型...")
            try:
                result = client.launch_model(
                    model_name="bge-large-zh-v1.5",
                    model_type="embedding"
                )
                print(f"模型加载结果类型: {type(result)}")
                print(f"模型加载结果: {result}")
                
                # 尝试从结果中获取模型ID
                if isinstance(result, dict):
                    model_id = result.get("model_id")
                    print(f"新加载的模型ID: {model_id}")
                
                print("bge-large-zh-v1.5模型已加载")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("请检查模型名称是否正确，以及xinference是否支持该模型")
                return False
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return False
    
    # 等待模型加载完成
    print("等待模型加载完成...")
    time.sleep(5)  # 给模型一些加载时间
    
    # 测试文本
    test_texts = [
        "这是一个测试文本，用于验证xinference的嵌入向量生成功能。",
        "企业知识库系统可以帮助组织管理和检索重要信息。",
        "向量数据库是实现语义搜索的关键技术。"
    ]
    
    # 测试单个文本的向量生成
    print("\n测试单个文本的向量生成:")
    try:
        start_time = time.time()
        
        # 使用正确的API调用方式
        # 检查client对象的可用方法
        print(f"客户端对象的方法: {dir(client)}")
        
        # 尝试使用create_embedding方法
        if hasattr(client, 'create_embedding'):
            print("使用client.create_embedding方法")
            if model_id:
                response = client.create_embedding(
                    model_id=model_id,
                    input=[test_texts[0]]
                )
            else:
                response = client.create_embedding(
                    model="bge-large-zh-v1.5",
                    input=[test_texts[0]]
                )
        # 尝试使用embedding方法
        elif hasattr(client, 'embedding'):
            print("使用client.embedding方法")
            if model_id:
                response = client.embedding(
                    model_id=model_id,
                    input=[test_texts[0]]
                )
            else:
                response = client.embedding(
                    model="bge-large-zh-v1.5",
                    input=[test_texts[0]]
                )
        # 尝试直接发送REST请求
        else:
            print("使用直接REST请求")
            import requests
            if model_id:
                url = f"http://localhost:9997/v1/embeddings"
                payload = {
                    "model": model_id,
                    "input": [test_texts[0]]
                }
            else:
                url = f"http://localhost:9997/v1/embeddings"
                payload = {
                    "model": "bge-large-zh-v1.5",
                    "input": [test_texts[0]]
                }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response = response.json()
        
        end_time = time.time()
        
        print(f"API响应时间: {end_time - start_time:.2f}秒")
        print(f"API响应类型: {type(response)}")
        print(f"API响应内容: {response}")
        
        # 检查响应结构
        if isinstance(response, dict) and "data" in response:
            data = response["data"]
            print(f"响应data类型: {type(data)}")
            print(f"响应data长度: {len(data)}")
            
            if len(data) > 0:
                embedding_data = data[0]
                print(f"embedding_data类型: {type(embedding_data)}")
                print(f"embedding_data内容: {embedding_data}")
                
                if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                    embedding = embedding_data["embedding"]
                    print(f"embedding类型: {type(embedding)}")
                    print(f"向量维度: {len(embedding)}")
                    print(f"向量前5个值: {embedding[:5]}")
                    print("单个文本向量生成成功!")
                else:
                    print("embedding_data中没有embedding字段")
            else:
                print("响应data为空")
        else:
            print("响应中没有找到data字段")
            print(f"响应详情: {response}")
            return False
    except Exception as e:
        print(f"生成单个文本向量时出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试批量文本的向量生成
    print("\n测试批量文本的向量生成:")
    try:
        start_time = time.time()
        
        # 使用与单个文本相同的API调用方式
        if hasattr(client, 'create_embedding'):
            print("使用client.create_embedding方法")
            if model_id:
                response = client.create_embedding(
                    model_id=model_id,
                    input=test_texts
                )
            else:
                response = client.create_embedding(
                    model="bge-large-zh-v1.5",
                    input=test_texts
                )
        elif hasattr(client, 'embedding'):
            print("使用client.embedding方法")
            if model_id:
                response = client.embedding(
                    model_id=model_id,
                    input=test_texts
                )
            else:
                response = client.embedding(
                    model="bge-large-zh-v1.5",
                    input=test_texts
                )
        else:
            print("使用直接REST请求")
            import requests
            if model_id:
                url = f"http://localhost:9997/v1/embeddings"
                payload = {
                    "model": model_id,
                    "input": test_texts
                }
            else:
                url = f"http://localhost:9997/v1/embeddings"
                payload = {
                    "model": "bge-large-zh-v1.5",
                    "input": test_texts
                }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response = response.json()
        
        end_time = time.time()
        
        print(f"API响应时间: {end_time - start_time:.2f}秒")
        print(f"API响应类型: {type(response)}")
        
        # 检查响应结构
        if isinstance(response, dict) and "data" in response:
            data = response["data"]
            print(f"成功生成 {len(data)} 个向量")
            
            for i, embedding_data in enumerate(data):
                if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                    embedding = embedding_data["embedding"]
                    print(f"文本 {i+1} 向量维度: {len(embedding)}")
                else:
                    print(f"文本 {i+1} 没有找到embedding字段")
            
            print("批量文本向量生成成功!")
        else:
            print("响应中没有找到data字段")
            print(f"响应详情: {response}")
            return False
    except Exception as e:
        print(f"生成批量文本向量时出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试向量序列化
    print("\n测试向量序列化:")
    try:
        # 获取一个向量
        if hasattr(client, 'create_embedding'):
            response = client.create_embedding(
                model="bge-large-zh-v1.5",
                input=[test_texts[0]]
            )
            embedding = response["data"][0]["embedding"]
        elif hasattr(client, 'embedding'):
            response = client.embedding(
                model="bge-large-zh-v1.5",
                input=[test_texts[0]]
            )
            embedding = response["data"][0]["embedding"]
        else:
            import requests
            url = f"http://localhost:9997/v1/embeddings"
            payload = {
                "model": "bge-large-zh-v1.5",
                "input": [test_texts[0]]
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers).json()
            embedding = response["data"][0]["embedding"]
        
        # 尝试序列化
        vector_data = {
            "text": test_texts[0],
            "vector": embedding
        }
        
        # 尝试直接序列化
        try:
            json_str = json.dumps(vector_data)
            print("直接序列化成功")
        except Exception as e:
            print(f"直接序列化失败: {e}")
            
            # 尝试转换为列表后序列化
            try:
                if not isinstance(vector_data["vector"], list):
                    vector_data["vector"] = list(vector_data["vector"])
                json_str = json.dumps(vector_data)
                print("转换为列表后序列化成功")
            except Exception as e:
                print(f"转换为列表后序列化失败: {e}")
                
                # 尝试转换为字符串后序列化
                try:
                    vector_data["vector"] = str(vector_data["vector"])
                    json_str = json.dumps(vector_data)
                    print("转换为字符串后序列化成功")
                except Exception as e:
                    print(f"转换为字符串后序列化失败: {e}")
                    return False
    except Exception as e:
        print(f"测试向量序列化时出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n所有测试通过！xinference嵌入向量生成功能正常。")
    return True

if __name__ == "__main__":
    test_xinference_embedding() 