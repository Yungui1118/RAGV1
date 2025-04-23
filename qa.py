import streamlit as st
import os
import json
import time
import pandas as pd
import requests
import numpy as np
import faiss
from datetime import datetime
import re
import jieba

# 修改为正确的模型服务配置
MODEL_API_URL = "http://localhost:9997"  # 修改为9997端口

# 添加日志函数
def log_info(message):
    """打印带时间戳的日志信息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] INFO: {message}")

def log_error(message):
    """打印带时间戳的错误信息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] ERROR: {message}")

def qa_page():
    st.title("智能问答系统")
    
    # 检查用户是否已登录
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("请先登录")
        return
    
    # 获取用户可访问的知识库列表
    accessible_kbs = []
    kb_root = "knowledge_bases"
    if os.path.exists(kb_root):
        for kb_name in os.listdir(kb_root):
            kb_path = os.path.join(kb_root, kb_name)
            if os.path.isdir(kb_path):
                accessible_kbs.append(kb_name)
    
    log_info(f"可访问的知识库: {accessible_kbs}")
    
    # 侧边栏配置
    with st.sidebar:
        st.subheader("对话设置")
        
        # 知识库选择
        selected_kb = st.selectbox(
            "选择知识库",
            ["无 (直接对话)"] + accessible_kbs,
            help="选择要查询的知识库，或选择'无'直接与模型对话"
        )
        
        log_info(f"选择的知识库: {selected_kb}")
        
        # 固定使用qwen2.5-instruct模型
        selected_model = "qwen2.5-instruct"
        st.info(f"当前使用模型: {selected_model}")
        
        # 温度参数
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                              help="较低的值会使输出更加确定和基于事实，较高的值会使其更加创造性")
        
        # RAG参数（仅在选择知识库时显示）
        if selected_kb != "无 (直接对话)":
            st.subheader("RAG 参数")
            top_k = st.slider("召回文档数", min_value=1, max_value=10, value=3, step=1,
                             help="从知识库中检索的相关文档数量")
            score_threshold = st.slider("相似度距离阈值", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                                      help="较小的值表示更严格的匹配要求")
        
        # 清空对话按钮
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 添加一个会话状态变量来跟踪当前的引用文档
    if "current_references" not in st.session_state:
        st.session_state.current_references = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 不在历史消息中显示引用文档
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        log_info(f"用户问题: {prompt}")
        
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示助手消息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_container = st.empty()
            
            try:
                # 显示静态等待消息
                status_message = status_container.info("正在生成回答，请稍候...")
                
                if selected_kb != "无 (直接对话)":
                    # 使用RAG流程
                    with st.spinner("正在查询知识库..."):
                        try:
                            log_info(f"开始RAG流程，知识库: {selected_kb}")
                            total_start_time = time.time()
                            
                            # 1. 先进行向量检索
                            retrieval_start_time = time.time()
                            log_info("开始检索相关文档...")
                            docs = retrieve_relevant_documents(
                                query=prompt,
                                kb_name=selected_kb,
                                top_k=top_k,
                                score_threshold=score_threshold
                            )
                            retrieval_time = time.time() - retrieval_start_time
                            log_info(f"文档检索完成，耗时: {retrieval_time:.2f}秒，找到 {len(docs)} 个相关文档")
                            
                            # 2. 如果找到相关文档，则调用大模型生成回答；否则直接返回友善回复
                            if docs:
                                log_info("开始使用两阶段流式生成方法...")
                                generation_start_time = time.time()
                                
                                # 在开始流式生成之前初始化answer_container
                                answer_container = ""
                                actual_answer_started = False
                                
                                # 使用两阶段流式生成
                                for text_chunk in two_stage_streaming_generation(query=prompt, docs=docs, model=selected_model, temperature=temperature):
                                    # 检测特殊标记
                                    if text_chunk == "__START_ANSWER__":
                                        actual_answer_started = True
                                        status_container.empty()  # 清除状态消息
                                        continue
                                        
                                    # 处理状态消息
                                    if not actual_answer_started and text_chunk.startswith("正在") and "..." in text_chunk:
                                        status_container.info(text_chunk)
                                        continue
                                        
                                    # 处理实际回答
                                    if actual_answer_started:
                                        answer_container += text_chunk
                                        message_placeholder.markdown(answer_container + "▌")
                                
                                # 检查是否有未授权部门的提醒
                                if hasattr(st.session_state, 'unauthorized_departments') and st.session_state.unauthorized_departments:
                                    unauthorized_depts = list(st.session_state.unauthorized_departments)
                                    reminder = f"\n\n---\n\n**温馨提醒**：您的查询可能涉及 **{', '.join(unauthorized_depts)}** 部门的知识，但您没有访问权限。如需了解更多信息，请联系管理员申请相关权限。"
                                    answer_container += reminder
                                    # 清除会话中的未授权部门记录，避免影响下一次查询
                                    st.session_state.unauthorized_departments = set()
                                
                                # 最终显示完整回答
                                message_placeholder.markdown(answer_container)
                                
                                generation_time = time.time() - generation_start_time
                                log_info(f"流式生成完成，耗时: {generation_time:.2f}秒")
                                
                                # 显示引用文档
                                if docs:
                                    with st.expander("查看引用文档"):
                                        for i, doc in enumerate(docs):
                                            st.markdown(f"**文档 {i+1}** (相似度: {doc.get('score', 'N/A'):.2f})")
                                            st.markdown(f"**内容:** {doc.get('text', 'N/A')}")
                                            st.markdown(f"**来源:** {doc.get('document_title', 'N/A')}")
                                            st.markdown(f"**部门:** {doc.get('department', 'N/A')}")
                                            st.markdown("---")
                                
                                # 保存到聊天历史
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer_container,
                                    "docs": docs  # 保存引用文档
                                })
                                
                                total_time = time.time() - total_start_time
                                log_info(f"RAG流程完成，总耗时: {total_time:.2f}秒")
                                
                                # 保存当前的引用文档
                                st.session_state.current_references = docs
                                
                                # 完成后清除状态消息
                                status_container.empty()
                                
                            else:
                                log_info("没有找到相关文档，返回友好提示")
                                status_container.empty()  # 清除状态消息
                                
                                # 没有找到相关文档，检查是否有未授权部门
                                if hasattr(st.session_state, 'unauthorized_departments') and st.session_state.unauthorized_departments:
                                    # 有未授权部门，说明用户可能在查询没有权限的内容
                                    unauthorized_depts = list(st.session_state.unauthorized_departments)
                                    no_access_message = f"抱歉，您的查询可能涉及 **{', '.join(unauthorized_depts)}** 部门的知识，但您没有访问权限。如需了解更多信息，请联系管理员申请相关权限。"
                                    message_placeholder.warning(no_access_message)
                                    
                                    # 保存到聊天历史
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": no_access_message,
                                        "docs": []
                                    })
                                    
                                    # 清除会话中的未授权部门记录
                                    st.session_state.unauthorized_departments = set()
                                else:
                                    # 没有未授权部门，说明知识库中确实没有相关内容
                                    no_docs_message = "抱歉，我在知识库中没有找到与您问题相关的信息。您可以尝试：\n\n1. 换一种方式提问\n2. 查询其他相关主题\n3. 联系管理员添加相关知识"
                                    message_placeholder.info(no_docs_message)
                                    
                                    # 保存到聊天历史
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": no_docs_message,
                                        "docs": []
                                    })
                        except Exception as e:
                            status_container.empty()
                            error_msg = f"查询知识库时出错: {str(e)}"
                            log_error(error_msg)
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
                else:
                    # 直接与模型对话
                    with st.spinner("正在生成回答..."):
                        try:
                            log_info("开始直接对话模式")
                            chat_start_time = time.time()
                            
                            # 初始化answer_container
                            answer_container = ""
                            actual_answer_started = False
                            
                            # 使用流式直接对话
                            for text_chunk in direct_chat_streaming(
                                messages=st.session_state.messages,
                                model=selected_model,
                                temperature=temperature
                            ):
                                # 检测特殊标记
                                if text_chunk == "__START_ANSWER__":
                                    actual_answer_started = True
                                    status_container.empty()  # 清除状态消息
                                    continue
                                
                                # 处理实际回答
                                if actual_answer_started:
                                    answer_container += text_chunk
                                    message_placeholder.markdown(answer_container + "▌")
                            
                            # 检查是否有未授权部门的提醒
                            if hasattr(st.session_state, 'unauthorized_departments') and st.session_state.unauthorized_departments:
                                unauthorized_depts = list(st.session_state.unauthorized_departments)
                                reminder = f"\n\n---\n\n**温馨提醒**：您的查询可能涉及 **{', '.join(unauthorized_depts)}** 部门的知识，但您没有访问权限。如需了解更多信息，请联系管理员申请相关权限。"
                                answer_container += reminder
                                # 清除会话中的未授权部门记录，避免影响下一次查询
                                st.session_state.unauthorized_departments = set()
                            
                            # 最终显示完整回答
                            message_placeholder.markdown(answer_container)
                            
                            chat_time = time.time() - chat_start_time
                            log_info(f"流式直接对话完成，耗时: {chat_time:.2f}秒")
                            
                            # 保存到聊天历史
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer_container,
                                "docs": []  # 直接对话模式下，docs为空
                            })
                            
                            # 完成后清除状态消息
                            status_container.empty()
                            
                        except Exception as e:
                            status_container.empty()
                            error_msg = f"生成回答时出错: {str(e)}"
                            log_error(error_msg)
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
            
            except Exception as e:
                status_container.empty()
                message_placeholder.error(f"发生错误: {str(e)}")
                log_error(f"处理用户问题时发生错误: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"发生错误: {str(e)}"
                })

            finally:
                # 确保清除状态消息
                status_container.empty()

def retrieve_relevant_documents(query, kb_name, top_k=3, score_threshold=0.7):
    """检索相关文档"""
    try:
        log_info("正在获取查询向量...")
        query_vector = get_embedding(query)
        log_info(f"成功获取嵌入向量，维度: {len(query_vector)}")
        
        kb_dir = os.path.join("knowledge_bases", kb_name)
        index_dir = os.path.join(kb_dir, "index")
        faiss_index_path = os.path.join(index_dir, "faiss_index")
        metadata_path = os.path.join(index_dir, "faiss_metadata.json")
        
        if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
            log_error(f"知识库索引文件不存在: {faiss_index_path}")
            return []
        
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        log_info(f"加载了 {faiss_index.ntotal} 个向量和 {len(metadata)} 条元数据")
        
        D, I = faiss_index.search(np.array([query_vector]).astype('float32'), top_k * 2)
        log_info(f"向量搜索结果: 距离={D[0]}, 索引={I[0]}")
        
        user_departments = st.session_state.get('user_knowledge_access', [])
        log_info(f"用户 {st.session_state.get('user_name', '未知用户')} 可访问部门: {user_departments}")
        
        results = []
        seen_content_hash = set()
        unauthorized_departments = set()  # 记录用户没有权限的部门
        
        for i in range(len(I[0])):
            idx = I[0][i]
            score = float(D[0][i])
            
            if idx < 0 or idx >= len(metadata):
                continue
                
            # 修正相似度判断逻辑：FAISS中分数越小表示越相似
            if score > score_threshold:
                log_info(f"文档 {idx} 相似度分数 {score} 大于阈值 {score_threshold}，跳过")
                continue
                
            doc_data = metadata[idx]
            
            # 检查部门权限
            doc_department = doc_data.get("department", "未知部门")
            if doc_department and doc_department not in user_departments and "全部" not in user_departments:
                log_info(f"用户无权访问部门 '{doc_department}' 的文档 {idx}")
                unauthorized_departments.add(doc_department)  # 记录未授权的部门
                continue
                
            # 检查内容重复
            content = doc_data.get("text", "")
            content_hash = hash(content)
            if content_hash in seen_content_hash:
                log_info(f"文档 {idx} 内容重复，跳过")
                continue
                
            seen_content_hash.add(content_hash)
            doc_data["score"] = score
            results.append(doc_data)
            
            if len(results) >= top_k:
                break
        
        # 如果有未授权的部门，记录到会话状态中
        if unauthorized_departments:
            if "unauthorized_departments" not in st.session_state:
                st.session_state.unauthorized_departments = set()
            st.session_state.unauthorized_departments.update(unauthorized_departments)
            log_info(f"用户查询涉及未授权部门: {unauthorized_departments}")
        
        return results
        
    except Exception as e:
        log_error(f"检索相关文档时出错: {e}")
        raise

def get_embedding(text):
    """获取文本的向量表示"""
    url = f"{MODEL_API_URL}/v1/embeddings"
    payload = {
        "model": "bge-large-zh-v1.5",
        "input": text
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        log_info(f"正在请求嵌入向量，文本长度: {len(text)}")
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        request_time = time.time() - start_time
        log_info(f"嵌入向量请求完成，耗时: {request_time:.2f}秒")
        
        if "data" in result and len(result["data"]) > 0:
            embedding = result["data"][0]["embedding"]
            log_info(f"成功获取嵌入向量，维度: {len(embedding)}")
            return embedding
        else:
            log_error(f"无法获取文本向量: {result}")
            raise Exception("无法获取文本向量")
    
    except Exception as e:
        log_error(f"获取向量时出错: {e}")
        raise

def generate_answer(query, docs, model="qwen2.5-instruct", temperature=0.7):
    """根据检索到的文档生成回答"""
    if not docs:
        log_info("没有找到相关文档，返回友好提示")
        return "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试换一种方式提问，或者咨询其他您有权限访问的内容。"
    
    prompt = """你是一个严格的知识库助手，必须遵循以下规则：
1. 你的回答必须完全基于提供的参考资料
2. 在回答中使用[文档X]格式标注信息来源
3. 如果参考资料中没有足够信息回答问题，请明确说明
4. 不要编造或添加参考资料中没有的信息

参考资料:
"""
    for i, doc in enumerate(docs):
        content = doc.get("text", "")
        source = doc.get("document_title", "未知来源")
        prompt += f"\n[文档{i+1}] {content}\n来源: {source}\n"
    
    prompt += f"\n用户问题: {query}\n\n请严格按照上述规则，基于参考资料回答问题："
    
    log_info(f"生成提示词，长度: {len(prompt)}")
    
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的知识库助手，只能使用提供的参考资料回答问题，不得添加任何参考资料中没有的信息。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # 降低温度，使回答更加确定性
        "stream": False
    }
    
    try:
        log_info("正在生成最终回答...")
        # 使用带重试的API请求
        result = make_api_request_with_retry(
            url=url,
            payload=payload,
            max_retries=3,
            timeout=90
        )
        
        if 'choices' in result and len(result['choices']) > 0:
            final_answer = result['choices'][0]['message']['content']
            log_info(f"成功生成回答，长度: {len(final_answer)}")
            return final_answer
        else:
            log_error("生成回答失败")
            return "抱歉，生成回答时出现错误。请稍后再试。"
    
    except Exception as e:
        log_error(f"两阶段生成过程出错: {e}")
        return f"抱歉，生成回答时出现错误: {e}"

def call_direct_chat(messages, model="qwen2.5-instruct", temperature=0.7):
    """直接与大模型对话，不使用知识库"""
    url = f"{MODEL_API_URL}/v1/chat/completions"
    formatted_messages = []
    for msg in messages:
        if msg["role"] in ["user", "assistant", "system"]:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    log_info(f"直接对话消息数量: {len(formatted_messages)}")
    
    payload = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        log_info(f"正在请求模型直接对话")
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        request_time = time.time() - start_time
        log_info(f"模型直接对话请求完成，耗时: {request_time:.2f}秒")
        
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content']
            log_info(f"成功获取模型回答，长度: {len(answer)}")
            return answer
        else:
            log_error(f"模型没有返回有效回答: {result}")
            return "模型没有返回有效回答。"
    
    except Exception as e:
        log_error(f"直接对话请求失败: {e}")
        raise Exception(f"连接到模型服务失败: {e}")

def verify_answer(answer, docs):
    """验证回答是否基于参考资料"""
    # 如果没有参考资料或在直接对话模式下，直接返回原始回答
    if not docs:
        return answer
        
    # 提取回答中的关键句子
    answer_sentences = re.split(r'[。！？.!?]', answer)
    answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
    
    # 检查每个句子是否在参考资料中有对应内容
    all_docs_text = " ".join([doc.get("text", "") for doc in docs])
    
    unsupported = []
    for sentence in answer_sentences:
        if len(sentence) < 10:  # 忽略太短的句子
            continue
            
        # 检查句子的关键部分是否在参考资料中
        words = jieba.lcut(sentence)
        key_phrase = "".join(words[:min(10, len(words))])
        
        if key_phrase not in all_docs_text and sentence not in all_docs_text:
            unsupported.append(sentence)
    
    if unsupported:
        warning = "\n\n⚠️ 警告：以下内容可能不完全基于参考资料："
        for sentence in unsupported[:2]:  # 最多显示2个
            warning += f"\n- {sentence}"
        
        return answer + warning
    
    return answer

def two_stage_generation(query, docs, model="qwen2.5-instruct"):
    """两阶段生成：先提取信息，再生成回答"""
    # 第一阶段：从参考资料中提取相关信息
    extraction_prompt = f"""从以下参考资料中提取与问题"{query}"直接相关的关键信息，不要添加任何不在参考资料中的内容：

参考资料：
"""
    for i, doc in enumerate(docs):
        content = doc.get("text", "")
        extraction_prompt += f"[文档{i+1}] {content}\n\n"
    
    # 调用模型提取信息
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的信息提取助手，只提取参考资料中与问题相关的信息，不添加任何额外内容。"},
            {"role": "user", "content": extraction_prompt}
        ],
        "temperature": 0.1
    }
    
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
    result = response.json()
    extracted_info = result['choices'][0]['message']['content']
    
    # 第二阶段：基于提取的信息生成回答
    answer_prompt = f"""基于以下提取的信息回答问题"{query}"。如果信息不足以回答问题，请明确说明：

提取的信息：
{extracted_info}

回答："""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的知识库助手，只使用提供的信息回答问题，不添加任何额外内容。"},
            {"role": "user", "content": answer_prompt}
        ],
        "temperature": 0.3
    }
    
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
    result = response.json()
    final_answer = result['choices'][0]['message']['content']
    
    return final_answer

def two_stage_generation_with_chunking(query, docs, model="qwen2.5-instruct", max_tokens=4000):
    """处理大量文档的两阶段生成"""
    # 计算总文本长度
    total_text = ""
    for doc in docs:
        total_text += doc.get("text", "")
    
    # 如果总文本太长，分批处理
    if len(total_text) > max_tokens:
        log_info(f"文档总长度 {len(total_text)} 超过限制，分批处理")
        
        # 按相关性排序文档
        sorted_docs = sorted(docs, key=lambda x: x.get("score", 1.0))
        
        # 分批处理，确保每批不超过最大长度
        batches = []
        current_batch = []
        current_length = 0
        
        for doc in sorted_docs:
            doc_text = doc.get("text", "")
            doc_length = len(doc_text)
            
            if current_length + doc_length > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [doc]
                current_length = doc_length
            else:
                current_batch.append(doc)
                current_length += doc_length
        
        if current_batch:
            batches.append(current_batch)
        
        log_info(f"文档分为 {len(batches)} 批处理")
        
        # 分批生成回答
        all_extracted_info = []
        
        for i, batch in enumerate(batches):
            log_info(f"处理第 {i+1}/{len(batches)} 批文档")
            # 提取每批文档的信息
            batch_info = extract_information(query, batch, model)
            all_extracted_info.append(batch_info)
        
        # 合并所有提取的信息
        combined_info = "\n\n".join(all_extracted_info)
        
        # 基于合并的信息生成最终回答
        return generate_final_answer(query, combined_info, model)
    else:
        # 如果文档不多，使用标准两阶段生成
        return two_stage_generation(query, docs, model)

def make_api_request_with_retry(url, payload, max_retries=3, timeout=60):
    """带重试机制的API请求函数"""
    import time
    
    for attempt in range(max_retries):
        try:
            log_info(f"API请求尝试 {attempt+1}/{max_retries}")
            response = requests.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}, 
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            log_error(f"请求超时 (尝试 {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                # 指数退避策略
                wait_time = 2 ** attempt
                log_info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                log_error("达到最大重试次数，放弃请求")
                raise
        except Exception as e:
            log_error(f"请求失败: {e}")
            raise

def generate_answer_streaming(query, docs, model="qwen2.5-instruct", temperature=0.3):
    """使用流式响应生成回答"""
    # 构建提示词
    prompt = """你是一个严格的知识库助手，必须遵循以下规则：
1. 你的回答必须完全基于提供的参考资料
2. 在回答中使用[文档X]格式标注信息来源
3. 如果参考资料中没有足够信息回答问题，请明确说明
4. 不要编造或添加参考资料中没有的信息

参考资料:
"""
    for i, doc in enumerate(docs):
        content = doc.get("text", "")
        source = doc.get("document_title", "未知来源")
        prompt += f"\n[文档{i+1}] {content}\n来源: {source}\n"
    
    prompt += f"\n用户问题: {query}\n\n请严格按照上述规则，基于参考资料回答问题："
    
    log_info(f"生成流式提示词，长度: {len(prompt)}")
    
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的知识库助手，只能使用提供的参考资料回答问题，不得添加任何参考资料中没有的信息。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": True  # 启用流式响应
    }
    
    try:
        # 使用流式响应
        with requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=120) as response:
            response.raise_for_status()
            
            # 用于累积完整回答
            full_answer = ""
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_answer += content
                                    # 这里可以逐步更新UI显示
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            # 如果没有生成任何内容，返回一个友好的提示
            if not full_answer:
                yield "抱歉，我无法基于提供的参考资料回答这个问题。"
                
    except requests.exceptions.Timeout:
        yield "\n\n[生成超时] 回答生成时间过长，请尝试简化问题或稍后再试。"
    except Exception as e:
        log_error(f"流式生成回答时出错: {e}")
        yield f"\n\n[生成错误] {str(e)}"

def extract_information(query, docs, model="qwen2.5-instruct"):
    """从文档中提取与查询相关的信息"""
    extraction_prompt = f"""从以下参考资料中提取与问题"{query}"直接相关的关键信息，不要添加任何不在参考资料中的内容：

参考资料：
"""
    for i, doc in enumerate(docs):
        content = doc.get("text", "")
        extraction_prompt += f"[文档{i+1}] {content}\n\n"
    
    # 调用模型提取信息
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的信息提取助手，只提取参考资料中与问题相关的信息，不添加任何额外内容。"},
            {"role": "user", "content": extraction_prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        result = make_api_request_with_retry(
            url=url,
            payload=payload,
            max_retries=3,
            timeout=90
        )
        
        if 'choices' in result and len(result['choices']) > 0:
            extracted_info = result['choices'][0]['message']['content']
            log_info(f"成功提取信息，长度: {len(extracted_info)}")
            return extracted_info
        else:
            log_error("提取信息失败")
            return "无法从参考资料中提取相关信息。"
    except Exception as e:
        log_error(f"提取信息时出错: {e}")
        return f"提取信息时出错: {e}"

def generate_final_answer(query, extracted_info, model="qwen2.5-instruct"):
    """基于提取的信息生成最终回答"""
    answer_prompt = f"""基于以下提取的信息回答问题"{query}"。如果信息不足以回答问题，请明确说明：

提取的信息：
{extracted_info}

回答："""
    
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的知识库助手，只使用提供的信息回答问题，不添加任何额外内容。"},
            {"role": "user", "content": answer_prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        result = make_api_request_with_retry(
            url=url,
            payload=payload,
            max_retries=3,
            timeout=90
        )
        
        if 'choices' in result and len(result['choices']) > 0:
            final_answer = result['choices'][0]['message']['content']
            log_info(f"成功生成最终回答，长度: {len(final_answer)}")
            return final_answer
        else:
            log_error("生成最终回答失败")
            return "抱歉，无法生成回答。"
    except Exception as e:
        log_error(f"生成最终回答时出错: {e}")
        return f"生成回答时出错: {e}"

def two_stage_streaming_generation(query, docs, model="qwen2.5-instruct", temperature=0.3):
    """两阶段流式生成：先提取信息，再生成回答"""
    # 第一阶段：从参考资料中提取相关信息
    yield "正在分析参考资料..."
    
    extraction_prompt = f"""从以下参考资料中提取与问题"{query}"直接相关的关键信息，不要添加任何不在参考资料中的内容：

参考资料：
"""
    for i, doc in enumerate(docs):
        content = doc.get("text", "")
        extraction_prompt += f"[文档{i+1}] {content}\n\n"
    
    # 调用模型提取信息
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格的信息提取助手，只提取参考资料中与问题相关的信息，不添加任何额外内容。"},
            {"role": "user", "content": extraction_prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        result = make_api_request_with_retry(
            url=url,
            payload=payload,
            max_retries=3,
            timeout=90
        )
        
        if 'choices' in result and len(result['choices']) > 0:
            extracted_info = result['choices'][0]['message']['content']
            log_info(f"成功提取信息，长度: {len(extracted_info)}")
            
            # 第二阶段：基于提取的信息生成回答
            yield "正在生成回答..."
            
            # 使用流式响应生成最终回答
            answer_prompt = f"""基于以下提取的信息回答问题"{query}"。如果信息不足以回答问题，请明确说明：

提取的信息：
{extracted_info}

回答："""
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个严格的知识库助手，只使用提供的信息回答问题，不添加任何额外内容。"},
                    {"role": "user", "content": answer_prompt}
                ],
                "temperature": temperature,
                "stream": True
            }
            
            # 标记实际回答的开始
            yield "__START_ANSWER__"  # 这是一个特殊标记，会在UI代码中被处理
            
            with requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=120) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
        else:
            # 标记实际回答的开始
            yield "__START_ANSWER__"
            yield "无法从参考资料中提取相关信息。"
    except Exception as e:
        log_error(f"两阶段流式生成时出错: {e}")
        # 标记实际回答的开始
        yield "__START_ANSWER__"
        yield f"生成回答时出错: {e}"

def generate_answer_with_recovery(query, docs, model="qwen2.5-instruct", max_retries=2):
    """带有错误恢复的流式生成"""
    for attempt in range(max_retries + 1):
        try:
            yield from generate_answer_streaming(query, docs, model)
            break  # 如果成功完成，跳出循环
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                yield f"\n\n[连接超时] 正在尝试重新连接 (尝试 {attempt+1}/{max_retries})...\n"
                time.sleep(2)  # 短暂等待后重试
            else:
                yield f"\n\n[连接超时] 达到最大重试次数。请稍后再试。"
        except Exception as e:
            if attempt < max_retries:
                yield f"\n\n[发生错误] 正在尝试恢复 (尝试 {attempt+1}/{max_retries})...\n"
                time.sleep(2)
            else:
                yield f"\n\n[发生错误] 无法完成回答生成: {str(e)}"

def direct_chat_streaming(messages, model="qwen2.5-instruct", temperature=0.3):
    """流式直接对话模式"""
    # 准备最近的消息历史
    recent_messages = messages[-10:]  # 只使用最近的10条消息
    
    # 转换消息格式
    formatted_messages = []
    for msg in recent_messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 添加系统消息
    formatted_messages.insert(0, {
        "role": "system",
        "content": "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"
    })
    
    url = f"{MODEL_API_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "stream": True  # 启用流式响应
    }
    
    log_info(f"开始流式直接对话，消息数: {len(formatted_messages)}")
    
    # 标记实际回答的开始
    yield "__START_ANSWER__"
    
    try:
        # 使用流式响应
        with requests.post(url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=120) as response:
            response.raise_for_status()
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
    except requests.exceptions.Timeout:
        yield "\n\n[生成超时] 回答生成时间过长，请稍后再试。"
    except Exception as e:
        log_error(f"流式直接对话出错: {e}")
        yield f"\n\n[生成错误] {str(e)}"

if __name__ == "__main__":
    qa_page()
