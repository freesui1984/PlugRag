import os
import logging
import numpy as np
import yaml
import json
import inspect  # <--- 必须加这一行！
# ... 其他代码 ...
import asyncio
from tqdm.asyncio import tqdm_asyncio  # 需要安装 tqdm: pip install tqdm
from nano_graphrag import GraphRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
import nest_asyncio  # 1. 导入
nest_asyncio.apply() # 2. 应用补丁
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import re # 1. 记得引入正则表达式库
# Load configuration from YAML file
config_path = os.environ.get("PLUGRAG_CONFIG", "config_example.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
import openai
# Extract configurations from the selected YAML file
api_config = config.get('api_keys', {})
llm_config = config.get('llm', {})
model_config = config.get('models', {})
storage_config = config.get('storage', {})
retrieval_config = config.get('retrieval', {})
batch_config = config.get('batch_query', {})

OPENAI_EMBEDDING_MODEL = model_config.get('embedding_model', 'BAAI/bge-m3')
OPENAI_MODEL = llm_config.get('model', 'gpt-4o-mini')
OPENAI_API_KEY = api_config.get('openai_api_key') or os.environ.get("OPENAI_API_KEY")
OPENAI_URL = api_config.get('openai_base_url') or os.environ.get("OPENAI_BASE_URL") or 'https://api.siliconflow.cn/v1'

# For backward compatibility with the user's defined functions
config['model_params'] = {
    'openai_embedding_dim': 1024, # BGE-M3
    'max_token_size': 8192,
    'glm_embedding_dim': 1024
}
config['glm'] = {
    'api_key': 'none',
    'model': 'none',
    'base_url': 'none',
    'embedding_model': 'none'
}

GLM_API_KEY = config['glm']['api_key']
GLM_MODEL = config['glm']['model']
GLM_URL = config['glm']['base_url']


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

def robust_api_retry():
    return retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError
        )),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(10) 
    )

@robust_api_retry()
@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['openai_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def OPENAI_embedding(texts: list[str]) -> np.ndarray:
    model_name = "BAAI/bge-m3" 
    openai_async_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        timeout=600.0
    )
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

@robust_api_retry()
@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = config['glm']['embedding_model']
    client = AsyncOpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)

@robust_api_retry()
async def OPENAI_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY, base_url=OPENAI_URL, timeout=600.0
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(OPENAI_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, **kwargs, temperature=0.0
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": OPENAI_MODEL}}
        )
    return response.choices[0].message.content

# ================= HiRAG 初始化 =================
# 注意：如果在批量跑脚本时，通常不需要每次都 re-index (insert)。
# 如果已构建过索引，建议注释掉 graph_func.insert 部分。
# --- 4. 解析配置 ---
model_config = config.get('models', {})
storage_config = config.get('storage', {})
retrieval_config = config.get('retrieval', {})
rgcn_train_config = config.get('rgcn_training', {})
HIRAG_GRAPH_DIR = storage_config.get('working_dir', './nano_graphrag_cache')
graph_func = GraphRAG(
    working_dir=HIRAG_GRAPH_DIR,
    embedding_func=OPENAI_embedding,
    best_model_func=OPENAI_model_if_cache,
    cheap_model_func=OPENAI_model_if_cache,
    rerank_api_key=api_config.get('openai_api_key'), # 使用 SiliconFlow API Key
    rerank_base_url=api_config.get('openai_base_url'), # 使用 SiliconFlow URL
    rerank_model_name=model_config.get('rerank_model', 'BAAI/bge-reranker-v2-m3'),
    rgcn_enabled=retrieval_config.get('rgcn_enabled', False),
    rgcn_embeddings_file=storage_config.get('rgcn_embeddings_file', 'rgcn_embeddings.pt'),
    rgcn_mapping_file=storage_config.get('rgcn_mapping_file', 'rgcn_embeddings_mapping.json'),
    rgcn_expand_k=retrieval_config.get('rgcn_expand_k', 5),
    rgcn_rerank_alpha=retrieval_config.get('rgcn_rerank_alpha', 0.5),
)

# [重要] 如果之前已经跑过 insert，这里建议注释掉，否则每次启动都会重新索引，浪费时间。
# with open("datasets/mix/mix_unique_contexts.json", "r", encoding="utf-8") as f:
#    graph_func.insert(f.read())

# ================= 修复版核心逻辑 =================

# ================= 修复版核心逻辑 (顺序单条执行、无限重试、防崩溃) =================

INPUT_FILE = os.environ.get("PLUGRAG_INPUT_FILE", batch_config.get("input_file", "queries.jsonl"))
OUTPUT_FILE = os.environ.get("PLUGRAG_OUTPUT_FILE", batch_config.get("output_file", "answers.jsonl"))

# 配置重试和延迟时间
MAX_RETRIES = 50           # 单条问题最大重试次数（可设大一点，相当于一直重试）
RETRY_DELAY_BASE = 5       # 失败后的基础等待时间（秒）
SUCCESS_DELAY = 3          # 成功处理一条后，强制休息的时间（秒），防止请求过于频繁

from tqdm import tqdm # 换用普通 tqdm 即可

async def process_single_query_with_retry(line_data, param):
    """处理单个查询，自带无限循环重试和延迟退避机制"""
    query_text = line_data.get("query")
    if not query_text:
        return None
    
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # 1. 发起查询
            result_or_coro = graph_func.query(query_text, param=param)
            
            # 2. 智能类型判断
            if inspect.iscoroutine(result_or_coro) or inspect.isawaitable(result_or_coro):
                answer = await result_or_coro
            elif isinstance(result_or_coro, str):
                answer = result_or_coro
            else:
                answer = str(result_or_coro)

            # 3. 检查答案是否为空
            if not answer or "ERROR:" in answer:
                raise ValueError("API returned empty or error answer.")

            # 成功返回 JSON 字符串
            return json.dumps({"query": query_text, "answer": answer}, ensure_ascii=False)
            
        except Exception as e:
            attempt += 1
            error_msg = str(e)
            print(f"\n[Error] Q: {query_text[:15]}... | 第 {attempt} 次尝试失败 | 错误: {error_msg}")
            
            if attempt >= MAX_RETRIES:
                print(f"[Fatal] 彻底放弃该问题: {query_text[:15]}...")
                return json.dumps({"query": query_text, "answer": f"ERROR: {error_msg}"}, ensure_ascii=False)
            
            # 失败后休眠，随失败次数延长休眠时间（例如 5秒, 10秒, 15秒...）
            sleep_time = RETRY_DELAY_BASE * attempt
            print(f"--> 等待 {sleep_time} 秒后重试...")
            await asyncio.sleep(sleep_time)

async def main_batch_process():
    # 1. 定义参数
    param = QueryParam(
        mode=batch_config.get("mode", "plugrag_full"), 
        top_k=batch_config.get("top_k", retrieval_config.get('top_k', 20)),
        rgcn_mode=retrieval_config.get('rgcn_enabled', False)
    )

    # 2. 读取输入
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_lines = [json.loads(line) for line in f if line.strip()]

    # 3. 断点续传逻辑 (读取已经跑成功的问题)
    existing_answers = set()
    if os.path.exists(OUTPUT_FILE):
        print("发现已存在的输出文件，正在检查断点...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("query")
                    ans = obj.get("answer")
                    if q and ans and not ans.startswith("ERROR:"):
                        existing_answers.add(q)
                except:
                    pass
    print(f"已跳过 {len(existing_answers)} 个已完成的问题。")

    # 4. 筛选出还未处理的问题
    queries_to_process = [item for item in input_lines if item.get("query") not in existing_answers]

    if not queries_to_process:
        print("所有问题都已处理完成！")
        return

    print(f"剩余 {len(queries_to_process)} 个问题需要处理，将采用单线程逐条处理...")

    # 5. 顺序执行，写一条存一条（防止中途 503 导致前面跑的数据丢失）
    # 使用 'a' 追加模式打开文件
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for item in tqdm(queries_to_process, desc="处理进度"):
            # 阻塞等待这条问题处理完成（如果失败会在函数内无限重试）
            res_str = await process_single_query_with_retry(item, param)
            
            if res_str:
                # 跑完一条立马写入本地磁盘
                f_out.write(res_str + "\n")
                f_out.flush() # 强制刷新缓冲区到磁盘
                
            # 每成功处理完一条，强制休息一下，避免触发 API 频率限制 (Rate Limit)
            await asyncio.sleep(SUCCESS_DELAY)

    print(f"全部完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    # 解决 Jupyter/Loop 冲突
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(main_batch_process())
