import os
import logging
import numpy as np
import yaml
import json
import inspect  # <--- 必须加这一行！
# ... 其他代码 ...
import asyncio
from tqdm.asyncio import tqdm_asyncio  # 需要安装 tqdm: pip install tqdm
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from hirag.plugrag_config import resolve_graph_encoder_files
import nest_asyncio  # 1. 导入
nest_asyncio.apply() # 2. 应用补丁
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import openai

# ================= 配置加载部分 (保持不变) =================
config_path = os.environ.get("PLUGRAG_CONFIG", "config.example.yaml")
with open(config_path, 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)
batch_config = config.get("batch_query", {})
rerank_config = config.get("rerank", {})

OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai'].get('api_key') or os.environ.get("OPENAI_API_KEY")
OPENAI_URL = config['openai'].get('base_url') or os.environ.get("OPENAI_BASE_URL")
GLM_API_KEY = config['glm'].get('api_key') or os.environ.get("GLM_API_KEY")
GLM_MODEL = config['glm']['model']
GLM_URL = config['glm']['base_url']
graph_encoder_config = resolve_graph_encoder_files(config, batch_config)

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

graph_func = HiRAG(working_dir=config['hirag']['working_dir'],
                      enable_llm_cache=config['hirag']['enable_llm_cache'],
                      embedding_func=OPENAI_embedding,
                      best_model_func=OPENAI_model_if_cache,
                      cheap_model_func=OPENAI_model_if_cache,
                      enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
                      embedding_batch_num=config['hirag']['embedding_batch_num'],
                      embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                      enable_naive_rag=config['hirag']['enable_naive_rag'],
                      rgcn_enabled=graph_encoder_config["enabled"],
                      rgcn_embeddings_file=graph_encoder_config["embeddings_file"],
                      rgcn_mapping_file=graph_encoder_config["mapping_file"],
                      rerank_api_key=rerank_config.get("api_key") or config["openai"].get("api_key") or os.environ.get("RERANK_API_KEY"),
                      rerank_base_url=rerank_config.get("base_url") or config["openai"].get("base_url") or os.environ.get("RERANK_BASE_URL"),
                      rerank_model_name=rerank_config.get("model", "BAAI/bge-reranker-v2-m3"))

# [重要] 如果之前已经跑过 insert，这里建议注释掉，否则每次启动都会重新索引，浪费时间。
# with open("datasets/mix/mix_unique_contexts.json", "r", encoding="utf-8") as f:
#    graph_func.insert(f.read())

# ================= 修复版核心逻辑 =================

INPUT_FILE = os.environ.get("PLUGRAG_INPUT_FILE", batch_config.get("input_file", "queries.jsonl"))
OUTPUT_FILE = os.environ.get("PLUGRAG_OUTPUT_FILE", batch_config.get("output_file", "answers.jsonl"))
CONCURRENCY_LIMIT = int(os.environ.get("PLUGRAG_CONCURRENCY", batch_config.get("concurrency", 5)))

async def process_single_query(semaphore, line_data, param):
    """处理单个查询，智能判断是否需要 await"""
    async with semaphore:
        query_text = line_data.get("query")
        if not query_text:
            return None
        
        try:
            # 1. 获取返回值，不要直接 await
            # Use QueryParam.mode to switch PlugRag variants.
            result_or_coro = graph_func.query(query_text, param=param)
            
            # 2. 智能类型判断
            if inspect.iscoroutine(result_or_coro) or inspect.isawaitable(result_or_coro):
                # 是协程 -> 等待
                answer = await result_or_coro
            elif isinstance(result_or_coro, str):
                # 是字符串 -> 直接用
                answer = result_or_coro
            else:
                # 其他情况 -> 尝试转字符串
                answer = str(result_or_coro)

            # 3. 检查答案是否为空
            if not answer:
                answer = "No answer found."

            return json.dumps({"query": query_text, "answer": answer}, ensure_ascii=False)
            
        except Exception as e:
            # 打印简短错误日志
            print(f"[Error] Q: {query_text[:15]}... | {e}")
            # 返回带错误标记的 JSON，方便后续重试
            return json.dumps({"query": query_text, "answer": f"ERROR: {str(e)}"}, ensure_ascii=False)

async def main_batch_process():
    # 1. 定义参数
    param = QueryParam(
        mode=batch_config.get("mode", "hi"),
        top_k=int(batch_config.get("top_k", 20)),
    )

    # 2. 读取输入
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_lines = [json.loads(line) for line in f if line.strip()]

    # 3. 断点续传逻辑 (Resume Logic)
    existing_answers = {}
    if os.path.exists(OUTPUT_FILE):
        print("发现已存在的输出文件，正在检查断点...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("query")
                    ans = obj.get("answer")
                    # 只有非 ERROR 的答案才算有效
                    if q and ans and not ans.startswith("ERROR:"):
                        existing_answers[q] = ans
                except:
                    pass
    print(f"已跳过 {len(existing_answers)} 个已完成的问题。")

    # 4. 准备任务
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    queries_to_process = []

    for item in input_lines:
        q_text = item.get("query")
        if q_text in existing_answers:
            continue # 跳过已完成
        
        # 将需要处理的任务加入列表
        tasks.append(process_single_query(semaphore, item, param))
        queries_to_process.append(item)

    if not tasks:
        print("所有问题都已处理完成！")
        return

    print(f"剩余 {len(tasks)} 个问题需要处理...")

    # 5. 执行并发
    results = await tqdm_asyncio.gather(*tasks)

    # 6. 写入结果 (追加模式)
    # 注意：为了防止重复写入，建议先把之前的有效结果读出来，重写整个文件，或者只追加新的
    # 这里采用：重写整个文件以保持顺序和整洁
    
    final_results_map = existing_answers.copy()
    
    # 更新新跑出来的结果
    for res_str in results:
        if res_str:
            res_obj = json.loads(res_str)
            final_results_map[res_obj["query"]] = res_obj["answer"]
    
    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for q_in in input_lines: # 按原始顺序写入
            q_key = q_in.get("query")
            if q_key in final_results_map:
                line_json = json.dumps({"query": q_key, "answer": final_results_map[q_key]}, ensure_ascii=False)
                f_out.write(line_json + "\n")
            else:
                # 理论上不应该走到这里，除非出错了没返回
                pass

    print(f"全部完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    # 解决 Jupyter/Loop 冲突
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(main_batch_process())
