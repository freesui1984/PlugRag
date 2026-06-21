import os
import logging
import numpy as np
import yaml
import asyncio  # 必须引入 asyncio 用于信号量
from nano_graphrag import GraphRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
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
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    retry_if_exception_type
)
import openai
def robust_api_retry():
    return retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,    # 错误码 429 (太快了)
            openai.APITimeoutError,   # 请求超时 (网络卡)
            openai.APIConnectionError,# 连接失败 (断网)
            openai.InternalServerError # 错误码 500+ (服务器崩了)
        )),
        # 等待策略：指数退避
        # # 第1次等1s，第2次等2s，第3次等4s... 最大等到60s
        # wait=wait_random_exponential(multiplier=1, max=60),
        # # 停止策略：最多重试 10 次 (你可以改得更大，比如 20)
        # stop=stop_after_attempt(10) 
        # 等待策略：指数退避，最小等待4秒，最大等待120秒
        wait=wait_exponential(multiplier=2, min=4, max=120),
        # 停止策略：最多重试 20 次，给更多机会处理超时
        stop=stop_after_attempt(20) 
        
    )
@robust_api_retry()
@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['openai_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def OPENAI_embedding(texts: list[str]) -> np.ndarray:
    # openai_async_client = AsyncOpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
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
    embedding = await client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)
@robust_api_retry()
async def OPENAI_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # if system_prompt:
    #     print(f"\n[SYSTEM PROMPT]:\n{system_prompt}")
    # if history_messages:
    #     print(f"\n[HISTORY]:\n{history_messages}")
        
    print(f"\n[USER PROMPT (包含所有上下文)]:\n{prompt}")
    print("="*80)
    openai_async_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY, base_url=OPENAI_URL, timeout=600.0,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(OPENAI_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, temperature = 0.0,**kwargs
    )

    # # Cache the response if having-------------------
    # if hashing_kv is not None:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": OPENAI_MODEL}}
    #     )
    # # -----------------------------------------------------
    # return response.choices[0].message.content # 获取原始返回内容
    raw_content = response.choices[0].message.content

    # ========================== 核心修改区域 ==========================
    # 过滤 <think>...</think> 标签内容
    # flags=re.DOTALL 确保 . 能匹配换行符，因为思考过程通常包含多行
    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    # 兜底逻辑：如果过滤后内容为空（例如模型只输出了思考过程），则保留原始内容，避免返回空字符串
    if not clean_content:
        clean_content = raw_content
    # ================================================================

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            # 注意：这里我们缓存的是【过滤后】的内容 (clean_content)
            # 这样下次命中缓存时，直接拿到的就是干净的答案
            {args_hash: {"return": clean_content, "model": OPENAI_MODEL}}
        )
    # -----------------------------------------------------
    
    return clean_content

# --- 4. 解析配置 ---
model_config = config.get('models', {})
storage_config = config.get('storage', {})
retrieval_config = config.get('retrieval', {})
rgcn_train_config = config.get('rgcn_training', {})
HIRAG_GRAPH_DIR = storage_config.get('working_dir', './nano_graphrag_cache')

# --- 5. 初始化 GraphRAG ---
rag = GraphRAG(
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

async def main():
    # --- 5. 从 HiRAG 的图引导启动 ---
    # await rag.abootstrap_from_existing_graph(rebuild_reports=True)


    # --- 7. 执行检索问答 ---
    query = "What does the book suggest about the use of histograms in data analysis?"

    query_params = QueryParam(
        # mode="rgcn", 
        mode="local", 
        top_k=retrieval_config.get('top_k', 20),
        rgcn_mode=retrieval_config.get('rgcn_enabled', False)
    )

    result = await rag.aquery(query, param=query_params)

    print("\n" + "="*50)
    print("--- Final Answer ---")
    print(result)
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
