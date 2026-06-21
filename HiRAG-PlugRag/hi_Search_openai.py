import os
import logging
import numpy as np
import yaml
import asyncio  # 必须引入 asyncio 用于信号量
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from hirag.plugrag_config import resolve_graph_encoder_files
import re # 1. 记得引入正则表达式库
# Load configuration from YAML file
config_path = os.environ.get("PLUGRAG_CONFIG", "config.example.yaml")
with open(config_path, 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)
batch_config = config.get("batch_query", {})
rerank_config = config.get("rerank", {})
graph_encoder_config = resolve_graph_encoder_files(config, batch_config)
import openai
# Extract configurations
OPENAI_EMBEDDING_MODEL = config['openai']['embedding_model']
OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai'].get('api_key') or os.environ.get("OPENAI_API_KEY")
OPENAI_URL = config['openai'].get('base_url') or os.environ.get("OPENAI_BASE_URL")
GLM_API_KEY = config['glm'].get('api_key') or os.environ.get("GLM_API_KEY")
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
        
    # print(f"\n[USER PROMPT (包含所有上下文)]:\n{prompt}")
    # print("="*80)
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
# comment this if the working directory has already been indexed
# with open("datasets/cs/cs_unique_contexts.json", "r", encoding="utf-8") as f:
#     graph_func.insert(f.read())


# print("Perform hi search:")
# 调整查询参数以减少上下文大小、
#   max_token_for_text_unit=4000,  # 减少文本单元的最大token数
#     max_token_for_local_context=4000,  # 减少本地上下文的最大token数
#     max_token_for_bridge_knowledge=2000,  # 减少桥接知识的最大token数
#     max_token_for_community_report=2000  # 减少社区报告的最大token数
param = QueryParam(
    mode="hi",
    # mode="plugrag_full",
    top_k=20,  # 减少检索的实体数量
    # top_m=10,   # 减少每个社区中检索的实体数量
    # max_token_for_text_unit=2000,  # 减少文本单元的最大token数
    # max_token_for_local_context=4000,  # 减少本地上下文的最大token数
    # max_token_for_bridge_knowledge=2000,  # 减少桥接知识的最大token数
    # max_token_for_community_report=2000  # 减少社区报告的最大token数
)
print(graph_func.query("What are the representations and warranties of the Company regarding its qualification, organization, and subsidiaries?", param=param))
# print("Perform hi search:")
# # 调整查询参数以减少上下文大小
# param = QueryParam(
#     mode="hi",
#     top_k=15,  # 减少检索的实体数量
#     top_m=10,   # 减少每个社区中检索的实体数量
# )
# print(graph_func.query("针对与牙齿断裂最好的处理方法是什么啊？", param=param))
