import re
import json
import jsonlines
import argparse
import os
import time
import copy
import yaml
from openai import OpenAI
# os.environ["OPENAI_API_KEY"] = ""

# max test queries
DATASET = "mix"
if DATASET == "mix":
    MAX_QUERIES = 130
elif DATASET == "cs" or DATASET == "agriculture" or DATASET == "legal":
    MAX_QUERIES = 100

def _deep_update(base, updates):
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_eval_config():
    config = {
        "deepseek": {
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        },
        "glm": {
            "model": os.environ.get("GLM_MODEL", "glm-4-plus"),
            "api_key": os.environ.get("GLM_API_KEY", ""),
            "base_url": os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        },
        "openai": {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
    }
    config_candidates = [
        os.environ.get("PLUGRAG_EVAL_CONFIG"),
        os.path.join(os.path.dirname(__file__), "eval_config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
        os.path.join(os.getcwd(), "config.example.yaml"),
        os.path.join(os.getcwd(), "config_example.yaml"),
    ]
    for config_path in config_candidates:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
            _deep_update(config, loaded)
            break
    return config


config = _load_eval_config()

EVALUATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _eval_path(*parts):
    return os.path.join(EVALUATION_ROOT, *parts)

# Extract configurations
DEEPSEEK_MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']

GLM_MODEL = config['glm']['model']
GLM_API_KEY = config['glm']['api_key']
GLM_URL = config['glm']['base_url']

OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']


def eval_oq_openai_batch(query_file, result1_file, result2_file, output_file_path):  # with original query
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)

    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        request_data = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{OPENAI_MODEL}",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"Batch {batch.id} has been created.")
    return batch.id


def batch_eval_gq_openai(query_file, result1_file, result2_file, output_file_path):  # with generated query
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)

    with open(query_file, "r") as f:
        data = f.read()

    queries = re.findall(r"- Question \d+: (.+)", data)
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        request_data = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{OPENAI_MODEL}",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    print(f"Batch {batch.id} has been created.")
    return batch.id


def eval_oq_glm(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_URL)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=4095,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            try:
                evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
                evaluations.append(evaluation)
                print(f"Successfully evaluate {i}/{len(queries)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(e)
                    print("Failed after maximum retries")

    with jsonlines.open(output_file_path.replace(".jsonl", "_result_glm.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def eval_oq_deepseek(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_URL)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion. And you need to be very fair and have no bias towards the order.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=6400,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            try:
                evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
                evaluations.append(evaluation)
                print(f"Successfully evaluate {i}/{len(queries)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(e)
                    print("Failed after maximum retries")

    with jsonlines.open(output_file_path.replace(".jsonl", "_result_deepseek.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def eval_oq_openai_待定(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
   
    # ==============================================================================
    # 1. 定义稳健读取函数 (内置 helper，防止外部调用找不到)
    # ==============================================================================
    def read_jsonl_absolutely(file_path, field_target=""):
        print(f"📖 正在读取: {file_path}")
        data_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace(): pos += 1
                if pos >= len(content): break
                try:
                    obj, end_pos = decoder.raw_decode(content, idx=pos)
                    pos = end_pos
                    
                    if field_target == "query":
                        # 兼容 'query' 或 'input'
                        val = obj.get("query") or obj.get("input", "")
                        data_list.append(val)
                    elif field_target == "answer":
                        val = obj.get("answer", "")
                        if not isinstance(val, str): val = str(val)
                        data_list.append(val)
                    else:
                        data_list.append(obj)
                except json.JSONDecodeError:
                    pos += 1 # 跳过坏字符
        except FileNotFoundError:
            print(f"❌ 文件未找到: {file_path}")
            return []
        return data_list

    # ==============================================================================
    # 2. 加载数据 (使用稳健模式)
    # ==============================================================================
    queries = read_jsonl_absolutely(query_file, field_target="query")
    queries = queries[:MAX_QUERIES]

    answers1 = read_jsonl_absolutely(result1_file, field_target="answer")
    answers1 = answers1[:MAX_QUERIES]

    answers2 = read_jsonl_absolutely(result2_file, field_target="answer")
    answers2 = answers2[:MAX_QUERIES]

    # 检查对齐
    if not (len(queries) == len(answers1) == len(answers2)):
        print(f"❌ [Error] 数据长度不一致: Q={len(queries)}, A1={len(answers1)}, A2={len(answers2)}")
        print("⚠️ 请检查生成结果文件是否完整。")
        return None

    print(f"✅ 数据加载成功: {len(queries)} 组数据")
    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion. And you need to be very fair and have no bias towards the order.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        # try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=6400,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            # try:
            #     evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
            #     evaluations.append(evaluation)
            #     print(f"Successfully evaluate {i}/{len(queries)}")
            #     break
            # except Exception as e:
            #     if attempt < max_retries - 1:
            #         time.sleep(retry_delay)
            #     else:
            #         print(e)
            #         print("Failed after maximum retries")
            try:
                # --- 解析逻辑 ---
                json_str = response
                
                # 如果不是第一次尝试，说明原始字符串有问题，尝试修复反斜杠
                if attempt > 0:
                    # 正则含义：找到所有 "后面不跟合法转义字符" 的反斜杠，把它变成双反斜杠
                    # 合法转义字符包括: " \ / b f n r t u
                    json_str = re.sub(r'\\(?![/u"\\bfnrt])', r'\\\\', json_str)

                # 尝试提取 JSON (兼容 Markdown)
                match = re.search(r"\{[\s\S]*\}", json_str)
                if match:
                    json_str = match.group()
                else:
                    # 暴力清洗
                    json_str = json_str.replace("```json", "").replace("```", "").strip()

                evaluation = json.loads(json_str)
                
                # --- 成功 ---
                evaluations.append(evaluation)
                print(f"Successfully evaluated {i}/{len(queries)}")
                break 
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Parse failed (Attempt {attempt+1}), trying to fix escapes... Error: {e}")
                    # 下一次循环会进入 if attempt > 0 的修复逻辑
                    time.sleep(retry_delay)
                else:
                    # 彻底失败
                    print(f"❌ Failed to parse response for query {i}: {e}")
                    print(f"   Raw output snippet: {response[:100]}...")
                    # 防止错位，填入一个空结果
                    evaluations.append({
                        "Overall Winner": {"Winner": "Error", "Explanation": "JSON Parse Error"}
                    })
    with jsonlines.open(output_file_path.replace(".jsonl", "_result_openai.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def fetch_eval_result_glm(output_file):
    result = []
    with open(output_file.replace(".jsonl", "_result_glm.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            result.append(item)
    
    result_0 = result[0]
    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
    print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
    print("Empowerment:")
    print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
    print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
    print("Diversity:")
    print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
    print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
    print("Overall:")
    print(f"Answer 1: {overall_winner_ans1 / len(result)}")
    print(f"Answer 2: {overall_winner_ans2 / len(result)}")


def fetch_eval_result_deepseek(output_file):
    result = []
    with open(output_file.replace(".jsonl", "_result_deepseek.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            result.append(item)
    
    result_0 = result[0]
    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
    print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
    print("Empowerment:")
    print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
    print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
    print("Diversity:")
    print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
    print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
    print("Overall:")
    print(f"Answer 1: {overall_winner_ans1 / len(result)}")
    print(f"Answer 2: {overall_winner_ans2 / len(result)}")

def eval_oq_openai(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
   
    # ==============================================================================
    # 1. 定义稳健读取函数 (保持不变)
    # ==============================================================================
    def read_jsonl_absolutely(file_path, field_target=""):
        print(f"📖 正在读取: {file_path}")
        data_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace(): pos += 1
                if pos >= len(content): break
                try:
                    obj, end_pos = decoder.raw_decode(content, idx=pos)
                    pos = end_pos
                    
                    if field_target == "query":
                        val = obj.get("query") or obj.get("input", "")
                        data_list.append(val)
                    elif field_target == "answer":
                        val = obj.get("answer", "")
                        if not isinstance(val, str): val = str(val)
                        data_list.append(val)
                    else:
                        data_list.append(obj)
                except json.JSONDecodeError:
                    pos += 1 
        except FileNotFoundError:
            print(f"❌ 文件未找到: {file_path}")
            return []
        return data_list

    # ==============================================================================
    # 2. 加载数据 (保持不变)
    # ==============================================================================
    queries = read_jsonl_absolutely(query_file, field_target="query")
    if 'MAX_QUERIES' in globals(): queries = queries[:MAX_QUERIES]

    answers1 = read_jsonl_absolutely(result1_file, field_target="answer")
    if 'MAX_QUERIES' in globals(): answers1 = answers1[:MAX_QUERIES]

    answers2 = read_jsonl_absolutely(result2_file, field_target="answer")
    if 'MAX_QUERIES' in globals(): answers2 = answers2[:MAX_QUERIES]

    # 检查对齐
    if not (len(queries) == len(answers1) == len(answers2)):
        print(f"❌ [Error] 数据长度不一致: Q={len(queries)}, A1={len(answers1)}, A2={len(answers2)}")
        return None

    print(f"✅ 数据加载成功: {len(queries)} 组数据")
    
    # 翻倍数据以消除位置偏差
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    evaluations = []
    
    # 定义合法的 Winner 关键词，用于校验
    VALID_WINNERS = ["Answer 1", "Answer 2"]

    # ==============================================================================
    # 3. 核心评测循环 (这里是主要修改的地方)
    # ==============================================================================
    print(f"🚀 开始评测 {len(queries)} 组任务...")

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion. And you need to be very fair and have no bias towards the order.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        max_retries = 10
        valid_result = None
        
        # --- 重试循环开始 ---
        for attempt in range(max_retries):
            try:
                # 1. API 请求移入循环内部
                # 这样如果出错，下一次循环会重新生成，而不是只修补字符串
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.0, # 低温度保证稳定性
                    max_tokens=6400,
                    # response_format={"type": "json_object"} # 关键：强制 JSON 模式
                )

                content = response.choices[0].message.content.strip()

                # 2. 简单的清洗 (防止模型虽然开启JSON模式还是带了```json)
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                # 3. 解析 JSON
                evaluation = json.loads(content)
                
                metrics = ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]
                for m in metrics:
                    if m not in evaluation:
                        raise ValueError(f"缺少指标: {m}")
                    
                    w_text = evaluation[m].get("Winner", "")
                    is_valid = any(kw in w_text for kw in VALID_WINNERS)
                    if not is_valid:
                        raise ValueError(f"无效的 Winner: '{w_text}' (必须包含 Answer 1 或 Answer 2)")

                valid_result = evaluation
                # print(f"✅ Success: {i}")
                break 
            
            except Exception as e:
                print(f"⚠️ Retry {attempt+1}/{max_retries} for Query {i}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(6)
                else:
                    # 彻底失败，填入 Error
                    print(f"❌ Failed Query {i} after retries: {e}")
                    valid_result = {
                        "Overall Winner": {"Winner": "Error", "Explanation": str(e)}
                    }
        
        # --- 重试循环结束 ---
        evaluations.append(valid_result)
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(queries)}")

    # ==============================================================================
    # 4. 写入文件 (保持不变)
    # ==============================================================================
    output_filename = output_file_path.replace(".jsonl", "_result_openai.jsonl")
    with jsonlines.open(output_filename, mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_filename}")

# def fetch_eval_result_openai(output_file):
#     result = []
#     # 获取目标文件名
#     target_filename = output_file
    
#     # 只有当文件名里没有 "_result_openai" 时才添加后缀
#     if "_result_openai.jsonl" not in target_filename:
#         target_filename = target_filename.replace(".jsonl", "_result_openai.jsonl")
    
#     print(f"📖 正在读取结果文件: {target_filename}")
#     # with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'r') as f:
#     with open(target_filename, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             item = json.loads(line)
#             result.append(item)
    
#     result_0 = result[0]
#     comprehensiveness = result_0['Comprehensiveness']['Winner']
#     comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
#     empowerment = result_0['Empowerment']['Winner']
#     empowerment_explanation = result_0['Empowerment']['Explanation']
#     diversity = result_0['Diversity']['Winner']
#     diversity_explanation = result_0['Diversity']['Explanation']
#     overall_winner = result_0['Overall Winner']['Winner']
#     overall_explanation = result_0['Overall Winner']['Explanation']

#     print("===================================Comprehensiveness===================================")
#     print(f"Winner:\n{comprehensiveness}")
#     print(f"Explanation:\n{comprehensiveness_explanation}")
#     print("======================================Empowerment======================================")
#     print(f"Winner:\n{empowerment}")
#     print(f"Explanation:\n{empowerment_explanation}")
#     print("=======================================Diversity=======================================")
#     print(f"Winner:\n{diversity}")
#     print(f"Explanation:\n{diversity_explanation}")
#     print("=========================================Winner=========================================")
#     print(f"Winner:\n{overall_winner}")
#     print(f"Explanation:\n{overall_explanation}")


#     comprehensiveness_winner_ans1 = 0
#     comprehensiveness_winner_ans2 = 0
#     for i, item in enumerate(result):
#         if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
#             comprehensiveness_winner_ans1 += 1
#         elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
#             comprehensiveness_winner_ans2 += 1
#         elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
#             comprehensiveness_winner_ans2 += 1
#         elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
#             comprehensiveness_winner_ans1 += 1
#     empowerment_winner_ans1 = 0
#     empowerment_winner_ans2 = 0
#     for i, item in enumerate(result):
#         if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
#             empowerment_winner_ans1 += 1
#         elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
#             empowerment_winner_ans2 += 1
#         elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
#             empowerment_winner_ans2 += 1
#         elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
#             empowerment_winner_ans1 += 1
#     diversity_winner_ans1 = 0
#     diversity_winner_ans2 = 0
#     for i, item in enumerate(result):
#         if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
#             diversity_winner_ans1 += 1
#         elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
#             diversity_winner_ans2 += 1
#         elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
#             diversity_winner_ans2 += 1
#         elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
#             diversity_winner_ans1 += 1
#     overall_winner_ans1 = 0
#     overall_winner_ans2 = 0
#     for i, item in enumerate(result):
#         if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
#             overall_winner_ans1 += 1
#         elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
#             overall_winner_ans2 += 1
#         elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
#             overall_winner_ans2 += 1
#         elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
#             overall_winner_ans1 += 1
#     print("======================================Winner Accuracy=========================================")
#     print("Comprehensiveness:")
#     print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
#     print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
#     print("Empowerment:")
#     print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
#     print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
#     print("Diversity:")
#     print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
#     print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
#     print("Overall:")
#     print(f"Answer 1: {overall_winner_ans1 / len(result)}")
#     print(f"Answer 2: {overall_winner_ans2 / len(result)}")

def fetch_eval_result_openai(output_file):
    # 1. 读取文件
    target_filename = output_file
    if "_result_openai.jsonl" not in target_filename:
        target_filename = target_filename.replace(".jsonl", "_result_openai.jsonl")
    
    results_list = []
    with open(target_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results_list.append(json.loads(line))

    total_samples = len(results_list)
    # 假设数据是对称的（一半正序，一半反序）
    mid_point = total_samples // 2 
    
    print(f"📊 Total Samples: {total_samples} (Switch Point: {mid_point})")

    metrics = ["Comprehensiveness", "Empowerment", "Diversity", "Overall Winner"]
    counts = {m: {"MyModel": 0, "Baseline": 0, "Tie": 0} for m in metrics}

    for i, item in enumerate(results_list):
        # 兼容 custom_id 或直接索引
        # 如果没有 custom_id，我们就假设它是按顺序生成的
        # 0 ~ mid_point-1 : Ans1=MyModel, Ans2=Baseline
        # mid_point ~ end : Ans1=Baseline, Ans2=MyModel
        
        is_swapped = i >= mid_point
        
        # 解析结果 (兼容不同层级)
        if 'response' in item:
             content = item['response']['body']['choices'][0]['message']['content']
        else:
             # 兼容直接存储结果的情况
             content = json.dumps(item) # 或者是 item 本身就是 eval_data

        # 这里假设 item 就是 evaluation json 对象 (根据你之前的代码)
        eval_data = item 

        for metric in metrics:
            winner = eval_data.get(metric, {}).get("Winner", "")
            
            if "Answer 1" in winner:
                real_winner = "Baseline" if is_swapped else "MyModel"
                counts[metric][real_winner] += 1
            elif "Answer 2" in winner:
                real_winner = "MyModel" if is_swapped else "Baseline"
                counts[metric][real_winner] += 1
            else:
                counts[metric]["Tie"] += 1

    # 打印结果
    print("\n====== Corrected Accuracy ======")
    for metric in metrics:
        w1 = counts[metric]["MyModel"]
        w2 = counts[metric]["Baseline"]
        tie = counts[metric]["Tie"]
        print(f"--- {metric} ---")
        print(f"My Model (RGCN): {w1/total_samples:.2%} ({w1})")
        print(f"Baseline (HiRAG): {w2/total_samples:.2%} ({w2})")
        print(f"Tie:              {tie/total_samples:.2%} ({tie})")
        
def fetch_eval_result_openai_batch(batch_id, output_file):
    """
    Fetch evaluation result from OpenAI API.
    """ 
    client = OpenAI()
    batch_content = client.batches.retrieve(batch_id)
    print(batch_content.status)
    output_file_id = batch_content.output_file_id
    file_content = client.files.content(output_file_id)
    with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'wb') as file:
        file.write(file_content.content)

    result = []
    with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    
    result_0 = json.loads('\n'.join(result[0]['response']['body']['choices'][0]['message']['content'].strip().split('\n')[1:-1]))

    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")

    result_list = []
    for item in result:
        result_list.append(json.loads('\n'.join(item['response']['body']['choices'][0]['message']['content'].strip().split('\n')[1:-1])))
    
    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {float(comprehensiveness_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(comprehensiveness_winner_ans2 / len(result_list))}")
    print("Empowerment:")
    print(f"Answer 1: {float(empowerment_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(empowerment_winner_ans2 / len(result_list))}")
    print("Diversity:")
    print(f"Answer 1: {float(diversity_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(diversity_winner_ans2 / len(result_list))}")
    print("Overall:")
    print(f"Answer 1: {float(overall_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(overall_winner_ans2 / len(result_list))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_file", type=str, default=_eval_path("datasets", "mix", "queries", "mix_query.jsonl"))
    parser.add_argument("-r1", "--result1_file", type=str, default=_eval_path("datasets", "mix", "answers", "mix_hirag_plugrag_rgcn_path.jsonl"))
    parser.add_argument("-r2", "--result2_file", type=str, default=_eval_path("datasets", "mix", "answers", "mix_hirag_baseline.jsonl"))
    parser.add_argument("-o", "--output_file", type=str, default=_eval_path("datasets", "mix", "results", "mix_hirag_rgcn_vs_baseline_result_openai.jsonl"))
    parser.add_argument("-m", "--mode", type=str, default="result", help="request or result")
    parser.add_argument("-api", "--api", type=str, default="openai", help="openai or deepseek or glm")
    parser.add_argument("-b", "--batch_id", type=str, default="")
    args = parser.parse_args()


    if args.mode == "request":
        if args.api == "openai":
            batch_id = eval_oq_openai(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "openai_batch":
            batch_id = eval_oq_openai_batch(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "deepseek":
            batch_id = eval_oq_deepseek(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "glm":
            batch_id = eval_oq_glm(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
    elif args.mode == "result":
        if args.api == "openai_batch":
            fetch_eval_result_openai_batch(batch_id=args.batch_id, output_file=args.output_file)
        elif args.api == "openai":  
            fetch_eval_result_openai(output_file=args.output_file)
        elif args.api == "deepseek":
            fetch_eval_result_deepseek(output_file=args.output_file)
        elif args.api == "glm":
            fetch_eval_result_glm(output_file=args.output_file)
