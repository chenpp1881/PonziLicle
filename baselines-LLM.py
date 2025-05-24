import os
import json
import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# DeepSeek
# API_KEY = "sk-bd24cc0b976e47b9827cbc30beb410e0"
# API_BASE = "https://api.deepseek.com"
# PROGRESS_FILE = "repaired_indices.txt"
# MODEL = "deepseek-chat"

# yunwuAI
API_KEY = 'sk-GGi8JFWuUjwtUUGMi7N5orP9MmLXpFAJzFUpLcRM77JexsPh'
API_BASE = "https://yunwu.ai/v1"
PROGRESS_FILE = "repaired_indices_GPT.txt"
MODEL = 'gpt-4o'

openai.api_key = os.getenv(API_KEY)  # 替换为你的 key 或 set 环境变量


def extract_json_from_text(text):
    """
    尝试从 LLM 返回中提取合法 JSON 子串
    """
    try:
        # 匹配最外层的 {...} JSON 块（最常见场景）
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return None


# === Step 1: 加载 JSONL 数据 ===
def load_contracts_from_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for entry in raw_data:
            data.append({"label": entry.get("label", None), "code": entry.get("code", None), })
    return pd.DataFrame(data)


# === Step 2: 构造 Prompt 并调用 LLM ===
def ask_llm_is_ponzi(code, index=None):
    prompt = f"""You are a smart contract security expert.

Please analyze the following Solidity smart contract and determine whether it is a Ponzi scheme. 
Answer in JSON format: {{"is_ponzi": true/false, "reason": "..."}}

Smart contract:
{code}
"""
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content
        result = extract_json_from_text(result)

        return {
            "index": index,
            "llm_pred": int(result.get("is_ponzi", False)),
            "llm_reason": result.get("reason", "")
        }
    except Exception as e:
        print('error!~')
        return {
            "index": index,
            "llm_pred": -1,
            "llm_reason": str(e)
        }


# === Step 3: 并发运行多个 LLM 请求 ===
def evaluate_llm_multithread(df, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(ask_llm_is_ponzi, row["code"], idx): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results).set_index("index").sort_index()
    df = df.copy()
    df["llm_pred"] = results_df["llm_pred"]
    df["llm_reason"] = results_df["llm_reason"]

    report = classification_report(df["label"], df["llm_pred"], output_dict=True)
    return df, report


def read_data(path):
    df = pd.read_csv(path)
    classification_report(df["label"], df["llm_pred"], output_dict=True)


if __name__ == '__main__':
    df = load_contracts_from_jsonl("llm_explanations.json")
    df_result, metrics = evaluate_llm_multithread(df, max_workers=20)

    # 打印指标
    print(pd.DataFrame(metrics).T)

    # 保存结果
    df_result.to_csv("llm_ponzi_results_gpt.csv", index=False)

    # path = r'llm_ponzi_results.csv'
    # read_data(path)
