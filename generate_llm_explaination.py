import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import re

# === DeepSeek API 配置 ===
# API_KEY = "sk-bd24cc0b976e47b9827cbc30beb410e0"
# API_BASE = "https://api.deepseek.com"
# PROGRESS_FILE = "repaired_indices.txt"
# MODEL = "deepseek-chat"
# output_json = 'llm_explanations_deepseek.json'

API_KEY = 'sk-GGi8JFWuUjwtUUGMi7N5orP9MmLXpFAJzFUpLcRM77JexsPh'
API_BASE = "https://yunwu.ai/v1"
PROGRESS_FILE = "repaired_indices_GPT.txt"
MODEL = 'gpt-4o-2024-11-20'
output_json = 'llm_explanations_GPT.json'

def extract_json_block(text: str) -> str:
    """
    从 LLM 返回中提取合法 JSON 字符串（剥离 markdown 代码块 ```json ...```）
    """
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # fallback


# === DeepSeek 解释生成函数 ===
def generate_llm_explanations_deepseek(contract_code: str) -> dict | None:
    system_prompt = (
        "You are a Solidity smart contract security expert. Your task is to analyze the given smart contract from five critical aspects, "
        "and for each aspect, return:\n"
        "1. A qualitative binary judgment (`yes` or `no`) indicating whether this feature exists in the contract.\n"
        "2. A detailed explanation justifying your judgment based on contract structure, logic, or naming.\n\n"
        "You must output the result in structured JSON format. The JSON must contain exactly two top-level keys: `qualitative` and `explanations`, "
        "and both must include the same five keys: fund_flow, profit_logic, referral_mechanism, withdrawal_control, and camouflage.\n\n"
        "Only return valid JSON. Do not add any explanation, comments, or text outside the JSON."
    )

    task_prompt = (
        f"Contract code:\n{contract_code}\n\n"
        "Please analyze the contract based on the five aspects listed below. For each aspect, return both a `yes` or `no` answer and a supporting explanation.\n"
        "Follow this JSON structure exactly:\n"
        "{\n"
        "  \"qualitative\": {\n"
        "    \"fund_flow\": \"yes or no\",\n"
        "    \"profit_logic\": \"yes or no\",\n"
        "    \"referral_mechanism\": \"yes or no\",\n"
        "    \"withdrawal_control\": \"yes or no\",\n"
        "    \"camouflage\": \"yes or no\"\n"
        "  },\n"
        "  \"explanations\": {\n"
        "    \"fund_flow\": \"Explain in detail whether the investor’s payment is redistributed (e.g. via transfer/send/call), referencing specific functions or variables.\",\n"
        "    \"profit_logic\": \"Explain whether profit is calculated via fixed multipliers or based on new participants. Mention the formula or logic.\",\n"
        "    \"referral_mechanism\": \"Explain whether the contract uses a referral structure (e.g. mapping(address => address), inviter, etc.), and how rewards are distributed.\",\n"
        "    \"withdrawal_control\": \"Explain whether there are withdrawal restrictions (e.g. time-locks, cooldowns, block.timestamp, require statements, etc.).\",\n"
        "    \"camouflage\": \"Explain whether misleading naming is used (e.g. naming functions 'stake', 'mine', or 'game' when they actually transfer funds).\"\n"
        "  }\n"
        "}"
    )

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ],
            stream=False
        )

        content = response.choices[0].message.content
        content = extract_json_block(content)
        try:
            parsed = json.loads(content)
            if "qualitative" in parsed and "explanations" in parsed:
                return parsed
        except json.JSONDecodeError:
            print('json process error!~')
            return None

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        return None


# === 批量处理函数 ===
def generate_explanations_from_csv(csv_path: str, output_json: str, error_index_file: str, code_column="code",
                                   label_column="label", max_workers=10):
    df = pd.read_csv(csv_path)
    results = []
    failed_ids = []

    def task(idx, code, label):
        explanations = generate_llm_explanations_deepseek(code)
        return {
            "index": idx,
            "code": code,
            "label": label,
            "explanations": explanations
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(task, idx, row[code_column], row[label_column]): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Explanations"):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Task failed at index {idx}: {e}")
                failed_ids.append(idx)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(error_index_file, "w") as f:
        for idx in failed_ids:
            f.write(str(idx) + "\n")

    print(f"[INFO] {len(results)} explanations saved to {output_json}")
    print(f"[INFO] {len(failed_ids)} failed indices saved to {error_index_file}")


def is_explanation_empty(explanations_dict):
    # 如果 explanations 是 dict 类型，检查每个字段是否为空字符串
    if isinstance(explanations_dict, dict):
        return True
    return False  # 如果 explanations 不是 dict，视为无效


def filter_non_empty_explanations(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered = []
    for item in tqdm(data, desc="check empty"):
        explanations = item.get("explanations", {})
        if explanations is None:
            continue
        filtered.append(item)

    print(f"原始数据量: {len(data)}，过滤后剩余: {len(filtered)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)


# === 主入口 ===
if __name__ == "__main__":
    generate_explanations_from_csv(
        csv_path="Dataset.csv",
        output_json=output_json,
        error_index_file=PROGRESS_FILE,
        max_workers=20
    )

    # 用法示例
    filter_non_empty_explanations(output_json, output_json)
