import pandas as pd
import re
import os
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# === 配置 ===
PROGRESS_FILE = "repaired_indices.txt"
API_KEY = "sk-bd24cc0b976e47b9827cbc30beb410e0"
API_BASE = "https://api.deepseek.com"

def extract_solidity_code(response_text: str) -> str:
    code_block = re.findall(r"```solidity(.*?)```", response_text, re.DOTALL)
    if code_block:
        return code_block[0].strip()
    fallback = re.findall(r"```(.*?)```", response_text, re.DOTALL)
    if fallback:
        return fallback[0].strip()
    return response_text.strip()


def repair_contract_with_llm(code: str) -> str | None:
    system_prompt = (
        "You are an expert Solidity engineer. Your task is to repair the following incomplete or outdated Solidity contract code.\n\n"
        "Your goal is to make minimal modifications required to make the code syntactically correct and compilable using Solidity version 0.8.0 or above.\n\n"
        "⚠️ Do NOT modify or remove any existing logic, variables, or function behavior.\n\n"
        "✅ You may:\n"
        "- Add or update the `pragma` version to `^0.8.0` or higher\n"
        "- Fix outdated syntax from earlier versions (e.g., 0.4.x or 0.5.x) to make it compatible with 0.8.0\n"
        "- Add missing braces, semicolons, or memory/location specifiers\n"
        "- Wrap top-level code in a contract if needed\n"
        "- Fix unmatched parentheses or brackets\n\n"
        "Please output only the corrected Solidity code, wrapped in a single markdown code block with 'solidity' as the language.\n"
        "Do NOT include any explanation or comments outside the code block."
    )

    user_prompt = f"Solidity code:\n\n{code}"

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )

        raw_output = response.choices[0].message.content
        fixed = extract_solidity_code(raw_output)

        if fixed.strip() == "" or "contract" not in fixed:
            return None
        return fixed

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        return None

def load_processed_indices() -> set:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def save_processed_index(idx: int):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{idx}\n")

def process_and_save(df, output_path: str, code_column: str = "code", max_workers: int = 10):
    processed = load_processed_indices()

    # 初始化 CSV 文件（首次）
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=df.columns)
            writer.writeheader()

    def task(idx):
        if idx in processed:
            return None  # 跳过已处理
        code = df.at[idx, code_column]
        repaired_code = repair_contract_with_llm(code)
        if repaired_code:
            row = df.iloc[idx].copy()
            row[code_column] = repaired_code
            return idx, row.to_dict()
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, idx): idx for idx in df.index}

        with open(output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=df.columns)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing with resume"):
                result = future.result()
                if result:
                    idx, row_dict = result
                    writer.writerow(row_dict)
                    save_processed_index(idx)



if __name__ == "__main__":
    input_path = "Dataset.csv"
    output_path = "contracts_repaired.csv"
    df = pd.read_csv(input_path)
    process_and_save(df, output_path, code_column="code", max_workers=20)