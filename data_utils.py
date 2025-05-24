import logging
logger = logging.getLogger(__name__)
import json
import random
import torch
from torch.utils.data import Dataset
import re

def clean_solidity_code(source_code: str) -> str:
    """
    清洗 Solidity 源代码：去除注释、缩进、多余空行等模型学习无关结构。

    参数:
        source_code (str): 原始 Solidity 源代码

    返回:
        str: 清洗后的代码
    """
    # 1. 移除多行注释（包括 /** */ 和 /* */）
    source_code = re.sub(r'/\*[\s\S]*?\*/', '', source_code)

    # 2. 移除单行注释（//...）
    source_code = re.sub(r'//.*', '', source_code)

    # 3. 拆分为行，逐行处理
    lines = source_code.splitlines()
    cleaned_lines = []
    for line in lines:
        # 去除缩进和末尾空格
        line = line.strip()
        # 跳过空行或只有分号的行
        if not line or line == ';':
            continue
        cleaned_lines.append(line)

    # 4. 使用统一换行连接
    return '\n'.join(cleaned_lines)


class CodeDataset(Dataset):
    def __init__(self, datas, tokenizer, max_length):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.datas)

    def load_tokens(self, text):
        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in text_tokens.items()}

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx])

        code = data['code']
        label = int(data['label'])  # already 0/1

        # 5个视角
        keys = ['fund_flow', 'profit_logic', 'referral_mechanism', 'withdrawal_control', 'camouflage']

        explanations = data['explanations'].get("explanations", {})
        qualitative = data['explanations'].get("qualitative", {})

        # Tokenize 每个解释文本
        explanation_toks = [self.load_tokens(explanations.get(key, "")) for key in keys]

        # 定性结果转为二进制 1/0
        qualitative_binary = [
            1 if qualitative.get(key, "no").lower() == "yes" else 0
            for key in keys
        ]

        # Tokenize 合约代码
        # code_tok = self.load_tokens(clean_solidity_code(code))
        code_tok = self.load_tokens(code)

        return code_tok, explanation_toks, torch.tensor(qualitative_binary, dtype=torch.long), torch.tensor(label,
                                                                                                            dtype=torch.long), {
            'code': code, 'explanations': explanations, 'qualitative': qualitative}

# class CodeDataset(Dataset):
#     def __init__(self, datas, tokenizer, max_length, is_train=False):
#         self.tokenizer = tokenizer
#         self.max_len = max_length
#         self.is_train = is_train
#
#         keys = ['fund_flow', 'profit_logic', 'referral_mechanism', 'withdrawal_control', 'camouflage']
#         filtered_datas = []
#
#         for item in datas:
#             data = json.loads(item)
#             qualitative = data.get("explanations", {}).get("qualitative", {})
#             qualitative_binary = [
#                 1 if qualitative.get(key, "no").lower() == "yes" else 0
#                 for key in keys
#             ]
#
#             if not (is_train and sum(qualitative_binary) == 0):
#                 filtered_datas.append(item)
#
#         self.datas = filtered_datas
#
#     def __len__(self):
#         return len(self.datas)
#
#     def load_tokens(self, text):
#         text_tokens = self.tokenizer(
#             text,
#             padding='max_length',
#             max_length=self.max_len,
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {key: val.squeeze(0) for key, val in text_tokens.items()}
#
#     def __getitem__(self, idx):
#         data = json.loads(self.datas[idx])
#
#         code = data['code']
#         label = int(data['label'])
#
#         keys = ['fund_flow', 'profit_logic', 'referral_mechanism', 'withdrawal_control', 'camouflage']
#
#         explanations = data['explanations'].get("explanations", {})
#         qualitative = data['explanations'].get("qualitative", {})
#
#         explanation_toks = [self.load_tokens(explanations.get(key, "")) for key in keys]
#
#         qualitative_binary = [
#             1 if qualitative.get(key, "no").lower() == "yes" else 0
#             for key in keys
#         ]
#
#         code_tok = self.load_tokens(code)
#
#         return code_tok, explanation_toks, torch.tensor(qualitative_binary, dtype=torch.long), torch.tensor(label, dtype=torch.long), {
#             'code': code, 'explanations': explanations, 'qualitative': qualitative
#         }

def load_llm_explanation_dataset(json_path: str, train_ratio: float = 0.8, seed: int = 66):
    """
    读取 LLM 解释生成的 JSON 文件，并划分为训练集和测试集。

    返回值：
        train_data: list[str]  # 每个元素为一行 JSON 字符串
        test_data: list[str]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # raw_data 是一个 list，每个元素是 dict

    # 打乱
    random.seed(seed)
    random.shuffle(raw_data)

    # 转为 JSON string 列表
    data_lines = [json.dumps(entry, ensure_ascii=False) for entry in raw_data]

    split_idx = int(len(data_lines) * train_ratio)
    train_data = data_lines[:split_idx]
    test_data = data_lines[split_idx:]

    return train_data, test_data
