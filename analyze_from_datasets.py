import pandas as pd
import json
import traceback
from ponzi_plugin import generate_llm_explanations, align_explanations
from PonziDetector import (
    detect_fund_flow,
    detect_profit_logic,
    detect_referral_mechanism,
    detect_withdrawal_control,
    detect_camouflage
)
from slither import Slither
from tempfile import NamedTemporaryFile


def mock_contract_for_analysis(code: str):
    """
    创建临时合约文件并返回 Slither 合约对象。
    """
    with NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        slither = Slither(tmp.name)
        return slither.contracts[0]


def analyze_contracts_from_csv(csv_path: str, code_column: str = "code", label_column: str = "label", output_path: str = "aligned_ponzi_results.json"):
    """
    {
  "code": "...",
  "label": "...",
  "static_analysis": {
    "fund_flow": {
      "result": true,
      "explanation": "Static analysis result: Exists for fund flow."
    },
    ...
  },
  "explanations": {
    ...
  },
  "explanation_source": "aligned"  // 或 llm_only
    """


    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        code = row[code_column]
        label = row[label_column]

        try:
            # Step 1: LLM 解释（始终运行）
            llm_explanations = generate_llm_explanations(code)

            static_results = None
            aligned_explanations = None
            explanation_used = "llm_only"

            # Step 2: 尝试静态分析
            try:
                contract = mock_contract_for_analysis(code)
                static_results = {
                    "fund_flow": detect_fund_flow(contract),
                    "profit_logic": detect_profit_logic(contract),
                    "referral_mechanism": detect_referral_mechanism(contract),
                    "withdrawal_control": detect_withdrawal_control(contract),
                    "camouflage": detect_camouflage(contract)
                }

                # 包装静态分析结果
                for key in static_results:
                    if isinstance(static_results[key], bool):
                        static_results[key] = {
                            "result": static_results[key],
                            "explanation": f"Static analysis result: {'Exists' if static_results[key] else 'Not detected'} for {key.replace('_',' ')}."
                        }

                # Step 3: 对齐 LLM 和 静态结果
                aligned_explanations = align_explanations(llm_explanations, static_results, code)
                explanation_used = "aligned"

            except Exception as static_error:
                print(f"[WARNING] Static analysis failed at contract {idx}: {static_error}")
                traceback.print_exc()

            # Step 4: 合并结果
            result_item = {
                "code": code,
                "label": label,
                "explanations": aligned_explanations if aligned_explanations else llm_explanations,
                "explanation_source": explanation_used
            }

            if static_results:
                result_item["static_analysis"] = static_results

            results.append(result_item)

        except Exception as e:
            print(f"[ERROR] Skipping contract {idx}: {e}")
            traceback.print_exc()
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Completed! Results saved to: {output_path}")

