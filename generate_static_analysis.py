# generate_static_analysis.py
import pandas as pd
import json
import traceback
from PonziDetector import (
    detect_fund_flow,
    detect_profit_logic,
    detect_referral_mechanism,
    detect_withdrawal_control,
    detect_camouflage
)
from slither import Slither
from tempfile import NamedTemporaryFile
import os
os.environ["SOLC_VERSION"] = "0.8.19"

def mock_contract_for_analysis(code: str):
    with NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
        tmp.write(code)
        tmp.flush()
        slither = Slither(tmp.name)
        return slither.contracts[0]

def generate_static_json(csv_path: str, output_path: str, code_column: str = "code", label_column: str = "label"):
    df = pd.read_csv(csv_path)
    static_records = []

    for idx, row in df.iterrows():
        code = row[code_column]
        label = row[label_column]

        try:
            contract = mock_contract_for_analysis(code)
            static_result = {
                "fund_flow": detect_fund_flow(contract),
                "profit_logic": detect_profit_logic(contract),
                "referral_mechanism": detect_referral_mechanism(contract),
                "withdrawal_control": detect_withdrawal_control(contract),
                "camouflage": detect_camouflage(contract)
            }

            static_records.append({
                "id": idx,
                "label": label,
                "static_analysis": static_result
            })

        except Exception as e:
            print(f"[ERROR] Contract {idx} failed: {e}")
            traceback.print_exc()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(static_records, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Static analysis results saved to: {output_path}")


if __name__ == "__main__":
    generate_static_json("contracts_repaired.csv", "static_results.json")
