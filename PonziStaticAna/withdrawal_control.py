from slither.slithir.operations import Condition
from slither.core.declarations import Contract

def detect_withdrawal_control(contract: Contract) -> dict:
    for function in contract.functions:
        if "withdraw" in function.name.lower():
            for ir in function.all_slithir_operations():
                if isinstance(ir, Condition):
                    expr = str(ir)
                    if "block.timestamp" in expr or "now" in expr:
                        return {
                            "result": True,
                            "explanation": f"Withdrawal condition based on time detected in function {function.name}: {expr}"
                        }
    return {"result": False, "explanation": "No time-based withdrawal control logic detected."}
