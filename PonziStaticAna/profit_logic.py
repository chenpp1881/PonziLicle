from slither.slithir.operations import Binary, Operation
from slither.core.declarations import Contract

def detect_profit_logic(contract: Contract) -> dict:
    explanation = ""
    for function in contract.functions:
        for ir in function.all_slithir_operations():
            if isinstance(ir, Binary):
                if ir.type == "*" and (
                    any("msg.value" in str(v) for v in ir.read) or
                    ("invest" in function.name.lower() and "balance" in str(ir))
                ):
                    explanation = (
                        f"Function {function.name} contains multiplication logic suggesting fixed profit calculation: {str(ir)}"
                    )
                    return {"result": True, "explanation": explanation}
            if isinstance(ir, Operation) and "calculateProfit" in str(ir):
                explanation = f"Function {function.name} calls a profit calculation method: {str(ir)}"
                return {"result": True, "explanation": explanation}
    return {"result": False, "explanation": "No static profit logic or multiplier detected."}
