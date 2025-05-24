from slither.slithir.operations import Transfer, Send, Call
from slither.slithir.variables import Constant
from slither.core.declarations import Contract

def detect_fund_flow(contract: Contract) -> dict:
    explanation = ""
    for function in contract.functions:
        if function.payable:
            for ir in function.all_slithir_operations():
                if isinstance(ir, (Transfer, Send, Call)):
                    if ir.destination and not isinstance(ir.destination, Constant):
                        explanation = (
                            f"Function {function.name} transfers funds to a dynamic address via {ir.__class__.__name__}. "
                            "This indicates potential redirection of user deposits."
                        )
                        return {"result": True, "explanation": explanation}
    return {"result": False, "explanation": "No fund forwarding behavior detected in payable functions."}
