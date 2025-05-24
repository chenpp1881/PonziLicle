from slither.core.declarations import Contract

def detect_camouflage(contract: Contract) -> dict:
    misleading_terms = ["stake", "mine", "game", "reward", "pool"]
    suspicious_terms = ["transfer", "pay", "send", "invest"]

    for func in contract.functions:
        if any(term in func.name.lower() for term in misleading_terms):
            for ir in func.all_slithir_operations():
                if any(term in str(ir).lower() for term in suspicious_terms):
                    return {
                        "result": True,
                        "explanation": f"Misleading function name '{func.name}' performs suspicious logic: {str(ir)}"
                    }
    return {"result": False, "explanation": "No misleading naming paired with suspicious logic detected."}
