from slither.core.declarations import Contract

def detect_referral_mechanism(contract: Contract) -> dict:
    for var in contract.state_variables:
        if var.type and "mapping" in str(var.type):
            if any(keyword in var.name.lower() for keyword in ["referrer", "upline", "downline", "inviter"]):
                return {"result": True, "explanation": f"Found referral mapping variable: {var.name}"}
    for func in contract.functions:
        if any(keyword in func.name.lower() for keyword in ["referral", "invite", "register"]):
            return {"result": True, "explanation": f"Found referral-related function: {func.name}"}
    return {"result": False, "explanation": "No referral-related variables or functions found."}
