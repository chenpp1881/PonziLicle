
import json
from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification
from slither.slithir.operations import Transfer, Send, Call, Binary, Operation, Condition
from slither.slithir.variables import Constant
from slither.core.declarations import Contract

class PonziDetector(AbstractDetector):
    ARGUMENT = "ponzi-detector"
    HELP = "Detect Ponzi Scheme characteristics in smart contracts"
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.MEDIUM

    def _detect(self):
        results = []
        for contract in self.compilation_unit.contracts:
            fund_flow = detect_fund_flow(contract)
            profit_logic = detect_profit_logic(contract)
            referral = detect_referral_mechanism(contract)
            withdraw_control = detect_withdrawal_control(contract)
            camouflage = detect_camouflage(contract)

            if any([fund_flow, profit_logic, referral, withdraw_control, camouflage]):
                result = {
                    "contract_name": contract.name,
                    "static_analysis": {
                        "fund_flow": fund_flow,
                        "profit_logic": profit_logic,
                        "referral_mechanism": referral,
                        "withdrawal_control": withdraw_control,
                        "camouflage": camouflage
                    },
                    "source_code": contract.source_code,
                    "label": "Ponzi"  # 如果你有标签数据可替换为自动标注机制
                }
                results.append(result)

        # 导出为 JSON 文件（你可以改为其他存储方式）
        with open("ponzi_analysis_output.json", "w") as f:
            json.dump(results, f, indent=2)

        return []  # 返回空 Slither 报告，分析结果已导出


def detect_fund_flow(contract: Contract) -> bool:
    for function in contract.functions:
        if function.payable:
            for ir in function.all_slithir_operations():
                if isinstance(ir, (Transfer, Send, Call)):
                    if ir.destination and not isinstance(ir.destination, Constant):
                        return True
    return False

# profit_logic.py
# 检测收益是否基于静态比例或与后续用户有关
def detect_profit_logic(contract: Contract) -> bool:
    for function in contract.functions:
        for ir in function.all_slithir_operations():
            if isinstance(ir, Binary):
                if ir.type in ("*") and (
                    ("msg.value" in ir.read and any(v.name in str(ir) for v in ir.read)) or
                    ("invest" in function.name.lower() and "balance" in str(ir))
                ):
                    return True
            if isinstance(ir, Operation) and "calculateProfit" in str(ir):
                return True
    return False

# referral_mechanism.py
# 检测是否存在推荐机制或层级关系
def detect_referral_mechanism(contract: Contract) -> bool:
    for var in contract.state_variables:
        if var.type and "mapping" in str(var.type):
            if any(keyword in var.name.lower() for keyword in ["referrer", "upline", "downline", "inviter"]):
                return True
    for func in contract.functions:
        if any(keyword in func.name.lower() for keyword in ["referral", "invite", "register"]):
            return True
    return False

# withdrawal_control.py
# 检测提现限制（锁仓时间、时间检查等）
def detect_withdrawal_control(contract: Contract) -> bool:
    for function in contract.functions:
        if "withdraw" in function.name.lower():
            for ir in function.all_slithir_operations():
                if isinstance(ir, Condition):
                    expr = str(ir)
                    if "block.timestamp" in expr or "now" in expr:
                        return True
    return False

# camouflage_detector.py
# 检测是否存在掩盖真实用途的命名
def detect_camouflage(contract: Contract) -> bool:
    misleading_terms = ["stake", "mine", "game", "reward", "pool"]
    suspicious_terms = ["transfer", "pay", "send", "invest"]

    for func in contract.functions:
        if any(term in func.name.lower() for term in misleading_terms):
            for ir in func.all_slithir_operations():
                if any(term in str(ir).lower() for term in suspicious_terms):
                    return True
    return False
