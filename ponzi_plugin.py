# ponzi_plugin.py
import json
import openai
from slither.detectors.abstract_detector import AbstractDetector, DetectorClassification
from PonziDetector import detect_fund_flow, detect_profit_logic, detect_referral_mechanism, detect_withdrawal_control, detect_camouflage
from openai import OpenAI

PROGRESS_FILE = "repaired_indices.txt"
API_KEY = "sk-bd24cc0b976e47b9827cbc30beb410e0"
API_BASE = "https://api.deepseek.com"
def generate_llm_explanations(contract_code: str) -> dict:
    system_prompt = (
        "You are a Solidity smart contract security expert. Your task is to analyze the given smart contract from five critical aspects, "
        "and for each aspect, return:\n"
        "1. A qualitative binary judgment (`yes` or `no`) indicating whether this feature exists in the contract.\n"
        "2. A concise explanation justifying your judgment based on contract structure, logic, or naming.\n\n"
        "You must output the result in structured JSON format. The JSON must contain exactly two top-level keys: `qualitative` and `explanations`, "
        "and both must include the same five keys: fund_flow, profit_logic, referral_mechanism, withdrawal_control, and camouflage.\n\n"
        "Only return valid JSON. Do not add any explanation, comments, or text outside the JSON."
    )

    task_prompt = (
        f"Contract code:\n{contract_code}\n\n"
        "Please analyze the contract based on the five aspects listed below. For each aspect, return both a `yes` or `no` answer and a supporting explanation.\n"
        "Follow this JSON structure exactly:\n"
        "{\n"
        "  \"qualitative\": {\n"
        "    \"fund_flow\": \"yes or no\",\n"
        "    \"profit_logic\": \"yes or no\",\n"
        "    \"referral_mechanism\": \"yes or no\",\n"
        "    \"withdrawal_control\": \"yes or no\",\n"
        "    \"camouflage\": \"yes or no\"\n"
        "  },\n"
        "  \"explanations\": {\n"
        "    \"fund_flow\": \"Explain in detail whether the investor’s payment is redistributed (e.g. via transfer/send/call), referencing specific functions or variables.\",\n"
        "    \"profit_logic\": \"Explain whether profit is calculated via fixed multipliers or based on new participants. Mention the formula or logic.\",\n"
        "    \"referral_mechanism\": \"Explain whether the contract uses a referral structure (e.g. mapping(address => address), inviter, etc.), and how rewards are distributed.\",\n"
        "    \"withdrawal_control\": \"Explain whether there are withdrawal restrictions (e.g. time-locks, cooldowns, block.timestamp, require statements, etc.).\",\n"
        "    \"camouflage\": \"Explain whether misleading naming is used (e.g. naming functions 'stake', 'mine', or 'game' when they actually transfer funds).\"\n"
        "  }\n"
        "}"
    )

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ],
            stream=False
        )

        content = response.choices[0].message.content

        # 自动修正模型输出的非标准 JSON
        try:
            explanations = json.loads(content)
        except json.JSONDecodeError:
            # 简单尝试补全缺失字段
            explanations = {
                "fund_flow": "[ERROR]",
                "profit_logic": "[ERROR]",
                "referral_mechanism": "[ERROR]",
                "withdrawal_control": "[ERROR]",
                "camouflage": "[ERROR]"
            }
            try:
                partial = json.loads(content + "}")
                for key in explanations:
                    if key in partial:
                        explanations[key] = partial[key]
            except:
                pass

    except Exception as e:
        explanations = {
            "fund_flow": f"[ERROR] {e}",
            "profit_logic": f"[ERROR] {e}",
            "referral_mechanism": f"[ERROR] {e}",
            "withdrawal_control": f"[ERROR] {e}",
            "camouflage": f"[ERROR] {e}"
        }

    return explanations

def regenerate_with_static_guidance(aspect: str, code: str, static_result: dict) -> str:
    system_prompt = (
        "You are an expert in Solidity smart contract analysis. You need to analyze the contract source code and combine the conclusions provided by static analysis."
        "Provide a natural language explanation for the following aspect: {aspect}. Your response must be consistent with the results of the static analysis."
    )

    guidance = (
        f"The results of the static analysis are as follows: Result : {static_result['result']} Explanation: {static_result['explanation']}"
        f"Please reinterpret the contract's behavior in the aspect of [{aspect}] based on this result. Do not contradict the static analysis conclusion."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt.format(aspect=aspect)},
                {"role": "user", "content": f" The source code of the contract is as follows: {code} {guidance}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def align_explanations(llm_explanations: dict, static_results: dict, contract_code: str) -> dict:
    corrected = {}
    for aspect, llm_exp in llm_explanations.items():
        static_result = static_results[aspect]["result"]
        static_explanation = static_results[aspect]["explanation"]

        hallucinated = (
            (not static_result and any(word in llm_exp for word in ["exists", "immediately", "redirect", "to", "referrer"])) or
            (static_result and any(word in llm_exp for word in ["not exist", "not set", "none", "does not contain"]))
        )

        if hallucinated:
            corrected[aspect] = regenerate_with_static_guidance(aspect, contract_code, static_results[aspect])
        else:
            corrected[aspect] = llm_exp

    return corrected


class PonziDetector(AbstractDetector):
    ARGUMENT = "ponzi-detector"
    HELP = "Detect Ponzi Scheme characteristics in smart contracts"
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.MEDIUM

    def _detect(self):
        results = []
        for contract in self.compilation_unit.contracts:
            source_code = contract.source_code


            llm_explanations = generate_llm_explanations(source_code)

            static_results = {
                "fund_flow": detect_fund_flow(contract),
                "profit_logic": detect_profit_logic(contract),
                "referral_mechanism": detect_referral_mechanism(contract),
                "withdrawal_control": detect_withdrawal_control(contract),
                "camouflage": detect_camouflage(contract)
            }

            for key in static_results:
                if isinstance(static_results[key], bool):
                    static_results[key] = {
                        "result": static_results[key],
                        "explanation": f"Static analysis result: {'Exists' if static_results[key] else 'Not detected'} for {key.replace('_',' ')}."
                    }

            aligned_explanations = align_explanations(llm_explanations, static_results, source_code)

            result = {
                "contract_name": contract.name,
                "source_code": source_code,
                "static_analysis": static_results,
                "explanations": aligned_explanations,
                "label": "Ponzi"
            }
            results.append(result)

        with open("ponzi_analysis_output.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return []
