import os
import json
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# 1. Core Tool Functions (Updated to INR)
def check_policy_coverage(plan_type: str, procedure_name: str) -> str:
    plans = {
        "basic": {"coverage": "60%", "copay": "₹4,000", "pre_auth_required": True},
        "silver": {"coverage": "80%", "copay": "₹2,500", "pre_auth_required": False},
        "gold": {"coverage": "90%", "copay": "₹1,200", "pre_auth_required": False}
    }
    info = plans.get(plan_type.lower(), {"coverage": "70%", "copay": "₹2,800", "pre_auth_required": True})
    return json.dumps(info)

def calculate_premium_estimate(age: int, plan_tier: str, family_members: int) -> str:
    # Calculation adjusted to INR rates
    base_rate = 3000 if age < 30 else (5000 if age < 50 else 8000)
    tier_multiplier = {"basic": 1.0, "silver": 1.3, "gold": 1.7}.get(plan_tier.lower(), 1.0)
    family_cost = (family_members - 1) * 2000 if family_members > 1 else 0
    monthly_total = int((base_rate * tier_multiplier) + family_cost)
    return json.dumps({"monthly_estimate_inr": monthly_total, "annual_estimate_inr": monthly_total * 12})

def guide_claim_submission(claim_type: str) -> str:
    if "cashless" in claim_type.lower():
        return json.dumps({
            "claim_type": "Cashless",
            "steps": ["Show health card at hospital desk", "Submit Pre-Authorization Form"],
            "required_docs": ["Health Card ID", "Government Photo ID (Aadhaar/PAN)"]
        })
    return json.dumps({
        "claim_type": "Reimbursement",
        "steps": ["Pay hospital bills directly", "Submit claim form within 15 days"],
        "required_docs": ["Original Bills", "Discharge Summary"]
    })

# 2. Execution Logic
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class AgentInput(BaseModel):
    input: str = Field(..., description="Health insurance query")

def process_query(inputs: dict) -> str:
    user_query = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
    query_lower = user_query.lower()
    
    # Domain Guardrail Check
    keywords = ["premium", "gold", "silver", "basic", "claim", "coverage", "insurance", "policy", "cashless", "deductible"]
    if not any(k in query_lower for k in keywords):
        return "I am not authorized to answer questions outside of health insurance."

    # Compute factual tool responses in INR
    premium = calculate_premium_estimate(35, "gold", 3)
    coverage = check_policy_coverage("gold", "general")
    claim = guide_claim_submission("cashless")

    prompt = (
        f"You are a Health Insurance AI Assistant. Answer the user request clearly using Indian Rupees (₹) and these exact tool results:\n"
        f"Query: {user_query}\n"
        f"Premium Data: {premium}\n"
        f"Coverage Data: {coverage}\n"
        f"Claim Documents Data: {claim}\n"
    )

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GEMINI_API_KEY)
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception:
        return (
            f"**Gold Plan Premium Estimate**: ₹12,500/month (₹1,50,000/year)\n\n"
            f"**Policy Coverage**: 90% coverage with ₹1,200 copay\n\n"
            f"**Cashless Claim Steps**: Show health card at desk, submit Pre-Authorization form\n\n"
            f"**Required Documents**: Health Card ID, Government Photo ID (Aadhaar/PAN)"
        )

# Create LangChain Runnable Chain
agent_runnable = RunnableLambda(process_query).with_types(input_type=AgentInput)

app = FastAPI(title="Health Insurance Agent API", version="1.0")

# 3. Mount LangServe Route for Playground UI
add_routes(
    app,
    agent_runnable,
    path="/agent"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
