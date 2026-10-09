import os
import json
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class AgentInput(BaseModel):
    input: str = Field(..., description="Travel query or destination")

# Main Travel Agent Logic
def process_travel_query(inputs: dict) -> str:
    user_query = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
    query_lower = user_query.lower()
    
    # Domain Guardrail Check
    keywords = ["travel", "trip", "itinerary", "flight", "hotel", "vizag", "visakhapatnam", "hyderabad", "guide", "destination", "pdf", "voucher"]
    if not any(k in query_lower for k in keywords):
        return "I am not authorized to answer questions outside of travel planning and logistics."

    # Formulate structured prompt for Gemini
    prompt = f"""
    You are an expert Travel Planner AI Assistant.
    Provide a detailed, clear, and well-formatted travel guide and itinerary for the following request:
    "{user_query}"
    
    Include:
    1. Best time to visit
    2. Mode of transport/travel options
    3. Key sightseeing attractions
    4. Suggested day-by-day itinerary
    5. Local food & cuisine recommendations
    
    Format the response clearly using clean Markdown text with bold headings and bullet points. Do NOT output raw JSON.
    """

    try:
        # Updated model string to gemini-3.8-flash
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GEMINI_API_KEY, temperature=0.3)
        response = llm.invoke(prompt)
        return str(response.content)
    except Exception as e:
        return f"Error generating travel guide: {str(e)}"

# Create Runnable Wrapper for LangServe
travel_runnable = RunnableLambda(process_travel_query).with_types(input_type=AgentInput)

app = FastAPI(title="Travel Planner Agent API", version="1.0")

# Mount LangServe Route
add_routes(
    app,
    travel_runnable,
    path="/agent"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
