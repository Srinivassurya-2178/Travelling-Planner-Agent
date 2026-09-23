import os
import uvicorn
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

@tool
def get_destination_guide(city: str) -> str:
    """Get tourist highlights and peak season for a destination city."""
    return f"Destination guide for {city}: Popular attractions, optimal weather months, and transport guides."

@tool
def check_flight_status(flight_number: str) -> str:
    """Check status of an ongoing flight using Flight Number."""
    return json.dumps({"flight_number": flight_number, "status": "On Time", "gate": "A12"})

@tool
def analyze_travel_pdf(file_path: str) -> str:
    """Extract and analyze details from a travel PDF document."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs[:10]])
        return f"Travel Document Content:\n\n{full_text[:4000]}"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

tools = [get_destination_guide, check_flight_status, analyze_travel_pdf]

# Explicitly grab the API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize ChatGoogleGenerativeAI with gemini-3.6-flash and explicit api_key
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=GEMINI_API_KEY
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to travel planning, destination guides, itinerary building, and flight tracking. "
        "For any other topic, you must say exactly: "
        "'I am not authorized to answer questions outside of travel planning, destinations, and flight tracking.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your travel query")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)
    messages = agent_output.get("messages")
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break
    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))
    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

app = FastAPI(title="Travel Planner Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
