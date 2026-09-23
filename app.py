import os
import uvicorn
import requests
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Tools ---
@tool
def get_destination_weather(city: str) -> str:
    """Get weather and temperature for a destination city."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1}
    geo_resp = requests.get(geo_url, params=geo_params).json()
    if "results" not in geo_resp:
        return f"Could not find destination: {city}"

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }
    weather = requests.get(weather_url, params=weather_params).json()["current"]
    return json.dumps({"resolved_city": geo_resp["results"][0]["name"], "temperature_celsius": weather["temperature_2m"]})

@tool
def search_flights_and_trains(origin: str, destination: str) -> str:
    """Search travel transport routes between origin and destination."""
    return f"Available routes from {origin} to {destination}: Direct Flight (2h 15m, ~₹4,500) and Express Train (12h, ~₹1,800)."

@tool
def search_hotels(city: str, budget_tier: str) -> str:
    """Find hotels in a city based on budget tier (budget, mid-range, luxury)."""
    return f"Top recommended {budget_tier} stays in {city}: Central Park Hotel, Grand Plaza."

tools = [get_destination_weather, search_flights_and_trains, search_hotels]

# --- 2. Initialize Model & Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to real-time travel planning, flights, hotels, and destination weather. "
        "For any other topic, you must say exactly: "
        "'I am not authorized to answer questions outside of travel planning, accommodation, and destination conditions.'"
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

# --- 3. FastAPI App ---
app = FastAPI(title="Real-Time Travel Planner Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
