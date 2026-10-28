from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
import requests

@tool
def get_weather(city : str):
    """Return the weather through the API calling"""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    current = response.json()["current_condition"][0]
    return {
        "city": city,
        "temp_C": current["temp_C"],
        "feels_like_C": current["FeelsLikeC"],
        "humidity": current["humidity"],
        "description": current["weatherDesc"][0]["value"],
    }

model = ChatOllama(model = "qwen3.5:9b")

agent = create_agent(
    model = model,
    tools = [get_weather],
    system_prompt = "You are a helpful assistant. Use the get_weather tool to answer weather questions.",
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Delhi?"}]})
    print(result["messages"][-1].content)
