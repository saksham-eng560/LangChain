from langchain.agents import create_agent
from langchain_ollama import ChatOllama


agent = create_agent(
    model = ChatOllama(model = "qwen3.5:9b"),
    system_prompt = "You are a helpful assistant."
)

response = agent.invoke({"messages": [{"role": "user", "content": "Hello, how are you?"}]})
print(response["messages"][-1].content)