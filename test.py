# pip install -qU langchain-openai langgraph python-dotenv to call the model
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Check if API key is set (now it should be loaded from .env)
if not os.getenv("XAI_API_KEY"):
    print("XAI_API_KEY not found in environment variables or .env file.")
    print("Please check your .env file or set the environment variable.")
    print("You can get an API key from: https://console.x.ai/")
    exit(1)

print("✓ Environment variables loaded from .env file")

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]


# Create Grok model
grok_model = ChatOpenAI(
    model="grok-3-mini",
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

checkpointer = InMemorySaver()
class WeatherResponse(BaseModel):
    conditions: str
    
agent = create_react_agent(
    model=grok_model,
    tools=[get_weather],
    prompt=prompt,
    checkpointer=checkpointer,
    response_format=WeatherResponse 
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
print("Running agent to get weather in Auckland...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Auckland"}]},
    config=config
)

print("\nAgent response:")
for message in result["messages"]:
    if hasattr(message, 'content'):
        print(f"{message.__class__.__name__}: {message.content}")
    else:
        print(f"{message.__class__.__name__}: {message}")

result["structured_response"]