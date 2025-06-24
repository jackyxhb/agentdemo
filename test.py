# pip install -qU langchain-openai langgraph python-dotenv to call the model
import os
from dotenv import load_dotenv
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

# Create Grok model
grok_model = ChatOpenAI(
    model="grok-3-mini",
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

agent = create_react_agent(
    model=grok_model,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
print("Running agent to get weather in San Francisco...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Auckland"}]}
)

print("\nAgent response:")
for message in result["messages"]:
    if hasattr(message, 'content'):
        print(f"{message.__class__.__name__}: {message.content}")
    else:
        print(f"{message.__class__.__name__}: {message}")