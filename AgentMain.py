# pip install -qU langchain-openai langgraph python-dotenv to call the model
import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


class WeatherResponse(BaseModel):
    """Response format for weather queries."""
    conditions: str


class AgentMain:
    """Main agent class for weather queries using xAI's Grok API."""
    
    def __init__(self, api_key: str = None, model_name: str = "grok-3-mini"):
        """
        Initialize the AgentMain class.
        
        Args:
            api_key (str, optional): xAI API key. If not provided, will load from environment.
            model_name (str): Name of the model to use. Defaults to "grok-3-mini".
        """
        self.model_name = model_name
        self.api_key = api_key
        self.logger = self._setup_logging()
        self.checkpointer = InMemorySaver()
        self.agent = None
        
        # Load environment variables
        load_dotenv()
        self.logger.info("✓ Environment variables loaded from .env file")
        
        # Validate and set API key
        self._validate_api_key()
        
        # Initialize the agent
        self._initialize_agent()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('agent.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _validate_api_key(self) -> None:
        """Validate that API key is available."""
        if not self.api_key:
            self.api_key = os.getenv("XAI_API_KEY")
        
        if not self.api_key:
            self.logger.error("XAI_API_KEY not found in environment variables or .env file.")
            self.logger.error("Please check your .env file or set the environment variable.")
            self.logger.error("You can get an API key from: https://console.x.ai/")
            raise ValueError("XAI_API_KEY is required")
    
    def get_weather(self, city: str) -> str:
        """Get weather for a given city."""
        self.logger.info(f"Getting weather for city: {city}")
        return f"It's always sunny in {city}!"
    
    def _create_prompt(self, state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
        """Create prompt for the agent."""
        user_name = config["configurable"].get("user_name", "User")
        system_msg = f"You are a helpful assistant. Address the user as {user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]
    
    def _initialize_agent(self) -> None:
        """Initialize the Grok model and agent."""
        self.logger.info("Initializing Grok model with xAI API")
        
        # Create Grok model
        grok_model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Create agent
        self.logger.info("Creating LangGraph agent with weather tools")
        self.agent = create_react_agent(
            model=grok_model,
            tools=[self.get_weather],
            prompt=self._create_prompt,
            checkpointer=self.checkpointer,
            response_format=WeatherResponse
        )
    
    def query_weather(self, city: str, thread_id: str = "1", user_name: str = "User") -> Dict[str, Any]:
        """
        Query weather for a specific city.
        
        Args:
            city (str): Name of the city to get weather for
            thread_id (str): Thread ID for conversation tracking
            user_name (str): Name of the user for personalized responses
        
        Returns:
            Dict[str, Any]: Agent response containing messages and structured response
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call _initialize_agent() first.")
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_name": user_name
            }
        }
        
        self.logger.info(f"Running agent to get weather in {city}...")
        
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": f"what is the weather in {city}"}]},
            config=config
        )
        
        self.logger.info("Agent execution completed successfully")
        return result
    
    def print_response(self, result: Dict[str, Any]) -> None:
        """Print the agent response in a formatted way."""
        print("\nAgent response:")
        for message in result["messages"]:
            if hasattr(message, 'content'):
                print(f"{message.__class__.__name__}: {message.content}")
                self.logger.debug(f"Message: {message.__class__.__name__} - {message.content}")
            else:
                print(f"{message.__class__.__name__}: {message}")
                self.logger.debug(f"Message: {message.__class__.__name__} - {message}")
    
    def run_demo(self, city: str = "Auckland") -> None:
        """Run a demo query for the specified city."""
        try:
            result = self.query_weather(city)
            self.print_response(result)
            self.logger.info("Demo execution completed")
        except Exception as e:
            self.logger.error(f"Error during demo execution: {e}")
            raise


def main():
    """Main function to run the agent demo."""
    try:
        # Create and run the agent
        agent = AgentMain()
        agent.run_demo("Auckland")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())