# pip install -qU langchain-xai python-dotenv
import os
import logging
from dotenv import load_dotenv
from langchain_xai import ChatXAI


class AgentMain:
    """Main agent class for news queries using xAI's Grok API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the AgentMain class.
        
        Args:
            api_key (str, optional): xAI API key. If not provided, will load from environment.
        """
        self.api_key = api_key
        self.logger = self._setup_logging()
        
        # Load environment variables
        load_dotenv()
        self.logger.info("✓ Environment variables loaded from .env file")
        
        # Validate and set API key
        self._validate_api_key()
    
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
    
    def run_chat(self) -> None:
        """Run a chat session with the agent."""
        self.logger.info("Running chat session with the agent...")
        
        try:
            llm = ChatXAI(
                model="grok-3-latest",
                api_key=self.api_key,
                search_parameters={
                    "mode": "auto",
                    "max_search_results": 3,
                    "from_date": "2025-06-23",
                    "to_date": "2025-06-24",
                },
            )
            
            self.logger.info("Requesting world news digest...")
            response = llm.invoke("Provide me a digest of world news in the last 24 hours.")
            
            print("\n" + "="*50)
            print("WORLD NEWS DIGEST")
            print("="*50)
            print(response.content)
            print("="*50)
            
            self.logger.info("Chat session completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during chat session: {e}")
            print(f"Error: {e}")
            raise


def main():
    """Main function to run the agent demo."""
    try:
        # Create and run the agent
        agent = AgentMain()
        agent.run_chat()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())