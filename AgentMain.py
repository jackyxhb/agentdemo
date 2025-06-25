# pip install -qU langchain-xai langchain-google-genai python-dotenv
import os
import logging
import argparse
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI


class AgentMain:
    """Main agent class for news queries using xAI's Grok API or Google Gemini."""
    
    def __init__(self, model_source: str = "grok", api_key: str = None):
        """
        Initialize the AgentMain class.
        
        Args:
            model_source (str): Model source - either "grok" or "gemini"
            api_key (str, optional): API key. If not provided, will load from environment.
        """
        self.model_source = model_source.lower()
        self.api_key = api_key
        self.logger = self._setup_logging()
        
        # Load environment variables
        load_dotenv()
        self.logger.info("✓ Environment variables loaded from .env file")
        
        # Validate and set API key
        self._validate_api_key()
        
        self.logger.info(f"Initialized agent with {self.model_source} model")
    
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
            if self.model_source == "grok":
                self.api_key = os.getenv("XAI_API_KEY")
                key_name = "XAI_API_KEY"
                key_url = "https://console.x.ai/"
            elif self.model_source == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
                key_name = "GOOGLE_API_KEY"
                key_url = "https://aistudio.google.com/app/apikey"
            else:
                raise ValueError(f"Unsupported model source: {self.model_source}")
        
        if not self.api_key:
            self.logger.error(f"{key_name} not found in environment variables or .env file.")
            self.logger.error("Please check your .env file or set the environment variable.")
            self.logger.error(f"You can get an API key from: {key_url}")
            raise ValueError(f"{key_name} is required")
    
    def run_chat(self) -> None:
        """Run a chat session with the agent."""
        self.logger.info(f"Running chat session with {self.model_source} model...")
        
        try:
            if self.model_source == "grok":
                llm = ChatXAI(
                    model="grok-3-latest",
                    api_key=self.api_key,
                    search_parameters={
                        "mode": "auto",
                        "max_search_results": 3,
                        "from_date": "2025-06-25",
                        "to_date": "2025-06-26",
                    },
                )
            elif self.model_source == "gemini":
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    api_key=self.api_key,
                    temperature=0.7,
                )
            else:
                raise ValueError(f"Unsupported model source: {self.model_source}")
            
            self.logger.info("Requesting world news digest...")
            response = llm.invoke("Provide me a digest of world news in the last 24 hours.")
            
            print("\n" + "="*50)
            print(f"WORLD NEWS DIGEST ({self.model_source.upper()})")
            print("="*50)
            print(response.content)
            print("="*50)
            
            self.logger.info("Chat session completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during chat session: {e}")
            print(f"Error: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="News digest agent using xAI Grok or Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AgentMain.py -m grok      # Use xAI Grok model
  python AgentMain.py -m gemini    # Use Google Gemini model
  python AgentMain.py              # Default to Grok model
        """
    )
    
    parser.add_argument(
        "-m", "--model",
        choices=["grok", "gemini"],
        default="grok",
        help="Choose the LLM model source (default: grok)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the agent demo."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create and run the agent
        agent = AgentMain(model_source=args.model)
        agent.run_chat()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())