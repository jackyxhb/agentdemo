import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("✓ Environment variables loaded from .env file")
print(f"XAI_API_KEY exists: {bool(os.getenv('XAI_API_KEY'))}")
print(f"XAI_API_KEY: {os.getenv('XAI_API_KEY')[:10]}...")
