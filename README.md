# Agent Demo

A LangGraph agent demonstration using xAI's Grok API.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -qU langchain-openai langgraph python-dotenv
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your actual API keys:
   - Get Anthropic API key from: https://console.anthropic.com/
   - Get xAI API key from: https://console.x.ai/

3. **Run the agent:**
   ```bash
   python test.py
   ```

## Security

- The `.env` file is automatically ignored by git (see `.gitignore`)
- Never commit API keys to version control
- Use `.env.example` as a template for other developers

## Files

- `test.py` - Main agent script that uses Grok API
- `.env` - Environment variables (not tracked by git)
- `.env.example` - Template for environment variables
- `.gitignore` - Ensures sensitive files aren't committed