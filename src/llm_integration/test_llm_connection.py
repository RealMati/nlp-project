import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.llm_integration.config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL_NAME, LLM_API_BASE

def test_connection():
    print(f"Testing connection to {LLM_PROVIDER.upper()} using model {LLM_MODEL_NAME}...")
    
    if LLM_PROVIDER == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE if LLM_API_BASE else None)
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": "Hello, say 'Connection Successful'!"}],
                max_tokens=20
            )
            print("Response:", response.choices[0].message.content)
            print("✅ Connection verified.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            
    elif LLM_PROVIDER == "anthropic":
        print("Anthropic test not yet implemented.")
        
    elif LLM_PROVIDER == "ollama":
        # Usually compatible with OpenAI library if using v1 endpoint, or uses requests
        print("Testing Ollama via generic request...")
        # (Simplified for now, assuming OpenAI compatibility or raw request)
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=LLM_API_BASE if LLM_API_BASE else "http://localhost:11434/v1",
                api_key="ollama", # required but unused
            )
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": "Hello, say 'Connection Successful'!"}],
            )
            print("Response:", response.choices[0].message.content)
            print("✅ Connection verified.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    if not LLM_API_KEY and LLM_PROVIDER != 'ollama':
        print("⚠️  Warning: LLM_API_KEY is missing in environment variables.")
    
    test_connection()
