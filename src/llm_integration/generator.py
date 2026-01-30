from .config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL_NAME, LLM_API_BASE
import time

class LLMGenerator:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL_NAME
        self.api_key = LLM_API_KEY
        self.base_url = LLM_API_BASE
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model or "gemini-1.5-flash")
            except ImportError:
                raise ImportError("Google Generative AI SDK not found. Please run `pip install google-generativeai`.")
        elif self.provider in ["openai", "ollama", "local", "groq"]:
            try:
                from openai import OpenAI
                # Groq uses a specific base URL if not provided, but usually we set it in env
                base_url = self.base_url
                if self.provider == "groq" and not base_url:
                    base_url = "https://api.groq.com/openai/v1"
                
                return OpenAI(api_key=self.api_key or "missing_key", base_url=base_url)
            except ImportError:
                raise ImportError("OpenAI SDK not found. Install it with `pip install openai`.")
        elif self.provider == "anthropic":
            # Placeholder for Anthropic
            # from anthropic import Anthropic
            # return Anthropic(api_key=self.api_key)
            raise NotImplementedError("Anthropic provider not fully implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_sql(self, prompt: str) -> str:
        """
        Generates SQL from the given prompt using the configured LLM.
        """
        try:
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                # Cleanup potential markdown formatting
                return response.text.replace('```sql', '').replace('```', '').strip()

            if self.provider in ["openai", "ollama", "local", "groq"]:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sophisticated SQL expert. Generate only the SQL query without markdown formatting or explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0, # Deterministic output
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            
            # Add other providers here...
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return f"-- Error: {str(e)}"
