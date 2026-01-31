import time
from .retriever import Retriever
from .prompt_builder import PromptBuilder
from .generator import LLMGenerator

class LLMPipeline:
    def __init__(self):
        print("Initializing LLM Pipeline...")
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.generator = LLMGenerator()
        print("LLM Pipeline Ready.")

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, user_question: str, schema_info: str) -> dict:
        """
        End-to-end generation: Retrieve -> Prompt -> Generate
        Returns dict matching the existing app's expected format.
        """
        start_time = time.time()
        
        # 1. Retrieve specific examples
        print(f"\n--- [BACKEND FLOW: STEP 1] ---")
        print(f"Searching Keystore (BM25) for: '{user_question}'")
        examples = self.retriever.retrieve_similar(user_question, k=3)
        for i, (q, s) in enumerate(examples):
            print(f"  > Example {i+1} found: Q: '{q}'")
        
        # 2. Build Prompt
        print(f"\n--- [BACKEND FLOW: STEP 2] ---")
        print(f"Schema discovered from DB: {schema_info[:200]}...") # Truncated for terminal readability
        print(f"Building full prompt with {len(examples)} examples...")
        prompt = self.prompt_builder.build_prompt(user_question, schema_info, examples)
        
        # 3. Generate SQL
        print(f"\n--- [BACKEND FLOW: STEP 3] ---")
        print(f"Sending to Groq ({self.generator.model})...")
        generated_sql = self.generator.generate_sql(prompt)
        print(f"  > Received SQL: {generated_sql}\n")
        
        execution_time = time.time() - start_time
        
        return {
            "sql": generated_sql,
            "confidence": 0.85, # dynamic confidence not strictly implemented for LLM yet
            "tokens_generated": self.count_tokens(generated_sql),
            "generation_time": execution_time,
            "beam_size": 1, # Not applicable
            "temperature": 0.0,
            "prompt_used": prompt, # Extra debug info
            "examples_used": examples
        }
