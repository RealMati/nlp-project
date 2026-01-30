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
        examples = self.retriever.retrieve_similar(user_question, k=3)
        
        # 2. Build Prompt
        prompt = self.prompt_builder.build_prompt(user_question, schema_info, examples)
        
        # 3. Generate SQL
        generated_sql = self.generator.generate_sql(prompt)
        
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
