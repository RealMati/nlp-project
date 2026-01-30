from typing import List, Tuple

class PromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, user_question: str, schema: str, few_shot_examples: List[Tuple[str, str]]) -> str:
        """
        Constructs the prompt for the LLM using DAIL-SQL inspired format.
        
        Args:
            user_question: The target question.
            schema: DDL or schema representation.
            few_shot_examples: List of (question, sql) tuples.
        """
        prompt = ["### Task"]
        prompt.append("Generate a SQL query to answer the following question based on the database schema.")
        
        prompt.append("\n### Database Schema")
        prompt.append(schema)
        
        if few_shot_examples:
            prompt.append("\n### Examples")
            for q, sql in few_shot_examples:
                prompt.append(f"Q: {q}")
                prompt.append(f"SQL: {sql}")
                prompt.append("") # Separator
        
        prompt.append("\n### Target Question")
        prompt.append(f"Q: {user_question}")
        prompt.append("SQL:")
        
        return "\n".join(prompt)

    @staticmethod
    def format_schema(table_info: dict) -> str:
        """
        Helper to format schema from internal representation to string.
        (Placeholder - assumes we have a way to get DDL or similar)
        """
        # TODO: Integrate with existing SchemaHelper or create new one
        # For now, expect pre-formatted string or handle simple dict
        return str(table_info) 
