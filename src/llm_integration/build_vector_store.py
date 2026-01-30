import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.llm_integration.retriever import Retriever

def load_spider_train_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def build_vector_store():
    # Path to Spider training data
    train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'spider', 'train_spider.json')
    
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        return

    print("Loading Spider training data...")
    data = load_spider_train_data(train_file)
    
    # Extract questions and SQLs
    examples = []
    print(f"Found {len(data)} examples. processing...")
    
    for item in data:
        # Spider data format: {'question': ..., 'query': ..., ...}
        question = item['question']
        sql = item['query']
        examples.append({'question': question, 'sql': sql})
        
    print(f"Initializing Retriever and Vector Store...")
    retriever = Retriever()
    
    print("Populating Vector Store (this may take a while)...")
    retriever.add_examples(examples)
    print("Done!")

if __name__ == "__main__":
    build_vector_store()
