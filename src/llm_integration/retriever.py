import os
import json
import pickle
from typing import List, Dict, Tuple
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
import string

class Retriever:
    def __init__(self, data_path: str = "data/spider/train_spider.json"):
        # Resolve absolute path to data
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(base_dir, "data", "spider", "train_spider.json")
        self.examples = []
        self.bm25 = None
        self._initialize()

    def _initialize(self):
        """Loads data and builds BM25 index in memory (Fast, <1s)."""
        if BM25Okapi is None:
            print("Error: rank_bm25 not installed. Run `pip install rank-bm25`.")
            return

        try:
            if not os.path.exists(self.data_path):
                print(f"Warning: Data file not found at {self.data_path}")
                return

            with open(self.data_path, 'r') as f:
                self.examples = json.load(f)
            
            # Tokenize questions for BM25
            print(f"Building BM25 Index for {len(self.examples)} examples...")
            corpus = [self._tokenize(ex['question']) for ex in self.examples]
            self.bm25 = BM25Okapi(corpus)
            
            print(f"Retriever initialized (BM25). Ready to search.")
            
        except Exception as e:
            print(f"Error initializing Retriever: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def retrieve_similar(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        """
        Retrieves top-k similar examples using keyword overlap (BM25).
        Returns: List of (question, sql) tuples.
        """
        if not self.bm25:
            print("Retriever not initialized.")
            return []
            
        tokenized_query = self._tokenize(query)
        # Get top n documents
        top_n_docs = self.bm25.get_top_n(tokenized_query, self.examples, n=k)
        
        print(f"DEBUG [Retriever]: Found {len(top_n_docs)} similar examples for query: '{query}'")
        
        results = []
        for ex in top_n_docs:
            # We return (Question, SQL)
            # Adjust key 'query' vs 'sql' based on dataset format. 
            # Spider json uses 'query' for the SQL string.
            sql = ex.get('query') or ex.get('sql') 
            results.append((ex['question'], sql))
                 
        return results

    def add_examples(self, examples: List[Dict]):
        # No-op for read-only BM25 on static file
        pass
