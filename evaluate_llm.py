#!/usr/bin/env python3
"""
LLM Text-to-SQL Evaluation Script
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add root to path for evaluate.py import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm_integration.pipeline import LLMPipeline
from evaluate import SQLEvaluator, load_test_data, load_schemas, serialize_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Text-to-SQL")
    parser.add_argument("--test_data", type=str, default="./data/spider/dev.json", help="Path to test data")
    parser.add_argument("--db_dir", type=str, default="./data/spider/database", help="Database directory")
    parser.add_argument("--tables_path", type=str, default="./data/spider/tables.json", help="Tables JSON path")
    parser.add_argument("--output", type=str, default="llm_evaluation_results.json", help="Output file")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    
    args = parser.parse_args()

    # Load resources
    logger.info("Loading test data...")
    test_data = load_test_data(args.test_data)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    schemas = load_schemas(args.tables_path)
    
    # Initialize Pipeline
    logger.info("Initializing LLM Pipeline...")
    try:
        pipeline = LLMPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return

    # Generate
    logger.info("Generating predictions...")
    predictions = []
    gold_sqls = []
    db_ids = []
    questions = []
    
    for example in tqdm(test_data, desc="Generating"):
        try:
            question = example["question"]
            db_id = example["db_id"]
            gold_sql = example["query"]
            
            schema_str = serialize_schema(schemas[db_id]) if db_id in schemas else ""
            
            result = pipeline.generate(question, schema_str)
            pred_sql = result['sql']
            
            predictions.append(pred_sql)
            gold_sqls.append(gold_sql)
            db_ids.append(db_id)
            questions.append(question)
            
        except Exception as e:
            logger.error(f"Error processing example: {e}")
            predictions.append("SELECT * FROM error") # Placeholder for failed generation
            gold_sqls.append(example.get("query", ""))
            db_ids.append(example.get("db_id", ""))
            questions.append(example.get("question", ""))

    # Evaluate
    logger.info("Evaluating...")
    evaluator = SQLEvaluator(db_dir=args.db_dir)
    eval_results = evaluator.evaluate_batch(predictions, gold_sqls, db_ids)
    
    metrics = eval_results["metrics"]
    
    logger.info("=" * 60)
    logger.info("LLM EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {metrics['total_examples']}")
    logger.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy'] * 100:.2f}%")
    logger.info(f"Valid SQL Rate: {metrics['valid_sql_rate'] * 100:.2f}%")
    if "execution_accuracy" in metrics:
         logger.info(f"Execution Accuracy: {metrics['execution_accuracy'] * 100:.2f}%")

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "results": eval_results["results"]
        }, f, indent=2)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
