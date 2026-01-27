#!/usr/bin/env python3
"""
Text-to-SQL Evaluation Script

Comprehensive evaluation for fine-tuned models with multiple metrics:
- Exact Match Accuracy: Normalized SQL string comparison
- Execution Accuracy: Result set comparison
- Component Match: Individual clause accuracy

Usage:
    python evaluate.py --model_path ./models/t5_finetuned --test_data ./data/spider/dev.json
    python evaluate.py --model_path ./models/t5_finetuned --use_huggingface --split validation
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model_inference import Text2SQLInference, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SQLEvaluator:
    """
    Comprehensive SQL evaluation with multiple metrics.

    Metrics:
    - Exact Match: Predicted SQL exactly matches gold SQL (after normalization)
    - Execution Accuracy: Results of executing predicted SQL match gold results
    - Component Match: Individual SQL components (SELECT, WHERE, etc.) match
    """

    def __init__(self, db_dir: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            db_dir: Directory containing SQLite databases
        """
        self.db_dir = Path(db_dir) if db_dir else None

    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        sql = sql.strip().lower()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.rstrip(';')

        # Normalize quotes
        sql = sql.replace('"', "'")

        # Normalize operators
        sql = re.sub(r'\s*=\s*', ' = ', sql)
        sql = re.sub(r'\s*<>\s*', ' != ', sql)
        sql = re.sub(r'\s*!=\s*', ' != ', sql)

        return sql.strip()

    def exact_match(self, predicted: str, gold: str) -> bool:
        """Check if normalized SQL matches exactly."""
        return self.normalize_sql(predicted) == self.normalize_sql(gold)

    def execute_sql(
        self,
        sql: str,
        db_path: str,
        timeout: float = 30.0
    ) -> Tuple[Optional[List[Tuple]], Optional[str]]:
        """
        Execute SQL and return results.

        Returns:
            Tuple of (results, error_message)
        """
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return results, None
        except Exception as e:
            return None, str(e)

    def execution_match(
        self,
        predicted: str,
        gold: str,
        db_path: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if execution results match.

        Returns:
            Tuple of (is_match, details)
        """
        details = {
            "predicted_error": None,
            "gold_error": None,
            "predicted_rows": 0,
            "gold_rows": 0
        }

        # Execute predicted
        pred_results, pred_error = self.execute_sql(predicted, db_path)
        if pred_error:
            details["predicted_error"] = pred_error
            return False, details

        # Execute gold
        gold_results, gold_error = self.execute_sql(gold, db_path)
        if gold_error:
            details["gold_error"] = gold_error
            return False, details

        details["predicted_rows"] = len(pred_results) if pred_results else 0
        details["gold_rows"] = len(gold_results) if gold_results else 0

        # Compare results (order-independent)
        if pred_results is None or gold_results is None:
            return False, details

        # Convert to sets for comparison
        try:
            pred_set = set(map(tuple, pred_results)) if pred_results else set()
            gold_set = set(map(tuple, gold_results)) if gold_results else set()
            return pred_set == gold_set, details
        except (TypeError, Exception):
            # Handle unhashable types
            return pred_results == gold_results, details

    def component_match(self, predicted: str, gold: str) -> Dict[str, bool]:
        """
        Check individual SQL components.

        Returns:
            Dictionary of component matches
        """
        pred_lower = predicted.lower()
        gold_lower = gold.lower()

        components = {}

        # SELECT clause
        pred_select = re.search(r'select\s+(.+?)\s+from', pred_lower)
        gold_select = re.search(r'select\s+(.+?)\s+from', gold_lower)
        components["select"] = (
            pred_select and gold_select and
            self._normalize_list(pred_select.group(1)) == self._normalize_list(gold_select.group(1))
        )

        # FROM clause
        pred_from = re.search(r'from\s+(\w+)', pred_lower)
        gold_from = re.search(r'from\s+(\w+)', gold_lower)
        components["from"] = (
            pred_from and gold_from and
            pred_from.group(1) == gold_from.group(1)
        )

        # WHERE clause
        pred_where = 'where' in pred_lower
        gold_where = 'where' in gold_lower
        components["where_present"] = pred_where == gold_where

        # GROUP BY
        pred_group = 'group by' in pred_lower
        gold_group = 'group by' in gold_lower
        components["group_by"] = pred_group == gold_group

        # ORDER BY
        pred_order = 'order by' in pred_lower
        gold_order = 'order by' in gold_lower
        components["order_by"] = pred_order == gold_order

        # HAVING
        pred_having = 'having' in pred_lower
        gold_having = 'having' in gold_lower
        components["having"] = pred_having == gold_having

        # JOIN
        pred_join = 'join' in pred_lower
        gold_join = 'join' in gold_lower
        components["join"] = pred_join == gold_join

        # Aggregations
        agg_funcs = ['count(', 'sum(', 'avg(', 'min(', 'max(']
        pred_aggs = set(f for f in agg_funcs if f in pred_lower)
        gold_aggs = set(f for f in agg_funcs if f in gold_lower)
        components["aggregations"] = pred_aggs == gold_aggs

        return components

    def _normalize_list(self, items_str: str) -> set:
        """Normalize comma-separated items."""
        items = items_str.split(',')
        return set(item.strip() for item in items)

    def evaluate_single(
        self,
        predicted: str,
        gold: str,
        db_id: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Returns:
            Dictionary with all metrics
        """
        result = {
            "exact_match": self.exact_match(predicted, gold),
            "execution_match": None,
            "component_match": self.component_match(predicted, gold),
            "predicted_sql": predicted,
            "gold_sql": gold,
            "valid_sql": True,
            "error": None
        }

        # Try execution match if db_path provided
        if db_path and Path(db_path).exists():
            exec_match, exec_details = self.execution_match(predicted, gold, db_path)
            result["execution_match"] = exec_match
            result["execution_details"] = exec_details
            if exec_details.get("predicted_error"):
                result["valid_sql"] = False
                result["error"] = exec_details["predicted_error"]
        elif self.db_dir and db_id:
            db_path_computed = self.db_dir / db_id / f"{db_id}.sqlite"
            if db_path_computed.exists():
                exec_match, exec_details = self.execution_match(
                    predicted, gold, str(db_path_computed)
                )
                result["execution_match"] = exec_match
                result["execution_details"] = exec_details
                if exec_details.get("predicted_error"):
                    result["valid_sql"] = False
                    result["error"] = exec_details["predicted_error"]

        return result

    def evaluate_batch(
        self,
        predictions: List[str],
        golds: List[str],
        db_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions.

        Returns:
            Aggregated metrics
        """
        results = []
        db_ids = db_ids or [None] * len(predictions)

        iterator = zip(predictions, golds, db_ids)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                total=len(predictions),
                desc="Evaluating"
            )

        for pred, gold, db_id in iterator:
            result = self.evaluate_single(pred, gold, db_id)
            results.append(result)

        # Aggregate metrics
        n = len(results)
        metrics = {
            "exact_match_accuracy": sum(r["exact_match"] for r in results) / n if n > 0 else 0,
            "valid_sql_rate": sum(r["valid_sql"] for r in results) / n if n > 0 else 0,
            "total_examples": n
        }

        # Execution accuracy (only for examples with execution results)
        exec_results = [r for r in results if r["execution_match"] is not None]
        if exec_results:
            metrics["execution_accuracy"] = sum(
                r["execution_match"] for r in exec_results
            ) / len(exec_results)
            metrics["execution_evaluated"] = len(exec_results)

        # Component match rates
        component_totals = defaultdict(int)
        for r in results:
            for comp, matched in r["component_match"].items():
                if matched:
                    component_totals[comp] += 1
        metrics["component_accuracy"] = {
            comp: count / n if n > 0 else 0 for comp, count in component_totals.items()
        }

        return {
            "metrics": metrics,
            "results": results
        }


def load_test_data(
    test_path: Optional[str] = None,
    use_huggingface: bool = False,
    split: str = "validation"
) -> List[Dict[str, Any]]:
    """Load test data from file or Hugging Face."""
    if use_huggingface:
        from datasets import load_dataset
        dataset = load_dataset("spider", trust_remote_code=True)
        return list(dataset[split])
    else:
        with open(test_path, "r") as f:
            return json.load(f)


def load_schemas(tables_path: str) -> Dict[str, Any]:
    """Load schema information from tables.json."""
    if not Path(tables_path).exists():
        return {}
    with open(tables_path) as f:
        tables_data = json.load(f)
        return {db["db_id"]: db for db in tables_data}


def serialize_schema(db_schema: Dict[str, Any]) -> str:
    """Serialize database schema for model input."""
    tables = db_schema.get("table_names_original", db_schema.get("table_names", []))
    columns = db_schema.get("column_names_original", db_schema.get("column_names", []))

    # Group columns by table
    table_columns: Dict[int, List[str]] = {}
    for table_idx, col_name in columns:
        if table_idx == -1:  # Skip * column
            continue
        if table_idx not in table_columns:
            table_columns[table_idx] = []
        table_columns[table_idx].append(col_name)

    # Build schema string
    schema_parts = []
    for idx, table_name in enumerate(tables):
        cols = table_columns.get(idx, [])
        schema_parts.append(f"{table_name}: {', '.join(cols)}")

    return " | ".join(schema_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Text-to-SQL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Load test data from Hugging Face"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="./data/spider/database",
        help="Directory containing SQLite databases"
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        default="./data/spider/tables.json",
        help="Path to tables.json for schema"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for generation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum examples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test_data and not args.use_huggingface:
        parser.error("Must specify --test_data or --use_huggingface")

    # Load test data
    logger.info("Loading test data...")
    if args.test_data:
        test_data = load_test_data(args.test_data)
    else:
        test_data = load_test_data(use_huggingface=True, split=args.split)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    logger.info(f"Loaded {len(test_data)} examples")

    # Load schema
    schemas = load_schemas(args.tables_path)
    logger.info(f"Loaded schemas for {len(schemas)} databases")

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    gen_config = GenerationConfig(num_beams=args.num_beams)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    inference = Text2SQLInference(
        args.model_path,
        device=device,
        generation_config=gen_config
    )

    # Prepare inputs
    logger.info("Preparing inputs...")
    questions = []
    schema_strs = []
    gold_sqls = []
    db_ids = []

    for example in test_data:
        questions.append(example["question"])
        gold_sqls.append(example.get("query", example.get("sql", "")))
        db_id = example.get("db_id", "")
        db_ids.append(db_id)

        # Get schema
        if db_id in schemas:
            schema_str = serialize_schema(schemas[db_id])
        else:
            schema_str = ""
        schema_strs.append(schema_str)

    # Generate predictions
    logger.info("Generating predictions...")
    results = inference.predict_batch(
        questions=questions,
        schemas=schema_strs,
        batch_size=args.batch_size,
        show_progress=True
    )

    predictions = [r.predicted_sql for r in results]

    # Evaluate
    logger.info("Evaluating predictions...")
    evaluator = SQLEvaluator(db_dir=args.db_dir)
    eval_results = evaluator.evaluate_batch(predictions, gold_sqls, db_ids)

    # Print metrics
    metrics = eval_results["metrics"]
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {metrics['total_examples']}")
    logger.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy'] * 100:.2f}%")
    logger.info(f"Valid SQL Rate: {metrics['valid_sql_rate'] * 100:.2f}%")

    if "execution_accuracy" in metrics:
        logger.info(f"Execution Accuracy: {metrics['execution_accuracy'] * 100:.2f}%")
        logger.info(f"  (evaluated on {metrics['execution_evaluated']} examples)")

    logger.info("\nComponent Accuracy:")
    for comp, acc in metrics["component_accuracy"].items():
        logger.info(f"  {comp}: {acc * 100:.2f}%")

    logger.info("=" * 60)

    # Save detailed results
    output_path = Path(args.output)
    detailed_results = {
        "metrics": metrics,
        "args": vars(args),
        "predictions": [
            {
                "question": q,
                "predicted": p,
                "gold": g,
                "db_id": d,
                "exact_match": r["exact_match"],
                "valid_sql": r["valid_sql"],
                "execution_match": r.get("execution_match")
            }
            for q, p, g, d, r in zip(
                questions, predictions, gold_sqls, db_ids, eval_results["results"]
            )
        ]
    }

    with open(output_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    logger.info(f"\nDetailed results saved to {output_path}")

    # Create summary CSV
    df = pd.DataFrame(detailed_results["predictions"])
    csv_path = str(output_path).replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
