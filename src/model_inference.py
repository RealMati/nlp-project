"""
Text-to-SQL Model Inference Pipeline

Production-ready inference wrapper for fine-tuned T5/BART models
with schema injection, batch prediction, and SQL post-processing.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Optional SQL parsing
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for SQL generation."""
    max_length: int = 256
    min_length: int = 1
    num_beams: int = 5
    num_return_sequences: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True


@dataclass
class PredictionResult:
    """Container for a single prediction result."""
    question: str
    predicted_sql: str
    schema: Optional[str] = None
    confidence: Optional[float] = None
    alternatives: Optional[List[str]] = None
    is_valid: Optional[bool] = None
    execution_result: Optional[Any] = None
    error: Optional[str] = None


class Text2SQLInference:
    """
    Production inference wrapper for Text-to-SQL models.

    Features:
    - Automatic schema serialization
    - Batch prediction with progress tracking
    - SQL post-processing and validation
    - Multiple generation strategies (beam search, sampling)
    - Confidence scoring
    - Optional execution against SQLite databases
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to fine-tuned model directory
            device: Device to run inference on ("cuda", "cpu", or None for auto)
            generation_config: Configuration for text generation
        """
        self.model_path = Path(model_path)
        self.generation_config = generation_config or GenerationConfig()

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")

    def _load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from checkpoint."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def serialize_schema(
        self,
        tables: Dict[str, List[str]],
        format_type: str = "verbose"
    ) -> str:
        """
        Serialize database schema for model input.

        Args:
            tables: Dictionary mapping table names to column lists
                   Example: {"users": ["id", "name", "email"], "orders": ["id", "user_id", "amount"]}
            format_type: "verbose", "compact", or "natural"

        Returns:
            Serialized schema string
        """
        if format_type == "verbose":
            parts = []
            for table_name, columns in tables.items():
                cols_str = ", ".join(columns)
                parts.append(f"{table_name}: {cols_str}")
            return " | ".join(parts)

        elif format_type == "compact":
            parts = []
            for table_name, columns in tables.items():
                cols_str = ",".join(columns)
                parts.append(f"{table_name}({cols_str})")
            return " ".join(parts)

        elif format_type == "natural":
            parts = []
            for table_name, columns in tables.items():
                cols_str = ", ".join(columns)
                parts.append(f"{table_name} with columns {cols_str}")
            return "The database has tables: " + "; ".join(parts)

        else:
            return self.serialize_schema(tables, "verbose")

    def serialize_schema_from_sqlite(
        self,
        db_path: str,
        format_type: str = "verbose"
    ) -> str:
        """
        Extract and serialize schema from SQLite database.

        Args:
            db_path: Path to SQLite database file
            format_type: Schema format type

        Returns:
            Serialized schema string
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Get columns for each table
        schema_dict = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            schema_dict[table] = columns

        conn.close()

        return self.serialize_schema(schema_dict, format_type)

    def format_input(self, question: str, schema: str) -> str:
        """
        Format input for model.

        Args:
            question: Natural language question
            schema: Serialized schema string

        Returns:
            Formatted input string
        """
        return f"translate to SQL: {question} | schema: {schema}"

    def predict(
        self,
        question: str,
        schema: Optional[str] = None,
        tables: Optional[Dict[str, List[str]]] = None,
        db_path: Optional[str] = None,
        return_alternatives: bool = False,
        validate: bool = True,
        execute: bool = False
    ) -> PredictionResult:
        """
        Generate SQL for a single question.

        Args:
            question: Natural language question
            schema: Pre-serialized schema string (optional)
            tables: Schema as dict of tables to columns (optional)
            db_path: Path to SQLite database for schema extraction (optional)
            return_alternatives: Return multiple beam search results
            validate: Validate generated SQL syntax
            execute: Execute SQL against database (requires db_path)

        Returns:
            PredictionResult with generated SQL and metadata
        """
        # Determine schema
        if schema is None:
            if tables is not None:
                schema = self.serialize_schema(tables)
            elif db_path is not None:
                schema = self.serialize_schema_from_sqlite(db_path)
            else:
                schema = ""

        # Format input
        input_text = self.format_input(question, schema)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Configure generation
        gen_config = self.generation_config
        num_return = gen_config.num_return_sequences if return_alternatives else 1

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=gen_config.max_length,
                min_length=gen_config.min_length,
                num_beams=gen_config.num_beams,
                num_return_sequences=num_return,
                do_sample=gen_config.do_sample,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                repetition_penalty=gen_config.repetition_penalty,
                length_penalty=gen_config.length_penalty,
                no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                early_stopping=gen_config.early_stopping,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode
        sequences = outputs.sequences
        decoded = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Post-process
        predicted_sql = self.postprocess_sql(decoded[0])
        alternatives = [self.postprocess_sql(d) for d in decoded[1:]] if len(decoded) > 1 else None

        # Calculate confidence (based on sequence scores if available)
        confidence = None
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            # Convert log probability to probability
            confidence = float(torch.exp(outputs.sequences_scores[0]).item())

        # Validate SQL
        is_valid = None
        error = None
        if validate:
            is_valid, error = self.validate_sql(predicted_sql)

        # Execute if requested
        execution_result = None
        if execute and db_path and is_valid:
            try:
                execution_result = self.execute_sql(predicted_sql, db_path)
            except Exception as e:
                error = str(e)
                is_valid = False

        return PredictionResult(
            question=question,
            predicted_sql=predicted_sql,
            schema=schema,
            confidence=confidence,
            alternatives=alternatives,
            is_valid=is_valid,
            execution_result=execution_result,
            error=error
        )

    def predict_batch(
        self,
        questions: List[str],
        schemas: Optional[List[str]] = None,
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        Generate SQL for multiple questions.

        Args:
            questions: List of natural language questions
            schemas: List of schema strings (one per question, or None)
            batch_size: Batch size for inference
            show_progress: Show progress bar

        Returns:
            List of PredictionResult objects
        """
        results = []

        # Prepare inputs
        if schemas is None:
            schemas = [""] * len(questions)
        elif len(schemas) == 1:
            schemas = schemas * len(questions)

        # Process in batches
        num_batches = (len(questions) + batch_size - 1) // batch_size
        iterator = range(0, len(questions), batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Generating SQL")

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            batch_schemas = schemas[batch_start:batch_end]

            # Format inputs
            batch_inputs = [
                self.format_input(q, s)
                for q, s in zip(batch_questions, batch_schemas)
            ]

            # Tokenize
            inputs = self.tokenizer(
                batch_inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            gen_config = self.generation_config
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=gen_config.max_length,
                    min_length=gen_config.min_length,
                    num_beams=gen_config.num_beams,
                    do_sample=gen_config.do_sample,
                    temperature=gen_config.temperature,
                    early_stopping=gen_config.early_stopping
                )

            # Decode
            decoded = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Create results
            for question, schema, sql in zip(batch_questions, batch_schemas, decoded):
                processed_sql = self.postprocess_sql(sql)
                is_valid, error = self.validate_sql(processed_sql)

                results.append(PredictionResult(
                    question=question,
                    predicted_sql=processed_sql,
                    schema=schema,
                    is_valid=is_valid,
                    error=error
                ))

        return results

    def postprocess_sql(self, sql: str) -> str:
        """
        Post-process generated SQL.

        Handles:
        - Whitespace normalization
        - Common artifacts removal
        - Basic formatting

        Args:
            sql: Raw generated SQL

        Returns:
            Cleaned SQL string
        """
        if not sql:
            return ""

        # Strip whitespace
        sql = sql.strip()

        # Remove common artifacts
        sql = re.sub(r"<pad>|<unk>|<s>|</s>", "", sql)

        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)

        # Remove trailing semicolon inconsistencies
        sql = sql.rstrip(';')

        # Fix common tokenization issues
        sql = re.sub(r'\s+\(', '(', sql)
        sql = re.sub(r'\(\s+', '(', sql)
        sql = re.sub(r'\s+\)', ')', sql)
        sql = re.sub(r'\)\s+', ') ', sql)
        sql = re.sub(r'\s+,', ',', sql)
        sql = re.sub(r',\s+', ', ', sql)

        # Normalize SQL keywords to uppercase (optional, for consistency)
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE',
            'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AS',
            'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET',
            'UNION', 'INTERSECT', 'EXCEPT', 'DISTINCT', 'ALL',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'BETWEEN',
            'IS', 'NULL', 'ASC', 'DESC', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
        ]

        for keyword in keywords:
            # Case-insensitive replacement
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)

        return sql.strip()

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql:
            return False, "Empty SQL query"

        # Basic structure check
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
            return False, "SQL must start with SELECT, INSERT, UPDATE, DELETE, or WITH"

        # Parentheses balance check
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses"

        # Quote balance check
        single_quotes = sql.count("'") - sql.count("\\'")
        if single_quotes % 2 != 0:
            return False, "Unbalanced single quotes"

        # Use sqlparse for deeper validation if available
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql)
                if not parsed or not parsed[0].tokens:
                    return False, "Failed to parse SQL"

                # Check for complete statements
                stmt = parsed[0]
                stmt_type = stmt.get_type()
                if stmt_type == 'UNKNOWN':
                    return False, "Unknown SQL statement type"

            except Exception as e:
                return False, f"SQL parse error: {str(e)}"

        return True, None

    def execute_sql(
        self,
        sql: str,
        db_path: str,
        timeout: float = 5.0
    ) -> List[Tuple]:
        """
        Execute SQL against SQLite database.

        Args:
            sql: SQL query to execute
            db_path: Path to SQLite database
            timeout: Query timeout in seconds

        Returns:
            Query results as list of tuples

        Raises:
            sqlite3.Error: If query execution fails
        """
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return [tuple(row) for row in results]
        finally:
            conn.close()

    def compare_sql(
        self,
        predicted: str,
        ground_truth: str,
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare predicted SQL with ground truth.

        Args:
            predicted: Predicted SQL query
            ground_truth: Ground truth SQL query
            db_path: Optional database for execution accuracy

        Returns:
            Dictionary with comparison metrics
        """
        result = {
            "exact_match": False,
            "normalized_match": False,
            "execution_match": None
        }

        # Exact match
        result["exact_match"] = predicted.strip() == ground_truth.strip()

        # Normalized match (case-insensitive, whitespace-normalized)
        pred_normalized = re.sub(r'\s+', ' ', predicted.lower().strip())
        gt_normalized = re.sub(r'\s+', ' ', ground_truth.lower().strip())
        result["normalized_match"] = pred_normalized == gt_normalized

        # Execution match (if database provided)
        if db_path:
            try:
                pred_results = self.execute_sql(predicted, db_path)
                gt_results = self.execute_sql(ground_truth, db_path)
                result["execution_match"] = set(pred_results) == set(gt_results)
            except Exception as e:
                result["execution_match"] = False
                result["execution_error"] = str(e)

        return result


class SchemaHelper:
    """
    Helper class for schema management and serialization.

    Provides utilities for working with database schemas across
    different formats and sources.
    """

    @staticmethod
    def from_sqlite(db_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract detailed schema from SQLite database.

        Returns:
            Dictionary with table names as keys and column info as values
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5])
                })
            schema[table] = columns

        conn.close()
        return schema

    @staticmethod
    def from_dict(tables: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Create schema from simple dict format.

        Args:
            tables: Dict mapping table names to column name lists

        Returns:
            Detailed schema dictionary
        """
        schema = {}
        for table, columns in tables.items():
            schema[table] = [{"name": col, "type": "text"} for col in columns]
        return schema

    @staticmethod
    def serialize(
        schema: Dict[str, List[Dict[str, str]]],
        include_types: bool = True,
        include_pk: bool = False
    ) -> str:
        """
        Serialize schema to string format.

        Args:
            schema: Detailed schema dictionary
            include_types: Include column types
            include_pk: Include primary key markers

        Returns:
            Serialized schema string
        """
        parts = []
        for table, columns in schema.items():
            col_strs = []
            for col in columns:
                col_str = col["name"]
                if include_types and col.get("type"):
                    col_str += f" ({col['type']})"
                if include_pk and col.get("pk"):
                    col_str += " [PK]"
                col_strs.append(col_str)
            parts.append(f"{table}: {', '.join(col_strs)}")
        return " | ".join(parts)


def load_model(model_path: str, device: str = "auto") -> Text2SQLInference:
    """
    Convenience function to load model for inference.

    Args:
        model_path: Path to fine-tuned model
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        Text2SQLInference instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Text2SQLInference(model_path, device=device)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL Inference")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--schema", default=None, help="Database schema string")
    parser.add_argument("--db_path", default=None, help="Path to SQLite database")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Load model
    inference = load_model(args.model_path, args.device)

    # Generate SQL
    result = inference.predict(
        question=args.question,
        schema=args.schema,
        db_path=args.db_path,
        validate=True,
        execute=args.db_path is not None
    )

    print(f"\nQuestion: {result.question}")
    print(f"Generated SQL: {result.predicted_sql}")
    print(f"Valid: {result.is_valid}")
    if result.error:
        print(f"Error: {result.error}")
    if result.execution_result is not None:
        print(f"Results: {result.execution_result[:5]}...")  # First 5 rows
