"""
Text-to-SQL Data Preparation Pipeline

Handles Spider dataset loading, preprocessing, and schema serialization
for T5/BART model fine-tuning.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


@dataclass
class SchemaInfo:
    """Holds serialized schema information for a database."""
    db_id: str
    tables: List[str]
    columns: Dict[str, List[str]]
    primary_keys: List[str]
    foreign_keys: List[Tuple[str, str]]
    serialized: str


class SpiderDataProcessor:
    """
    Processes Spider dataset for Text-to-SQL training.

    Handles:
    - Dataset loading from Hugging Face or local files
    - Schema serialization with table/column information
    - Input formatting: "translate to SQL: {question} | schema: {schema}"
    - Train/validation/test splits
    """

    def __init__(
        self,
        data_dir: str = "./data/spider",
        max_input_length: int = 512,
        max_target_length: int = 256,
        schema_format: str = "verbose"  # "verbose", "compact", "natural"
    ):
        self.data_dir = Path(data_dir)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.schema_format = schema_format
        self.schema_cache: Dict[str, SchemaInfo] = {}

    def load_spider_from_huggingface(self) -> DatasetDict:
        """Load Spider dataset from Hugging Face Hub."""
        print("Loading Spider dataset from Hugging Face...")
        dataset = load_dataset("spider", trust_remote_code=True)
        return dataset

    def load_spider_from_local(self) -> DatasetDict:
        """Load Spider dataset from local JSON files."""
        train_path = self.data_dir / "train_spider.json"
        dev_path = self.data_dir / "dev.json"

        if not train_path.exists() or not dev_path.exists():
            raise FileNotFoundError(
                f"Spider dataset files not found in {self.data_dir}. "
                "Download from https://yale-lily.github.io/spider or use --download flag."
            )

        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(dev_data)
        })

    def load_tables_schema(self, tables_path: Optional[str] = None) -> Dict[str, Any]:
        """Load schema information from tables.json."""
        if tables_path is None:
            tables_path = self.data_dir / "tables.json"
        else:
            tables_path = Path(tables_path)

        if not tables_path.exists():
            print(f"Warning: tables.json not found at {tables_path}")
            return {}

        with open(tables_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)

        # Index by db_id
        return {db["db_id"]: db for db in tables_data}

    def serialize_schema_verbose(self, db_schema: Dict[str, Any]) -> str:
        """
        Serialize schema in verbose format.

        Format: table_name: column1 (type), column2 (type) | ...
        """
        tables = db_schema.get("table_names_original", db_schema.get("table_names", []))
        columns = db_schema.get("column_names_original", db_schema.get("column_names", []))
        column_types = db_schema.get("column_types", [])

        # Group columns by table
        table_columns: Dict[int, List[str]] = {}
        for i, (table_idx, col_name) in enumerate(columns):
            if table_idx == -1:  # Skip * column
                continue
            if table_idx not in table_columns:
                table_columns[table_idx] = []
            col_type = column_types[i] if i < len(column_types) else "text"
            table_columns[table_idx].append(f"{col_name} ({col_type})")

        # Build schema string
        schema_parts = []
        for idx, table_name in enumerate(tables):
            cols = table_columns.get(idx, [])
            schema_parts.append(f"{table_name}: {', '.join(cols)}")

        return " | ".join(schema_parts)

    def serialize_schema_compact(self, db_schema: Dict[str, Any]) -> str:
        """
        Serialize schema in compact format.

        Format: table1(col1,col2) table2(col3,col4)
        """
        tables = db_schema.get("table_names_original", db_schema.get("table_names", []))
        columns = db_schema.get("column_names_original", db_schema.get("column_names", []))

        table_columns: Dict[int, List[str]] = {}
        for table_idx, col_name in columns:
            if table_idx == -1:
                continue
            if table_idx not in table_columns:
                table_columns[table_idx] = []
            table_columns[table_idx].append(col_name)

        schema_parts = []
        for idx, table_name in enumerate(tables):
            cols = table_columns.get(idx, [])
            schema_parts.append(f"{table_name}({','.join(cols)})")

        return " ".join(schema_parts)

    def serialize_schema_natural(self, db_schema: Dict[str, Any]) -> str:
        """
        Serialize schema in natural language format.

        Format: "The database has tables: table1 with columns col1, col2; ..."
        """
        tables = db_schema.get("table_names_original", db_schema.get("table_names", []))
        columns = db_schema.get("column_names_original", db_schema.get("column_names", []))

        table_columns: Dict[int, List[str]] = {}
        for table_idx, col_name in columns:
            if table_idx == -1:
                continue
            if table_idx not in table_columns:
                table_columns[table_idx] = []
            table_columns[table_idx].append(col_name)

        parts = []
        for idx, table_name in enumerate(tables):
            cols = table_columns.get(idx, [])
            parts.append(f"{table_name} with columns {', '.join(cols)}")

        return "The database has tables: " + "; ".join(parts)

    def serialize_schema(self, db_schema: Dict[str, Any]) -> str:
        """Serialize schema using configured format."""
        if self.schema_format == "verbose":
            return self.serialize_schema_verbose(db_schema)
        elif self.schema_format == "compact":
            return self.serialize_schema_compact(db_schema)
        elif self.schema_format == "natural":
            return self.serialize_schema_natural(db_schema)
        else:
            return self.serialize_schema_verbose(db_schema)

    def format_input(self, question: str, schema: str) -> str:
        """
        Format input for T5 model.

        T5 expects: "task prefix: input"
        Our format: "translate to SQL: {question} | schema: {schema}"
        """
        return f"translate to SQL: {question} | schema: {schema}"

    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for consistent comparison.

        - Lowercase keywords
        - Normalize whitespace
        - Remove trailing semicolons
        """
        sql = sql.strip()
        sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
        sql = sql.rstrip(';')
        return sql

    def preprocess_example(
        self,
        example: Dict[str, Any],
        schemas: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Preprocess a single example.

        Returns dict with 'input_text' and 'target_text' keys.
        """
        question = example["question"]
        db_id = example["db_id"]
        sql = example.get("query", example.get("sql", ""))

        # Get schema for this database
        if db_id in schemas:
            schema_str = self.serialize_schema(schemas[db_id])
        else:
            schema_str = ""

        input_text = self.format_input(question, schema_str)
        target_text = self.normalize_sql(sql)

        return {
            "input_text": input_text,
            "target_text": target_text,
            "db_id": db_id,
            "question": question
        }

    def prepare_dataset(
        self,
        use_huggingface: bool = True,
        tables_path: Optional[str] = None
    ) -> DatasetDict:
        """
        Prepare full dataset for training.

        Returns DatasetDict with train/validation splits,
        each containing 'input_text' and 'target_text' columns.
        """
        # Load raw data
        if use_huggingface:
            raw_dataset = self.load_spider_from_huggingface()
        else:
            raw_dataset = self.load_spider_from_local()

        # Load schemas
        schemas = self.load_tables_schema(tables_path)

        # Process each split
        processed_splits = {}
        for split_name, split_data in raw_dataset.items():
            print(f"Processing {split_name} split ({len(split_data)} examples)...")

            processed_examples = []
            for example in tqdm(split_data, desc=f"Processing {split_name}"):
                processed = self.preprocess_example(example, schemas)
                processed_examples.append(processed)

            processed_splits[split_name] = Dataset.from_list(processed_examples)

        return DatasetDict(processed_splits)

    def create_tokenized_dataset(
        self,
        dataset: DatasetDict,
        tokenizer,
        num_proc: int = 4
    ) -> DatasetDict:
        """
        Tokenize dataset for model training.

        Args:
            dataset: Prepared dataset with input_text/target_text
            tokenizer: Hugging Face tokenizer
            num_proc: Number of processes for parallel tokenization

        Returns:
            Tokenized DatasetDict ready for Trainer
        """
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False  # We'll pad in data collator
            )

            # Tokenize targets
            labels = tokenizer(
                text_target=examples["target_text"],
                max_length=self.max_target_length,
                truncation=True,
                padding=False
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["input_text", "target_text", "db_id", "question"],
            desc="Tokenizing"
        )

        return tokenized_dataset


class WikiSQLDataProcessor:
    """
    Processes WikiSQL dataset as alternative/supplementary training data.
    Simpler single-table queries, good for initial training.
    """

    def __init__(
        self,
        max_input_length: int = 512,
        max_target_length: int = 256
    ):
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def load_dataset(self) -> DatasetDict:
        """Load WikiSQL from Hugging Face."""
        return load_dataset("wikisql", trust_remote_code=True)

    def serialize_table_schema(self, table: Dict[str, Any]) -> str:
        """Serialize WikiSQL table schema."""
        header = table.get("header", [])
        types = table.get("types", [])

        cols = []
        for i, col in enumerate(header):
            col_type = types[i] if i < len(types) else "text"
            cols.append(f"{col} ({col_type})")

        return f"table: {', '.join(cols)}"

    def format_input(self, question: str, schema: str) -> str:
        """Format input for T5."""
        return f"translate to SQL: {question} | schema: {schema}"

    def sql_from_wikisql(self, sql_dict: Dict[str, Any], table: Dict[str, Any]) -> str:
        """Convert WikiSQL structured SQL to string."""
        header = table.get("header", [])

        # SELECT clause
        sel_idx = sql_dict.get("sel", 0)
        agg_idx = sql_dict.get("agg", 0)
        agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]

        sel_col = header[sel_idx] if sel_idx < len(header) else "*"
        if agg_idx > 0 and agg_idx < len(agg_ops):
            select_clause = f"{agg_ops[agg_idx]}({sel_col})"
        else:
            select_clause = sel_col

        # WHERE clause
        conds = sql_dict.get("conds", {})
        cond_ops = ["=", ">", "<", ">=", "<=", "!="]

        where_parts = []
        col_indices = conds.get("column_index", [])
        operators = conds.get("operator_index", [])
        conditions = conds.get("condition", [])

        for i in range(len(col_indices)):
            col_idx = col_indices[i]
            op_idx = operators[i]
            cond_val = conditions[i]

            col_name = header[col_idx] if col_idx < len(header) else "col"
            op = cond_ops[op_idx] if op_idx < len(cond_ops) else "="

            # Quote string values
            if isinstance(cond_val, str):
                cond_val = f"'{cond_val}'"

            where_parts.append(f"{col_name} {op} {cond_val}")

        sql = f"SELECT {select_clause} FROM table"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        return sql

    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Preprocess a WikiSQL example."""
        question = example["question"]
        table = example["table"]
        sql_dict = example["sql"]

        schema_str = self.serialize_table_schema(table)
        input_text = self.format_input(question, schema_str)
        target_text = self.sql_from_wikisql(sql_dict, table)

        return {
            "input_text": input_text,
            "target_text": target_text
        }

    def prepare_dataset(self) -> DatasetDict:
        """Prepare WikiSQL dataset."""
        raw_dataset = self.load_dataset()

        processed_splits = {}
        for split_name in ["train", "validation", "test"]:
            if split_name not in raw_dataset:
                continue

            split_data = raw_dataset[split_name]
            print(f"Processing WikiSQL {split_name} ({len(split_data)} examples)...")

            processed = []
            for example in tqdm(split_data, desc=split_name):
                try:
                    processed.append(self.preprocess_example(example))
                except Exception as e:
                    continue  # Skip malformed examples

            processed_splits[split_name] = Dataset.from_list(processed)

        return DatasetDict(processed_splits)


def download_spider_dataset(output_dir: str = "./data/spider"):
    """
    Download Spider dataset files.
    Note: Actual download requires manual steps due to licensing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Spider dataset download instructions:")
    print("1. Visit: https://yale-lily.github.io/spider")
    print("2. Download the dataset files")
    print("3. Extract to:", output_path)
    print("\nRequired files:")
    print("  - train_spider.json")
    print("  - dev.json")
    print("  - tables.json")
    print("  - database/ (folder with SQLite databases)")
    print("\nAlternatively, use Hugging Face datasets:")
    print("  from datasets import load_dataset")
    print("  dataset = load_dataset('spider')")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Text-to-SQL datasets")
    parser.add_argument("--download", action="store_true", help="Show download instructions")
    parser.add_argument("--data_dir", default="./data/spider", help="Data directory")
    parser.add_argument("--output_dir", default="./data/preprocessed", help="Output directory")
    parser.add_argument("--use_huggingface", action="store_true", help="Load from Hugging Face")
    parser.add_argument("--schema_format", default="verbose",
                       choices=["verbose", "compact", "natural"])

    args = parser.parse_args()

    if args.download:
        download_spider_dataset(args.data_dir)
    else:
        processor = SpiderDataProcessor(
            data_dir=args.data_dir,
            schema_format=args.schema_format
        )

        dataset = processor.prepare_dataset(use_huggingface=args.use_huggingface)

        # Save preprocessed data
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path / "spider_processed"))

        print(f"\nDataset saved to {output_path / 'spider_processed'}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
