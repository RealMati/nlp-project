"""
Utility functions for Text-to-SQL system
Helper functions for schema serialization, data processing, and evaluation
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def serialize_schema_for_prompt(
    tables: List[str],
    columns: List[Tuple[int, str]],
    column_types: Optional[List[str]] = None,
    format_style: str = "verbose"
) -> str:
    """
    Serialize database schema into text format for model input.

    Args:
        tables: List of table names
        columns: List of (table_idx, column_name) tuples
        column_types: Optional list of column types
        format_style: "verbose", "compact", or "natural"

    Returns:
        Formatted schema string
    """
    table_columns: Dict[int, List[str]] = {}

    for i, (table_idx, col_name) in enumerate(columns):
        if table_idx == -1:  # Skip * column
            continue
        if table_idx not in table_columns:
            table_columns[table_idx] = []

        if column_types and i < len(column_types):
            col_str = f"{col_name} ({column_types[i]})"
        else:
            col_str = col_name

        table_columns[table_idx].append(col_str)

    if format_style == "verbose":
        parts = []
        for idx, table_name in enumerate(tables):
            cols = table_columns.get(idx, [])
            parts.append(f"{table_name}: {', '.join(cols)}")
        return " | ".join(parts)

    elif format_style == "compact":
        parts = []
        for idx, table_name in enumerate(tables):
            cols = [c.split('(')[0].strip() for c in table_columns.get(idx, [])]
            parts.append(f"{table_name}({','.join(cols)})")
        return " ".join(parts)

    elif format_style == "natural":
        parts = []
        for idx, table_name in enumerate(tables):
            cols = [c.split('(')[0].strip() for c in table_columns.get(idx, [])]
            parts.append(f"{table_name} with columns {', '.join(cols)}")
        return "The database has tables: " + "; ".join(parts)

    else:
        raise ValueError(f"Unknown format_style: {format_style}")


def normalize_sql_query(sql: str) -> str:
    """
    Normalize SQL query for comparison.

    - Remove extra whitespace
    - Lowercase keywords (optionally)
    - Remove trailing semicolons
    - Normalize quotes

    Args:
        sql: SQL query string

    Returns:
        Normalized SQL string
    """
    # Remove trailing semicolon
    sql = sql.strip().rstrip(';')

    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)

    # Normalize quotes (optional - may affect string literals)
    # sql = sql.replace('\"', "'")

    return sql.strip()


def extract_sql_keywords(sql: str) -> Dict[str, bool]:
    """
    Extract SQL keywords and operations from query.

    Useful for analyzing query complexity and types.

    Args:
        sql: SQL query string

    Returns:
        Dictionary of keywords and their presence
    """
    sql_upper = sql.upper()

    keywords = {
        'SELECT': 'SELECT' in sql_upper,
        'INSERT': 'INSERT' in sql_upper,
        'UPDATE': 'UPDATE' in sql_upper,
        'DELETE': 'DELETE' in sql_upper,
        'JOIN': 'JOIN' in sql_upper,
        'WHERE': 'WHERE' in sql_upper,
        'GROUP_BY': 'GROUP BY' in sql_upper,
        'ORDER_BY': 'ORDER BY' in sql_upper,
        'HAVING': 'HAVING' in sql_upper,
        'LIMIT': 'LIMIT' in sql_upper,
        'UNION': 'UNION' in sql_upper,
        'SUBQUERY': '(' in sql and 'SELECT' in sql_upper,
        'AGGREGATE': any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
        'DISTINCT': 'DISTINCT' in sql_upper
    }

    return keywords


def calculate_query_complexity(sql: str) -> int:
    """
    Calculate query complexity score.

    Complexity factors:
    - Number of tables (JOINs)
    - Subqueries
    - Aggregations
    - GROUP BY / HAVING
    - UNION operations

    Args:
        sql: SQL query string

    Returns:
        Complexity score (0-10)
    """
    keywords = extract_sql_keywords(sql)

    score = 0

    # Base query
    if keywords['SELECT']:
        score += 1

    # JOINs (count occurrences)
    join_count = sql.upper().count('JOIN')
    score += min(join_count, 3)

    # Subqueries
    subquery_count = sql.count('SELECT') - 1  # Exclude main SELECT
    score += min(subquery_count, 2)

    # Aggregations
    if keywords['AGGREGATE']:
        score += 1

    # GROUP BY / HAVING
    if keywords['GROUP_BY']:
        score += 1
    if keywords['HAVING']:
        score += 1

    # UNION
    if keywords['UNION']:
        score += 1

    return min(score, 10)


def load_json_safely(file_path: Path) -> Any:
    """
    Safely load JSON file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json_safely(data: Any, file_path: Path, indent: int = 2) -> None:
    """
    Safely save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def print_dataset_stats(dataset_dict: Any) -> None:
    """
    Print statistics about a HuggingFace DatasetDict.

    Args:
        dataset_dict: DatasetDict object
    """
    logger.info("Dataset Statistics:")
    logger.info("=" * 50)

    for split_name, dataset in dataset_dict.items():
        logger.info(f"\n{split_name.upper()} Split:")
        logger.info(f"  Examples: {len(dataset)}")
        logger.info(f"  Features: {list(dataset.features.keys())}")

        if hasattr(dataset, 'size_in_bytes'):
            logger.info(f"  Size: {format_bytes(dataset.size_in_bytes)}")

    logger.info("=" * 50)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_table_names_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL query using regex.

    Basic extraction - may not handle all edge cases.

    Args:
        sql: SQL query string

    Returns:
        List of table names
    """
    # Pattern for FROM and JOIN clauses
    patterns = [
        r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    ]

    tables = set()
    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        tables.update(matches)

    return sorted(list(tables))


def compare_sql_structure(predicted: str, gold: str) -> Dict[str, bool]:
    """
    Compare structural components of two SQL queries.

    Args:
        predicted: Predicted SQL query
        gold: Gold standard SQL query

    Returns:
        Dictionary of component matches
    """
    pred_keywords = extract_sql_keywords(predicted)
    gold_keywords = extract_sql_keywords(gold)

    comparisons = {}
    for key in pred_keywords.keys():
        comparisons[f"{key}_match"] = pred_keywords[key] == gold_keywords[key]

    # Table comparison
    pred_tables = set(extract_table_names_from_sql(predicted))
    gold_tables = set(extract_table_names_from_sql(gold))

    comparisons['tables_match'] = pred_tables == gold_tables
    comparisons['table_subset'] = pred_tables.issubset(gold_tables)

    return comparisons


def create_model_input(
    question: str,
    schema: str,
    task_prefix: str = "translate to SQL"
) -> str:
    """
    Create formatted model input for T5.

    T5 expects: "task prefix: input"

    Args:
        question: Natural language question
        schema: Serialized database schema
        task_prefix: Task description prefix

    Returns:
        Formatted input string
    """
    return f"{task_prefix}: {question} | schema: {schema}"


def batch_process_examples(
    examples: List[Dict[str, Any]],
    process_fn: callable,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Any]:
    """
    Process examples in batches with optional progress bar.

    Args:
        examples: List of examples to process
        process_fn: Function to apply to each batch
        batch_size: Batch size
        show_progress: Whether to show progress bar

    Returns:
        List of processed results
    """
    from tqdm import tqdm

    results = []
    num_batches = (len(examples) + batch_size - 1) // batch_size

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing batches")

    for i in iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(examples))
        batch = examples[start_idx:end_idx]

        batch_results = process_fn(batch)
        results.extend(batch_results)

    return results


def validate_model_output(
    sql: str,
    max_length: int = 512,
    forbidden_keywords: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate model-generated SQL output.

    Args:
        sql: Generated SQL query
        max_length: Maximum allowed length
        forbidden_keywords: Keywords that shouldn't appear (e.g., DROP, DELETE)

    Returns:
        (is_valid, error_message)
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    if len(sql) > max_length:
        return False, f"SQL exceeds maximum length ({len(sql)} > {max_length})"

    # Check forbidden keywords
    if forbidden_keywords is None:
        forbidden_keywords = ['DROP', 'TRUNCATE', 'ALTER']

    sql_upper = sql.upper()
    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            return False, f"Forbidden keyword detected: {keyword}"

    # Basic SQL structure check
    if not any(kw in sql_upper for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
        return False, "No valid SQL operation found"

    return True, None


if __name__ == "__main__":
    # Test functions
    test_sql = "SELECT s.name, c.course_name FROM students s JOIN courses c ON s.id = c.student_id WHERE s.gpa > 3.5 GROUP BY s.major"

    print("SQL Keywords:", extract_sql_keywords(test_sql))
    print("Query Complexity:", calculate_query_complexity(test_sql))
    print("Tables:", extract_table_names_from_sql(test_sql))
    print("Normalized:", normalize_sql_query(test_sql))
