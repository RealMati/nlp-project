#!/usr/bin/env python3
"""
Convert WikiSQL format to executable SQL and verify them against SQLite.
"""

import sys
import json
import sqlite3
import re
from typing import List, Dict, Any, Optional

# Aggregation operators
AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
OPS = ['=', '>', '<', '>=', '<=', '!=']


# SQL reserved words that need quoting (expanded list)
SQL_RESERVED = {
    # Basic keywords
    'order', 'group', 'table', 'index', 'select', 'from', 'where', 'join',
    'left', 'right', 'inner', 'outer', 'on', 'as', 'and', 'or', 'not',
    'limit', 'offset', 'union', 'all', 'distinct', 'null', 'is', 'like',
    'between', 'in', 'exists', 'case', 'when', 'then', 'else', 'end',
    'count', 'sum', 'avg', 'min', 'max', 'having', 'by', 'asc', 'desc',
    'primary', 'key', 'foreign', 'references', 'constraint', 'unique',
    'check', 'default', 'create', 'alter', 'drop', 'insert', 'update', 'delete',
    # Additional common SQL words
    'to', 'with', 'into', 'values', 'set', 'call', 'return', 'returning',
    'out', 'inout', 'procedure', 'function', 'trigger', 'view', 'schema',
    'current', 'timestamp', 'user', 'session', 'system', 'date', 'time',
    'datetime', 'year', 'month', 'day', 'hour', 'minute', 'second',
    'interval', 'zone', 'local', 'timestamp', 'extract', 'trim', 'overlay',
    'position', 'substring', 'convert', 'cast', 'collate', 'action',
    'cascade', 'begin', 'commit', 'rollback', 'savepoint', 'grant', 'revoke',
    'lock', 'explain', 'analyze', 'vacuum', 'reindex', 'pragma', 'trigger'
}

def clean_column_name(col: str, used_names: set = None) -> str:
    """Convert column name to valid SQL identifier (lowercase for SQLite compatibility)."""
    # Remove special chars, replace spaces with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', col)
    # Remove consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = 'col_' + cleaned
    # Handle empty column names (e.g., "#")
    if not cleaned:
        cleaned = 'col'
    # Normalize to lowercase for SQLite case-insensitivity
    cleaned = cleaned.lower()
    # Quote if reserved word
    if cleaned in SQL_RESERVED:
        cleaned = f'"{cleaned}"'
    # Handle duplicate names by adding suffix
    if used_names is not None:
        base_name = cleaned
        suffix = 0
        while cleaned in used_names:
            suffix += 1
            # Handle quoted names for reserved words
            if base_name.startswith('"') and base_name.endswith('"'):
                cleaned = f'"{base_name[1:-1]}_{suffix}"'
            else:
                cleaned = f'{base_name}_{suffix}'
        used_names.add(cleaned)
    return cleaned


def value_to_sql(value: Any) -> str:
    """Convert a value to SQL literal."""
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return '1' if value else '0'
    elif value is None:
        return 'NULL'
    else:
        return str(value)


def build_where_clause(conds: List[List], col_map: Dict[int, str]) -> str:
    """Build WHERE clause from conditions."""
    if not conds:
        return ''

    clauses = []
    for col_idx, op, val in conds:
        if col_idx in col_map:
            col_name = col_map[col_idx]
            val_sql = value_to_sql(val)
            op_str = OPS[op] if op < len(OPS) else '='
            clauses.append(f"{col_name} {op_str} {val_sql}")

    return ' WHERE ' + ' AND '.join(clauses) if clauses else ''


def build_select(sql_info: Dict, col_map: Dict[int, str], table_name: str) -> str:
    """Build SELECT clause."""
    sel = sql_info.get('sel', 0)
    agg = sql_info.get('agg', 0)

    col_name = col_map.get(sel, '1')
    if agg == 0:
        return f"SELECT {col_name} FROM {table_name}"
    else:
        agg_name = AGG_OPS[agg]
        return f"SELECT {agg_name}({col_name}) FROM {table_name}"


def get_column_names(headers: List[str]) -> tuple:
    """Generate clean column names with duplicate handling. Returns (col_names, used_names)."""
    used_names = set()
    col_names = [clean_column_name(h, used_names) for h in headers]
    return col_names, used_names


def convert_to_sql(query: Dict, table: Dict, table_name: str = "t1") -> str:
    """Convert WikiSQL query to SQL."""
    headers = table.get('header', [])
    sql_info = query.get('sql', {})

    # Create column index to name mapping with duplicate handling
    col_names, _ = get_column_names(headers)
    col_map = {i: col_names[i] for i in range(len(col_names))}

    select = build_select(sql_info, col_map, table_name)
    where = build_where_clause(sql_info.get('conds', []), col_map)

    return select + where


def create_table_sql(table: Dict, table_name: str = "t1") -> tuple:
    """Generate CREATE TABLE statement from table data."""
    headers = table.get('header', [])
    types = table.get('types', ['TEXT'] * len(headers))

    # Get column names with duplicate handling
    col_names, _ = get_column_names(headers)

    # Determine types based on data
    col_defs = []
    for i, (col_name, type_hint) in enumerate(zip(col_names, types)):
        if type_hint == 'real':
            col_type = 'REAL'
        elif type_hint == 'integer' or type_hint == 'int':
            col_type = 'INTEGER'
        else:
            col_type = 'TEXT'
        col_defs.append(f"{col_name} {col_type}")

    create = f"CREATE TABLE {table_name} ({', '.join(col_defs)})"
    return create, table_name, col_names


def populate_table(cursor, table: Dict, table_name: str, col_names: List[str] = None):
    """Insert rows into SQLite table."""
    headers = table.get('header', [])
    rows = table.get('rows', [])

    if not rows:
        return

    # Get column names (use provided or generate fresh)
    if col_names is None:
        col_names, _ = get_column_names(headers)

    placeholders = ', '.join(['?' for _ in col_names])
    insert_sql = f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES ({placeholders})"

    for row in rows:
        # Convert values appropriately
        processed_row = []
        for i, val in enumerate(row):
            type_hint = table.get('types', ['text'] * len(headers))[i]
            if type_hint == 'real' and isinstance(val, (int, float)):
                processed_row.append(float(val))
            elif type_hint == 'integer' and isinstance(val, (int, float)):
                processed_row.append(int(val))
            else:
                processed_row.append(str(val) if val is not None else None)

        try:
            cursor.execute(insert_sql, processed_row)
        except sqlite3.Error as e:
            # Print warning but continue
            print(f"Warning: Could not insert row {row}: {e}", file=sys.stderr)
            pass


def get_table_schema(tables_data: List[Dict], table_id: str) -> Optional[Dict]:
    """Get table schema by table_id."""
    for table in tables_data:
        if table.get('id') == table_id:
            return table
    return None


def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tables_file = os.path.join(base_dir, 'models', 'dataset', 'dev.tables.jsonl')
    queries_file = os.path.join(base_dir, 'models', 'dataset', 'dev.jsonl')

    # Load tables
    tables = []
    with open(tables_file, 'r') as f:
        for line in f:
            tables.append(json.loads(line))

    # Load queries
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))

    results = []
    errors = []

    for query in queries:
        table_id = query.get('table_id')
        question = query.get('question', '')
        table = get_table_schema(tables, table_id)

        if not table:
            errors.append((table_id, question, "Table not found"))
            continue

        sql = convert_to_sql(query, table)

        # Try to execute the SQL
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        try:
            create_sql, table_name, col_names = create_table_sql(table)
            cursor.execute(create_sql)
            populate_table(cursor, table, table_name, col_names)

            cursor.execute(sql)
            result = cursor.fetchone()

            results.append({
                'table_id': table_id,
                'question': question,
                'sql': sql,
                'result': str(result[0]) if result else 'NULL'
            })
        except sqlite3.Error as e:
            errors.append((table_id, question, f"{sql}: {e}"))
        finally:
            conn.close()

    # Output results
    print("table_id,question,sql,result")
    for r in results:
        q = r['question'].replace('"', '""')
        sql = r['sql'].replace('"', '""')
        res = r['result'].replace('"', '""')
        print(f'"{r["table_id"]}","{q}","{sql}","{res}"')

    print(f"\nTotal: {len(queries)}, Successful: {len(results)}, Errors: {len(errors)}", file=sys.stderr)


if __name__ == '__main__':
    main()
