"""
Database Manager for SQLite databases with schema introspection and safe query execution.

Features:
- SQLAlchemy-based connection management
- Schema introspection (tables, columns, types, foreign keys)
- Safe parameterized query execution
- Result formatting with pandas
- Transaction management
- Multiple database support
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: Union[str, Path], read_only: bool = True):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            read_only: If True, enforce read-only access (default: True)
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.engine: Optional[Engine] = None
        self._schema_cache: Optional[Dict[str, Any]] = None

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        uri = f"sqlite:///{self.db_path.absolute()}"
        if self.read_only:
            uri += "?mode=ro"

        try:
            self.engine = create_engine(uri, echo=False)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        if self.engine is None:
            raise RuntimeError("Database not connected")

        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def get_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get complete database schema information.

        Args:
            force_refresh: If True, bypass cache and re-inspect schema

        Returns:
            Dictionary containing schema information:
            {
                'tables': {
                    'table_name': {
                        'columns': [{'name': str, 'type': str, 'nullable': bool, 'primary_key': bool}],
                        'foreign_keys': [{'column': str, 'references_table': str, 'references_column': str}],
                        'indexes': [str]
                    }
                }
            }
        """
        if self._schema_cache is not None and not force_refresh:
            return self._schema_cache

        inspector = inspect(self.engine)
        schema = {'tables': {}}

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'primary_key': col.get('primary_key', False),
                    'default': col.get('default')
                })

            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                for local_col, ref_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    foreign_keys.append({
                        'column': local_col,
                        'references_table': fk['referred_table'],
                        'references_column': ref_col
                    })

            indexes = [idx['name'] for idx in inspector.get_indexes(table_name)]

            schema['tables'][table_name] = {
                'columns': columns,
                'foreign_keys': foreign_keys,
                'indexes': indexes
            }

        self._schema_cache = schema
        return schema

    def get_table_names(self) -> List[str]:
        """Get list of all table names in the database."""
        schema = self.get_schema()
        return list(schema['tables'].keys())

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Get column names for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            List of column names

        Raises:
            ValueError: If table doesn't exist
        """
        schema = self.get_schema()
        if table_name not in schema['tables']:
            raise ValueError(f"Table '{table_name}' does not exist")

        return [col['name'] for col in schema['tables'][table_name]['columns']]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return table_name in self.get_table_names()

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        if not self.table_exists(table_name):
            return False
        return column_name in self.get_column_names(table_name)

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute a SQL query safely.

        Args:
            query: SQL query string
            params: Optional dictionary of query parameters for safe substitution
            return_dataframe: If True, return pandas DataFrame; else list of dicts

        Returns:
            Query results as DataFrame or list of dictionaries

        Raises:
            PermissionError: If attempting write operations in read-only mode
            SQLAlchemyError: On query execution errors
        """
        # Check for write operations in read-only mode
        if self.read_only:
            write_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
            query_upper = query.upper().strip()
            for keyword in write_keywords:
                if keyword in query_upper:
                    raise PermissionError(
                        f"Write operation '{keyword}' not allowed in read-only mode"
                    )

        try:
            with self._get_connection() as conn:
                result = conn.execute(text(query), params or {})

                # For SELECT queries, fetch results
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = result.keys()

                    if return_dataframe:
                        return pd.DataFrame(rows, columns=columns)
                    else:
                        return [dict(zip(columns, row)) for row in rows]
                else:
                    # For non-SELECT queries (if allowed)
                    conn.commit()
                    return pd.DataFrame() if return_dataframe else []

        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"Query execution failed: {e}")

    def execute_write_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute a write query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string
            params: Optional dictionary of query parameters

        Returns:
            Number of rows affected

        Raises:
            PermissionError: If database is in read-only mode
        """
        if self.read_only:
            raise PermissionError("Cannot execute write queries in read-only mode")

        try:
            with self._get_connection() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"Write query execution failed: {e}")

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            with db_manager.transaction() as conn:
                conn.execute(text("INSERT INTO ..."))
                conn.execute(text("UPDATE ..."))
        """
        if self.read_only:
            raise PermissionError("Transactions not allowed in read-only mode")

        with self._get_connection() as conn:
            trans = conn.begin()
            try:
                yield conn
                trans.commit()
            except Exception as e:
                trans.rollback()
                raise

    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get sample rows from a table.

        Args:
            table_name: Name of the table
            limit: Number of rows to return

        Returns:
            DataFrame with sample rows
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        query = f"SELECT * FROM {table_name} LIMIT :limit"
        return self.execute_query(query, params={'limit': limit})

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table statistics (row count, column count, etc.)
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        row_count = self.execute_query(count_query).iloc[0]['row_count']

        columns = self.get_column_names(table_name)

        return {
            'table_name': table_name,
            'row_count': row_count,
            'column_count': len(columns),
            'columns': columns
        }

    def export_schema(self, output_format: str = 'dict') -> Union[Dict, str]:
        """
        Export database schema.

        Args:
            output_format: 'dict', 'sql', or 'markdown'

        Returns:
            Schema in requested format
        """
        schema = self.get_schema()

        if output_format == 'dict':
            return schema

        elif output_format == 'sql':
            # Generate CREATE TABLE statements
            sql_statements = []
            with self._get_connection() as conn:
                for table_name in schema['tables'].keys():
                    result = conn.execute(text(
                        f"SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"
                    ), {'name': table_name})
                    row = result.fetchone()
                    if row:
                        sql_statements.append(row[0])
            return '\n\n'.join(sql_statements)

        elif output_format == 'markdown':
            md_lines = ['# Database Schema\n']
            for table_name, table_info in schema['tables'].items():
                md_lines.append(f"## {table_name}\n")
                md_lines.append("| Column | Type | Nullable | Primary Key |")
                md_lines.append("|--------|------|----------|-------------|")
                for col in table_info['columns']:
                    md_lines.append(
                        f"| {col['name']} | {col['type']} | "
                        f"{'Yes' if col['nullable'] else 'No'} | "
                        f"{'Yes' if col['primary_key'] else 'No'} |"
                    )

                if table_info['foreign_keys']:
                    md_lines.append("\n**Foreign Keys:**")
                    for fk in table_info['foreign_keys']:
                        md_lines.append(
                            f"- `{fk['column']}` → `{fk['references_table']}.{fk['references_column']}`"
                        )
                md_lines.append('\n')

            return '\n'.join(md_lines)

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self._schema_cache = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        mode = "read-only" if self.read_only else "read-write"
        return f"DatabaseManager('{self.db_path}', mode={mode})"
