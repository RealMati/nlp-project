"""
SQL Validator for safe query execution.

Features:
- Syntax validation using sqlparse
- Schema compatibility checking
- Forbidden operation detection
- Query complexity scoring
- Detailed error reporting
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token, Function
from sqlparse.tokens import Keyword, DML, DDL


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = 1      # Single table, no joins
    MODERATE = 2    # Multiple tables or basic joins
    COMPLEX = 3     # Multiple joins, subqueries
    ADVANCED = 4    # CTEs, window functions, nested subqueries


class ValidationError:
    """Represents a validation error."""

    def __init__(self, error_type: str, message: str, severity: str = 'error'):
        self.error_type = error_type
        self.message = message
        self.severity = severity  # 'error', 'warning', 'info'

    def __repr__(self):
        return f"ValidationError({self.severity.upper()}: {self.error_type} - {self.message})"


class SQLValidator:
    """Validates SQL queries against schema and safety rules."""

    # Forbidden operations for read-only mode
    FORBIDDEN_WRITE_OPERATIONS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
    }

    # Dangerous functions (for warning)
    DANGEROUS_FUNCTIONS = {
        'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE', 'EXEC', 'EXECUTE'
    }

    def __init__(self, db_schema: Dict[str, any], read_only: bool = True):
        """
        Initialize SQL validator.

        Args:
            db_schema: Database schema dictionary from DatabaseManager.get_schema()
            read_only: If True, enforce read-only validation
        """
        self.db_schema = db_schema
        self.read_only = read_only
        self.tables = set(db_schema.get('tables', {}).keys())
        self.table_columns = {
            table: {col['name'] for col in info['columns']}
            for table, info in db_schema.get('tables', {}).items()
        }

    def validate(self, query: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a SQL query.

        Args:
            query: SQL query string

        Returns:
            Tuple of (is_valid: bool, errors: List[ValidationError])
        """
        errors = []

        # 1. Basic syntax validation
        syntax_errors = self._validate_syntax(query)
        errors.extend(syntax_errors)

        # 2. Parse the query
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                errors.append(ValidationError(
                    'parse_error',
                    'Failed to parse SQL query',
                    'error'
                ))
                return False, errors
        except Exception as e:
            errors.append(ValidationError(
                'parse_error',
                f'SQL parsing failed: {str(e)}',
                'error'
            ))
            return False, errors

        statement = parsed[0]

        # 3. Check for forbidden operations
        if self.read_only:
            forbidden_errors = self._check_forbidden_operations(statement)
            errors.extend(forbidden_errors)

        # 4. Check for dangerous functions
        dangerous_warnings = self._check_dangerous_functions(query)
        errors.extend(dangerous_warnings)

        # 5. Validate tables exist
        table_errors = self._validate_tables(statement)
        errors.extend(table_errors)

        # 6. Validate columns exist (when possible)
        column_errors = self._validate_columns(statement)
        errors.extend(column_errors)

        # 7. Check query complexity
        complexity = self._assess_complexity(statement)
        if complexity == QueryComplexity.ADVANCED:
            errors.append(ValidationError(
                'complexity',
                f'Query complexity: {complexity.name}. Consider optimization.',
                'info'
            ))

        # Check if any critical errors exist
        has_errors = any(e.severity == 'error' for e in errors)
        return not has_errors, errors

    def _validate_syntax(self, query: str) -> List[ValidationError]:
        """Validate basic SQL syntax."""
        errors = []

        # Check for empty query
        if not query or not query.strip():
            errors.append(ValidationError(
                'empty_query',
                'Query is empty',
                'error'
            ))
            return errors

        # Check for unterminated strings
        if query.count("'") % 2 != 0 or query.count('"') % 2 != 0:
            errors.append(ValidationError(
                'syntax_error',
                'Unterminated string literal detected',
                'error'
            ))

        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            errors.append(ValidationError(
                'syntax_error',
                'Unbalanced parentheses',
                'error'
            ))

        return errors

    def _check_forbidden_operations(self, statement) -> List[ValidationError]:
        """Check for forbidden write operations."""
        errors = []
        query_text = str(statement).upper()

        for operation in self.FORBIDDEN_WRITE_OPERATIONS:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + operation + r'\b'
            if re.search(pattern, query_text):
                errors.append(ValidationError(
                    'forbidden_operation',
                    f"Operation '{operation}' is forbidden in read-only mode",
                    'error'
                ))

        return errors

    def _check_dangerous_functions(self, query: str) -> List[ValidationError]:
        """Check for potentially dangerous SQL functions."""
        warnings = []
        query_upper = query.upper()

        for func in self.DANGEROUS_FUNCTIONS:
            if func in query_upper:
                warnings.append(ValidationError(
                    'dangerous_function',
                    f"Potentially dangerous function '{func}' detected",
                    'warning'
                ))

        return warnings

    def _validate_tables(self, statement) -> List[ValidationError]:
        """Validate that referenced tables exist in the schema."""
        errors = []
        referenced_tables = self._extract_table_names(statement)

        for table in referenced_tables:
            if table.lower() not in {t.lower() for t in self.tables}:
                errors.append(ValidationError(
                    'table_not_found',
                    f"Table '{table}' does not exist in database schema. "
                    f"Available tables: {', '.join(sorted(self.tables))}",
                    'error'
                ))

        return errors

    def _validate_columns(self, statement) -> List[ValidationError]:
        """Validate that referenced columns exist in their tables."""
        errors = []

        # Extract table.column references
        column_refs = self._extract_column_references(statement)

        for table, column in column_refs:
            table_lower = table.lower()
            column_lower = column.lower()

            # Find matching table (case-insensitive)
            matched_table = None
            for t in self.table_columns:
                if t.lower() == table_lower:
                    matched_table = t
                    break

            if matched_table:
                table_cols = {c.lower() for c in self.table_columns[matched_table]}
                if column_lower not in table_cols:
                    errors.append(ValidationError(
                        'column_not_found',
                        f"Column '{column}' does not exist in table '{table}'. "
                        f"Available columns: {', '.join(sorted(self.table_columns[matched_table]))}",
                        'error'
                    ))

        return errors

    def _extract_table_names(self, statement) -> List[str]:
        """Extract table names from SQL statement."""
        tables = []

        def extract_from_token(token):
            if token.ttype is None:
                if hasattr(token, 'tokens'):
                    for t in token.tokens:
                        extract_from_token(t)
            elif token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN', 'INTO', 'UPDATE', 'TABLE'):
                # Get the next non-whitespace token
                idx = statement.tokens.index(token)
                for next_token in statement.tokens[idx + 1:]:
                    if next_token.ttype in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                        continue
                    if isinstance(next_token, IdentifierList):
                        for identifier in next_token.get_identifiers():
                            tables.append(str(identifier.get_real_name()))
                    elif isinstance(next_token, Identifier):
                        tables.append(str(next_token.get_real_name()))
                    elif next_token.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Keyword):
                        # Simple table name
                        name = str(next_token).strip('`"[]')
                        if name.upper() not in ('AS', 'ON', 'WHERE', 'SET', 'VALUES'):
                            tables.append(name)
                    break

        for token in statement.tokens:
            extract_from_token(token)

        return tables

    def _extract_column_references(self, statement) -> List[Tuple[str, str]]:
        """
        Extract table.column references from SQL statement.

        Returns:
            List of (table_name, column_name) tuples
        """
        column_refs = []
        query_str = str(statement)

        # Match patterns like table.column or table.*
        pattern = r'(\w+)\.(\w+|\*)'
        matches = re.findall(pattern, query_str)

        for table, column in matches:
            # Skip wildcards
            if column != '*':
                column_refs.append((table, column))

        return column_refs

    def _assess_complexity(self, statement) -> QueryComplexity:
        """Assess query complexity level."""
        query_str = str(statement).upper()

        # Count complexity indicators
        join_count = len(re.findall(r'\bJOIN\b', query_str))
        subquery_count = query_str.count('SELECT') - 1  # Subtract main SELECT
        has_cte = 'WITH' in query_str
        has_window = any(func in query_str for func in ['OVER(', 'PARTITION BY', 'ROW_NUMBER'])

        # Determine complexity
        if has_cte or has_window or subquery_count > 2:
            return QueryComplexity.ADVANCED
        elif join_count > 2 or subquery_count > 0:
            return QueryComplexity.COMPLEX
        elif join_count > 0 or 'GROUP BY' in query_str:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def get_query_metadata(self, query: str) -> Dict[str, any]:
        """
        Extract metadata about a SQL query.

        Returns:
            Dictionary with query metadata (type, tables, complexity, etc.)
        """
        parsed = sqlparse.parse(query)
        if not parsed:
            return {}

        statement = parsed[0]
        query_type = statement.get_type()

        return {
            'query_type': query_type,
            'tables': self._extract_table_names(statement),
            'complexity': self._assess_complexity(statement).name,
            'has_joins': 'JOIN' in str(statement).upper(),
            'has_subquery': str(statement).upper().count('SELECT') > 1,
            'has_aggregation': any(
                func in str(statement).upper()
                for func in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP BY']
            ),
            'formatted_query': sqlparse.format(
                query,
                reindent=True,
                keyword_case='upper'
            )
        }

    def suggest_improvements(self, query: str) -> List[str]:
        """
        Suggest query improvements.

        Returns:
            List of improvement suggestions
        """
        suggestions = []
        query_upper = query.upper()

        # Check for SELECT *
        if re.search(r'SELECT\s+\*', query_upper):
            suggestions.append(
                "Consider selecting specific columns instead of SELECT * for better performance"
            )

        # Check for missing WHERE clause on large operations
        if 'DELETE' in query_upper or 'UPDATE' in query_upper:
            if 'WHERE' not in query_upper:
                suggestions.append(
                    "WARNING: DELETE/UPDATE without WHERE clause will affect all rows"
                )

        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+['\"]%", query_upper):
            suggestions.append(
                "LIKE with leading wildcard (LIKE '%...') cannot use indexes efficiently"
            )

        # Check for OR conditions (consider UNION)
        or_count = len(re.findall(r'\bOR\b', query_upper))
        if or_count > 3:
            suggestions.append(
                f"Query has {or_count} OR conditions. Consider using UNION for better performance"
            )

        # Check for missing LIMIT on SELECT
        if 'SELECT' in query_upper and 'LIMIT' not in query_upper:
            suggestions.append(
                "Consider adding LIMIT clause to prevent retrieving excessive rows"
            )

        return suggestions


def format_validation_report(
    is_valid: bool,
    errors: List[ValidationError],
    query: str
) -> str:
    """
    Format validation results into a human-readable report.

    Args:
        is_valid: Whether query is valid
        errors: List of validation errors
        query: Original SQL query

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SQL VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
    lines.append("")

    if errors:
        # Group by severity
        error_list = [e for e in errors if e.severity == 'error']
        warning_list = [e for e in errors if e.severity == 'warning']
        info_list = [e for e in errors if e.severity == 'info']

        if error_list:
            lines.append("ERRORS:")
            for i, err in enumerate(error_list, 1):
                lines.append(f"  {i}. [{err.error_type}] {err.message}")
            lines.append("")

        if warning_list:
            lines.append("WARNINGS:")
            for i, warn in enumerate(warning_list, 1):
                lines.append(f"  {i}. [{warn.error_type}] {warn.message}")
            lines.append("")

        if info_list:
            lines.append("INFO:")
            for i, info in enumerate(info_list, 1):
                lines.append(f"  {i}. [{info.error_type}] {info.message}")
            lines.append("")
    else:
        lines.append("No validation issues found.")
        lines.append("")

    lines.append("QUERY:")
    lines.append(sqlparse.format(query, reindent=True, keyword_case='upper'))
    lines.append("=" * 60)

    return "\n".join(lines)
