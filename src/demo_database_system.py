"""
Demonstration of Database Manager and SQL Validator.

This script shows how to:
1. Connect to a database
2. Validate SQL queries
3. Execute safe queries
4. Handle validation errors
5. Export schema information
"""

from pathlib import Path
from database_manager import DatabaseManager
from sql_validator import SQLValidator, format_validation_report


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_schema_introspection():
    """Demonstrate schema introspection capabilities."""
    print_section("1. SCHEMA INTROSPECTION")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        # Get table names
        tables = db.get_table_names()
        print(f"\nTables in database: {', '.join(tables)}")

        # Get schema information for each table
        for table in tables:
            print(f"\n{table.upper()}:")
            columns = db.get_column_names(table)
            print(f"  Columns: {', '.join(columns)}")

            stats = db.get_table_stats(table)
            print(f"  Row count: {stats['row_count']}")

        # Export schema as markdown
        print("\nSchema (Markdown format):")
        print(db.export_schema(output_format='markdown'))


def demo_valid_queries():
    """Demonstrate validation and execution of valid queries."""
    print_section("2. VALID QUERIES")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        # Get schema for validator
        schema = db.get_schema()
        validator = SQLValidator(schema, read_only=True)

        # Test queries
        valid_queries = [
            "SELECT * FROM students WHERE gpa > 3.5 LIMIT 5",
            "SELECT name, major, gpa FROM students ORDER BY gpa DESC LIMIT 10",
            """
            SELECT c.course_name, p.name as professor_name, c.credits
            FROM courses c
            JOIN professors p ON c.professor_id = p.id
            WHERE c.department = 'Computer Science'
            """,
            """
            SELECT s.name, COUNT(e.course_id) as course_count
            FROM students s
            LEFT JOIN enrollments e ON s.id = e.student_id
            GROUP BY s.id, s.name
            HAVING course_count > 5
            ORDER BY course_count DESC
            LIMIT 5
            """
        ]

        for i, query in enumerate(valid_queries, 1):
            print(f"\n--- Query {i} ---")
            print(query.strip())

            # Validate
            is_valid, errors = validator.validate(query)
            print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")

            if errors:
                for error in errors:
                    print(f"  {error.severity.upper()}: {error.message}")

            # Execute if valid
            if is_valid:
                result = db.execute_query(query)
                print(f"\nResults ({len(result)} rows):")
                print(result.to_string(index=False))

                # Get query metadata
                metadata = validator.get_query_metadata(query)
                print(f"\nQuery Metadata:")
                print(f"  Type: {metadata['query_type']}")
                print(f"  Complexity: {metadata['complexity']}")
                print(f"  Tables: {', '.join(metadata['tables'])}")
                print(f"  Has joins: {metadata['has_joins']}")
                print(f"  Has aggregation: {metadata['has_aggregation']}")

            print("\n" + "-" * 70)


def demo_invalid_queries():
    """Demonstrate validation errors for invalid queries."""
    print_section("3. INVALID QUERIES (Validation Errors)")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        schema = db.get_schema()
        validator = SQLValidator(schema, read_only=True)

        # Test invalid queries
        invalid_queries = [
            # Write operation in read-only mode
            ("INSERT INTO students (name, gpa) VALUES ('John Doe', 3.8)",
             "Write operation in read-only mode"),

            # Non-existent table
            ("SELECT * FROM nonexistent_table",
             "Table doesn't exist"),

            # Non-existent column
            ("SELECT name, invalid_column FROM students",
             "Column doesn't exist"),

            # Syntax error - unbalanced parentheses
            ("SELECT * FROM students WHERE (gpa > 3.0",
             "Syntax error - unbalanced parentheses"),

            # Forbidden operation
            ("DROP TABLE students",
             "Forbidden DROP operation"),
        ]

        for i, (query, description) in enumerate(invalid_queries, 1):
            print(f"\n--- Invalid Query {i}: {description} ---")
            print(f"Query: {query}")

            is_valid, errors = validator.validate(query)
            print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")

            if errors:
                print("\nErrors found:")
                for error in errors:
                    print(f"  [{error.severity.upper()}] {error.error_type}: {error.message}")

            print("\n" + "-" * 70)


def demo_query_suggestions():
    """Demonstrate query improvement suggestions."""
    print_section("4. QUERY IMPROVEMENT SUGGESTIONS")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        schema = db.get_schema()
        validator = SQLValidator(schema, read_only=True)

        # Queries that could be improved
        queries_to_improve = [
            "SELECT * FROM students",
            "SELECT * FROM students WHERE name LIKE '%Smith'",
            "SELECT * FROM students WHERE major = 'Computer Science' OR major = 'Engineering' OR major = 'Physics' OR major = 'Mathematics' OR major = 'Chemistry'",
        ]

        for query in queries_to_improve:
            print(f"\nQuery: {query}")
            suggestions = validator.suggest_improvements(query)

            if suggestions:
                print("\nSuggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            else:
                print("\nNo suggestions - query looks good!")

            print("\n" + "-" * 70)


def demo_validation_report():
    """Demonstrate formatted validation reports."""
    print_section("5. VALIDATION REPORTS")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        schema = db.get_schema()
        validator = SQLValidator(schema, read_only=True)

        # Complex query with warnings
        query = """
        SELECT *
        FROM students s
        JOIN enrollments e ON s.id = e.student_id
        JOIN courses c ON e.course_id = c.id
        WHERE s.name LIKE '%Smith'
        """

        is_valid, errors = validator.validate(query)
        report = format_validation_report(is_valid, errors, query)
        print(report)


def demo_safe_execution_flow():
    """Demonstrate complete safe query execution flow."""
    print_section("6. SAFE QUERY EXECUTION FLOW")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        schema = db.get_schema()
        validator = SQLValidator(schema, read_only=True)

        # User query
        user_query = """
        SELECT
            s.name,
            s.major,
            s.gpa,
            COUNT(e.course_id) as courses_taken,
            AVG(c.credits) as avg_credits
        FROM students s
        LEFT JOIN enrollments e ON s.id = e.student_id
        LEFT JOIN courses c ON e.course_id = c.id
        WHERE s.gpa > 3.0
        GROUP BY s.id, s.name, s.major, s.gpa
        ORDER BY s.gpa DESC
        LIMIT 10
        """

        print("User Query:")
        print(user_query)

        print("\n1. Validating query...")
        is_valid, errors = validator.validate(user_query)

        if not is_valid:
            print("   ✗ Validation failed!")
            for error in errors:
                if error.severity == 'error':
                    print(f"   ERROR: {error.message}")
            return

        print("   ✓ Validation passed!")

        # Show warnings if any
        warnings = [e for e in errors if e.severity == 'warning']
        if warnings:
            print("\n   Warnings:")
            for warning in warnings:
                print(f"   - {warning.message}")

        print("\n2. Executing query...")
        try:
            result = db.execute_query(user_query)
            print(f"   ✓ Query executed successfully!")
            print(f"   Retrieved {len(result)} rows")

            print("\n3. Results:")
            print(result.to_string(index=False))

        except Exception as e:
            print(f"   ✗ Execution failed: {e}")


def demo_parameterized_queries():
    """Demonstrate safe parameterized queries."""
    print_section("7. PARAMETERIZED QUERIES (SQL Injection Prevention)")

    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    with DatabaseManager(db_path, read_only=True) as db:
        print("Safe parameterized query (prevents SQL injection):")

        # Safe query with parameters
        query = "SELECT name, gpa, major FROM students WHERE gpa > :min_gpa AND major = :major"
        params = {
            'min_gpa': 3.5,
            'major': 'Computer Science'
        }

        print(f"\nQuery: {query}")
        print(f"Parameters: {params}")

        result = db.execute_query(query, params=params)
        print(f"\nResults ({len(result)} rows):")
        print(result.to_string(index=False))

        print("\n" + "-" * 70)
        print("\n⚠️  What NOT to do (SQL injection vulnerable):")
        print("❌ query = f\"SELECT * FROM students WHERE name = '{user_input}'\"")
        print("\n✓  What TO do (safe):")
        print("✓  query = \"SELECT * FROM students WHERE name = :name\"")
        print("✓  params = {'name': user_input}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  DATABASE MANAGEMENT SYSTEM DEMONSTRATION")
    print("  Safe SQL Validation and Execution")
    print("=" * 70)

    try:
        demo_schema_introspection()
        demo_valid_queries()
        demo_invalid_queries()
        demo_query_suggestions()
        demo_validation_report()
        demo_safe_execution_flow()
        demo_parameterized_queries()

        print("\n" + "=" * 70)
        print("  DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. Always validate queries before execution")
        print("2. Use parameterized queries to prevent SQL injection")
        print("3. Read-only mode by default for safety")
        print("4. Schema introspection helps validate table/column existence")
        print("5. Query complexity assessment helps identify optimization needs")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
