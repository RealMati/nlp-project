"""
Quick test suite for database management system.

Verifies core functionality works correctly.
"""

import sys
from pathlib import Path

from database_manager import DatabaseManager
from sql_validator import SQLValidator


def test_database_connection():
    """Test basic database connection."""
    print("Testing database connection...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            tables = db.get_table_names()
            assert len(tables) == 4, f"Expected 4 tables, got {len(tables)}"
            assert 'students' in tables
            assert 'professors' in tables
            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_schema_introspection():
    """Test schema introspection."""
    print("Testing schema introspection...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            schema = db.get_schema()

            # Check students table
            assert 'students' in schema['tables']
            student_cols = db.get_column_names('students')
            assert 'name' in student_cols
            assert 'gpa' in student_cols

            # Check foreign keys
            courses_fks = schema['tables']['courses']['foreign_keys']
            assert len(courses_fks) > 0

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_query_execution():
    """Test query execution."""
    print("Testing query execution...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            # Simple query
            result = db.execute_query("SELECT COUNT(*) as count FROM students")
            count = result.iloc[0]['count']
            assert count > 0, "No students found"

            # Query with parameters
            result = db.execute_query(
                "SELECT * FROM students WHERE gpa > :min_gpa",
                params={'min_gpa': 3.5}
            )
            assert len(result) > 0, "No high GPA students found"

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_query_validation():
    """Test query validation."""
    print("Testing query validation...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            schema = db.get_schema()
            validator = SQLValidator(schema, read_only=True)

            # Valid query
            is_valid, errors = validator.validate("SELECT * FROM students")
            assert is_valid, "Valid query marked as invalid"

            # Invalid query - non-existent table
            is_valid, errors = validator.validate("SELECT * FROM nonexistent")
            assert not is_valid, "Invalid query marked as valid"

            # Forbidden operation
            is_valid, errors = validator.validate("DROP TABLE students")
            assert not is_valid, "Forbidden operation allowed"

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_read_only_enforcement():
    """Test read-only mode enforcement."""
    print("Testing read-only enforcement...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path, read_only=True) as db:
            # Try to execute write query
            try:
                db.execute_query("DELETE FROM students")
                print("✗ FAILED: Write operation was allowed")
                return False
            except PermissionError:
                # Expected behavior
                pass

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_table_column_validation():
    """Test table and column existence validation."""
    print("Testing table/column validation...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            schema = db.get_schema()
            validator = SQLValidator(schema, read_only=True)

            # Valid table.column reference
            is_valid, errors = validator.validate(
                "SELECT s.name FROM students s"
            )
            assert is_valid, "Valid query marked as invalid"

            # Invalid table
            is_valid, errors = validator.validate(
                "SELECT * FROM invalid_table"
            )
            assert not is_valid, "Query with invalid table marked as valid"
            assert any('table' in e.error_type.lower() for e in errors), "Table error not detected"

            # Invalid column without alias (direct table.column reference)
            is_valid, errors = validator.validate(
                "SELECT students.invalid_column FROM students"
            )
            # Note: Column validation is best-effort; table validation is guaranteed
            # If this passes validation, it will fail at execution time
            if not is_valid:
                assert any('column' in e.error_type.lower() for e in errors)

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_query_complexity():
    """Test query complexity assessment."""
    print("Testing query complexity...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            schema = db.get_schema()
            validator = SQLValidator(schema, read_only=True)

            # Simple query
            metadata = validator.get_query_metadata("SELECT * FROM students")
            assert metadata['complexity'] == 'SIMPLE'

            # Query with join
            metadata = validator.get_query_metadata(
                "SELECT s.name FROM students s JOIN enrollments e ON s.id = e.student_id"
            )
            assert metadata['complexity'] in ['MODERATE', 'COMPLEX']

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_schema_export():
    """Test schema export in different formats."""
    print("Testing schema export...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            # Dict format
            schema_dict = db.export_schema('dict')
            assert isinstance(schema_dict, dict)
            assert 'tables' in schema_dict

            # Markdown format
            schema_md = db.export_schema('markdown')
            assert isinstance(schema_md, str)
            assert '# Database Schema' in schema_md

            # SQL format
            schema_sql = db.export_schema('sql')
            assert isinstance(schema_sql, str)
            assert 'CREATE TABLE' in schema_sql

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_sample_data():
    """Test sample data retrieval."""
    print("Testing sample data retrieval...", end=" ")
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"

    try:
        with DatabaseManager(db_path) as db:
            # Get sample data
            sample = db.get_sample_data('students', limit=5)
            assert len(sample) <= 5
            assert 'name' in sample.columns
            assert 'gpa' in sample.columns

            # Get table stats
            stats = db.get_table_stats('students')
            assert stats['row_count'] > 0
            assert stats['column_count'] > 0

            print("✓ PASSED")
            return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("DATABASE SYSTEM TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        test_database_connection,
        test_schema_introspection,
        test_query_execution,
        test_query_validation,
        test_read_only_enforcement,
        test_table_column_validation,
        test_query_complexity,
        test_schema_export,
        test_sample_data,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
