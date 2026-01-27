#!/usr/bin/env python3
"""
Test the SQL validation logic used in the Streamlit app with hard/edge case queries
"""

import sqlparse

def validate_sql(sql: str):
    """
    Validate SQL syntax (same function as in app.py)
    """
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "Empty SQL query"

        # Check for basic SQL structure
        sql_upper = sql.upper()
        if not any(kw in sql_upper for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return False, "No valid SQL operation found"

        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False, "Unbalanced parentheses"

        # Check for balanced quotes
        single_quotes = sql.count("'") - sql.count("\\'")
        if single_quotes % 2 != 0:
            return False, "Unbalanced quotes"

        return True, None
    except Exception as e:
        return False, str(e)

# Test cases - edge cases and potential SQL injection attempts
TEST_CASES = [
    # Valid SQL
    ("SELECT * FROM students", True, "Simple SELECT"),
    ("SELECT name, gpa FROM students WHERE gpa > 3.5", True, "SELECT with WHERE"),
    ("SELECT s.name, c.course_name FROM students s JOIN courses c ON s.id = c.student_id", True, "JOIN query"),

    # Complex valid SQL
    ("""SELECT s.*,
           (SELECT AVG(gpa) FROM students WHERE major = s.major) as avg_major_gpa
        FROM students s
        WHERE s.gpa > (SELECT AVG(gpa) FROM students)""", True, "Nested subqueries"),

    # Edge cases
    ("SELECT * FROM students WHERE name = 'O''Brien'", True, "Escaped single quote"),
    ("SELECT 1+1", True, "Simple expression"),
    ("SELECT CURRENT_TIMESTAMP", True, "Function without FROM"),

    # Invalid SQL
    ("SELECT * FROM students WHERE (name = 'John'", False, "Unbalanced parentheses"),
    ("SELECT * FROM students WHERE name = 'John", False, "Unbalanced quotes"),
    ("DELETE FROM students", True, "DELETE (valid syntax, but dangerous)"),
    ("DROP TABLE students", True, "DROP TABLE (valid syntax, but forbidden)"),
    ("", False, "Empty query"),
    ("Just some random text", False, "Not SQL"),

    # SQL Injection attempts (should pass syntax but would be caught by other validators)
    ("SELECT * FROM users WHERE username = 'admin' OR '1'='1'", True, "Classic SQL injection"),
    ("SELECT * FROM users WHERE id = 1; DROP TABLE users; --", True, "Stacked queries"),
    ("SELECT * FROM users WHERE username = 'admin'/**/OR/**/1=1", True, "Comment obfuscation"),

    # Very complex queries
    ("""WITH RECURSIVE department_hierarchy AS (
            SELECT id, name, parent_id, 1 as level
            FROM departments
            WHERE parent_id IS NULL
            UNION ALL
            SELECT d.id, d.name, d.parent_id, dh.level + 1
            FROM departments d
            JOIN department_hierarchy dh ON d.parent_id = dh.id
        )
        SELECT * FROM department_hierarchy""", True, "Recursive CTE"),

    ("""SELECT
            major,
            COUNT(*) as student_count,
            AVG(gpa) as avg_gpa,
            MAX(gpa) as max_gpa,
            MIN(gpa) as min_gpa,
            RANK() OVER (ORDER BY AVG(gpa) DESC) as rank
        FROM students
        GROUP BY major
        HAVING COUNT(*) > 5
        ORDER BY avg_gpa DESC""", True, "Window functions with aggregation"),
]

def run_validation_tests():
    print("=" * 80)
    print("SQL VALIDATION TESTING (Streamlit App Logic)")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for i, (sql, expected_valid, description) in enumerate(TEST_CASES, 1):
        is_valid, error = validate_sql(sql)

        # Check if result matches expectation
        test_passed = (is_valid == expected_valid)

        status = "✅ PASS" if test_passed else "❌ FAIL"
        result = "VALID" if is_valid else f"INVALID ({error})"

        print(f"{status} Test {i}: {description}")
        print(f"   Expected: {'VALID' if expected_valid else 'INVALID'} | Got: {result}")

        if not test_passed:
            print(f"   SQL: {sql[:100]}...")

        if test_passed:
            passed += 1
        else:
            failed += 1
        print()

    print("=" * 80)
    print(f"Results: {passed}/{len(TEST_CASES)} tests passed ({passed/len(TEST_CASES)*100:.1f}%)")
    print("=" * 80)
    print()

    # Security note
    print("⚠️  SECURITY NOTE:")
    print("   Syntax validation alone is NOT enough for security!")
    print("   The system also needs:")
    print("   - Forbidden operation detection (DROP, DELETE, etc.)")
    print("   - Parameterized queries for user inputs")
    print("   - Read-only database mode")
    print("   - Schema validation (table/column existence)")
    print()
    print("   ✅ All these features are implemented in:")
    print("      - src/sql_validator.py")
    print("      - src/database_manager.py")
    print()

if __name__ == "__main__":
    run_validation_tests()
