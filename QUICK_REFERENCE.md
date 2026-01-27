# Database System - Quick Reference

## Installation
```bash
pip install sqlalchemy pandas sqlparse
python3 src/create_sample_db.py  # Create sample database
```

## Basic Usage

### Import
```python
from src.database_manager import DatabaseManager
from src.sql_validator import SQLValidator
from src.example_text_to_sql_integration import SafeTextToSQLExecutor
```

### Connect to Database
```python
# Simple connection
with DatabaseManager('data/databases/sample_university.db') as db:
    result = db.execute_query("SELECT * FROM students LIMIT 5")
    print(result)
```

### Validate Query
```python
with DatabaseManager('database.db') as db:
    validator = SQLValidator(db.get_schema(), read_only=True)
    is_valid, errors = validator.validate(query)

    if is_valid:
        result = db.execute_query(query)
    else:
        for e in errors:
            print(f"{e.severity}: {e.message}")
```

### Text-to-SQL Integration
```python
executor = SafeTextToSQLExecutor('database.db')

# Get context for ML model
schema = executor.get_schema_context()
samples = executor.get_sample_data_context()

# Execute generated SQL
result = executor.execute_generated_query(
    "Find students with GPA > 3.5",
    "SELECT * FROM students WHERE gpa > 3.5"
)

if result['success']:
    print(result['results'])
else:
    print(result['error'])
```

## Common Operations

### Schema Introspection
```python
tables = db.get_table_names()
columns = db.get_column_names('students')
schema = db.get_schema()
stats = db.get_table_stats('students')
```

### Parameterized Queries (SQL Injection Safe)
```python
query = "SELECT * FROM students WHERE name = :name AND gpa > :min_gpa"
params = {'name': 'John Doe', 'min_gpa': 3.0}
result = db.execute_query(query, params=params)
```

### Export Schema
```python
schema_dict = db.export_schema('dict')
schema_md = db.export_schema('markdown')
schema_sql = db.export_schema('sql')
```

### Query Metadata
```python
metadata = validator.get_query_metadata(query)
print(f"Complexity: {metadata['complexity']}")
print(f"Tables: {metadata['tables']}")
print(f"Has joins: {metadata['has_joins']}")
```

### Query Suggestions
```python
suggestions = validator.suggest_improvements(query)
for suggestion in suggestions:
    print(f"- {suggestion}")
```

## Security

### Read-Only Mode (Default)
```python
db = DatabaseManager('database.db', read_only=True)
# Blocks: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER
```

### Write Mode (Use with Caution)
```python
db = DatabaseManager('database.db', read_only=False)
with db.transaction() as conn:
    conn.execute(text("INSERT INTO ..."))
```

## Sample Queries

### Simple SELECT
```python
result = db.execute_query("SELECT * FROM students WHERE gpa > 3.5")
```

### JOIN Query
```python
query = """
SELECT s.name, c.course_name, e.grade
FROM students s
JOIN enrollments e ON s.id = e.student_id
JOIN courses c ON e.course_id = c.id
WHERE s.gpa > 3.5
"""
result = db.execute_query(query)
```

### Aggregation
```python
query = """
SELECT major, COUNT(*) as count, AVG(gpa) as avg_gpa
FROM students
GROUP BY major
ORDER BY avg_gpa DESC
"""
result = db.execute_query(query)
```

## Error Handling

### Check Validation
```python
is_valid, errors = validator.validate(query)
if not is_valid:
    for error in errors:
        if error.severity == 'error':
            print(f"ERROR: {error.message}")
```

### Try-Catch Execution
```python
try:
    result = db.execute_query(query)
except PermissionError:
    print("Write operation not allowed in read-only mode")
except ValueError:
    print("Table or column doesn't exist")
except Exception as e:
    print(f"Execution failed: {e}")
```

## Testing

### Run Tests
```bash
python3 src/test_database_system.py
```

### Run Demos
```bash
python3 src/demo_database_system.py
python3 src/example_text_to_sql_integration.py
```

## Sample Database Schema

**students**: id, name, gpa, major, enrollment_year, email
**professors**: id, name, department, salary, hire_date, email
**courses**: id, course_name, credits, department, professor_id
**enrollments**: student_id, course_id, grade, semester, enrollment_date

## File Locations

```
src/database_manager.py          - Core database operations
src/sql_validator.py             - Query validation
src/example_text_to_sql_integration.py - ML integration
src/test_database_system.py      - Test suite
data/databases/sample_university.db - Sample database
```

## ValidationError Severities

- **error**: Critical (prevents execution)
- **warning**: Potential issue (review)
- **info**: Informational only

## Query Complexity Levels

- **SIMPLE**: Single table
- **MODERATE**: Joins or GROUP BY
- **COMPLEX**: Multiple joins, subqueries
- **ADVANCED**: CTEs, window functions

## Cheat Sheet

| Task | Code |
|------|------|
| Connect | `DatabaseManager('db.db')` |
| List tables | `db.get_table_names()` |
| List columns | `db.get_column_names('table')` |
| Get schema | `db.get_schema()` |
| Execute query | `db.execute_query(sql, params)` |
| Validate query | `validator.validate(sql)` |
| Get sample data | `db.get_sample_data('table', 5)` |
| Table stats | `db.get_table_stats('table')` |
| Export schema | `db.export_schema('markdown')` |

## Common Patterns

### Pattern 1: Safe Query Execution
```python
with DatabaseManager('db.db') as db:
    validator = SQLValidator(db.get_schema())
    is_valid, errors = validator.validate(query)

    if is_valid:
        result = db.execute_query(query, params)
        return result
    else:
        return errors
```

### Pattern 2: Text-to-SQL Flow
```python
executor = SafeTextToSQLExecutor('db.db')

# 1. Get context
context = executor.get_schema_context()

# 2. Generate SQL (your model)
sql = model.generate(question, context)

# 3. Execute safely
result = executor.execute_generated_query(question, sql)

# 4. Handle result
if result['success']:
    return result['results']
else:
    log_error(result['error'])
    return None
```

### Pattern 3: Batch Validation
```python
queries = [query1, query2, query3]
results = []

for query in queries:
    is_valid, errors = validator.validate(query)
    results.append({
        'query': query,
        'valid': is_valid,
        'errors': errors
    })
```

## Tips

1. **Always use parameterized queries** - Never concatenate user input
2. **Default to read-only** - Only use write mode when necessary
3. **Validate before execution** - Catch errors early
4. **Use context managers** - Automatic cleanup with `with` statement
5. **Check table existence** - Use `db.table_exists()` before querying
6. **Export schema for ML** - Include in model prompts for better results
7. **Monitor complexity** - High complexity queries may need optimization

## Need Help?

- Read: `DATABASE_SYSTEM_README.md` (full documentation)
- Run: `src/demo_database_system.py` (examples)
- Test: `src/test_database_system.py` (verify installation)
- Review: `IMPLEMENTATION_SUMMARY.md` (overview)

---

**Remember**: Validate → Execute → Handle Errors
