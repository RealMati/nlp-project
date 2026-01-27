# Database Management System for Text-to-SQL

A comprehensive SQLite database management and SQL validation system designed for safe query execution in Text-to-SQL applications. Built with security, validation, and ease of use in mind.

## Features

### Database Manager (`database_manager.py`)
- **SQLAlchemy-based connection management** for robust database operations
- **Complete schema introspection** (tables, columns, types, foreign keys, indexes)
- **Safe query execution** with parameterization support
- **Result formatting** as pandas DataFrames or dictionaries
- **Transaction management** for multi-step operations
- **Read-only mode enforcement** (default) for safety
- **Schema export** in multiple formats (dict, SQL, Markdown)
- **Multiple database support** with connection pooling

### SQL Validator (`sql_validator.py`)
- **Syntax validation** using sqlparse
- **Schema compatibility checking** (table/column existence)
- **Forbidden operation detection** (DROP, DELETE, UPDATE, etc.)
- **Dangerous function warnings** (LOAD_FILE, EXEC, etc.)
- **Query complexity assessment** (Simple → Advanced)
- **Column reference validation** (table.column checks)
- **Query improvement suggestions**
- **Detailed error reporting** with severity levels

### Sample University Database
- 30 students with diverse majors and GPAs
- 8 professors across multiple departments
- 15 courses with proper credits and assignments
- 200+ enrollments with grades and semesters
- Realistic foreign key relationships
- Indexed for query performance

## Installation

```bash
# Install dependencies
pip install sqlalchemy pandas sqlparse

# Or use requirements.txt
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Database Operations

```python
from database_manager import DatabaseManager

# Open database (read-only by default)
with DatabaseManager('data/databases/sample_university.db') as db:
    # Get schema information
    schema = db.get_schema()
    tables = db.get_table_names()
    print(f"Tables: {tables}")

    # Execute query
    result = db.execute_query(
        "SELECT name, gpa FROM students WHERE gpa > :min_gpa",
        params={'min_gpa': 3.5}
    )
    print(result)
```

### 2. Query Validation

```python
from database_manager import DatabaseManager
from sql_validator import SQLValidator

with DatabaseManager('database.db') as db:
    schema = db.get_schema()
    validator = SQLValidator(schema, read_only=True)

    # Validate query
    query = "SELECT * FROM students WHERE gpa > 3.5"
    is_valid, errors = validator.validate(query)

    if is_valid:
        result = db.execute_query(query)
        print(result)
    else:
        for error in errors:
            print(f"{error.severity}: {error.message}")
```

### 3. Text-to-SQL Integration

```python
from example_text_to_sql_integration import SafeTextToSQLExecutor

# Initialize executor
executor = SafeTextToSQLExecutor('database.db', read_only=True)

# Get schema context for your ML model
schema_context = executor.get_schema_context()
sample_data = executor.get_sample_data_context()

# Execute generated SQL safely
result = executor.execute_generated_query(
    natural_language="Show me students with GPA above 3.5",
    generated_sql="SELECT * FROM students WHERE gpa > 3.5"
)

if result['success']:
    print(result['results'])
else:
    print(f"Error: {result['error']}")
```

## Architecture

### DatabaseManager Class

```python
class DatabaseManager:
    def __init__(self, db_path: str, read_only: bool = True)
    def get_schema(self) -> Dict[str, Any]
    def get_table_names(self) -> List[str]
    def get_column_names(self, table_name: str) -> List[str]
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame
    def export_schema(self, output_format: str = 'dict') -> Union[Dict, str]
```

**Key Methods:**
- `get_schema()`: Returns complete database schema with columns, types, foreign keys
- `execute_query()`: Safely executes SQL with parameter substitution
- `table_exists()`: Check if table exists before querying
- `get_table_stats()`: Get row count and column information

### SQLValidator Class

```python
class SQLValidator:
    def __init__(self, db_schema: Dict, read_only: bool = True)
    def validate(self, query: str) -> Tuple[bool, List[ValidationError]]
    def get_query_metadata(self, query: str) -> Dict[str, Any]
    def suggest_improvements(self, query: str) -> List[str]
```

**Validation Checks:**
1. Syntax validation (balanced parentheses, terminated strings)
2. Forbidden operations (INSERT, UPDATE, DELETE in read-only mode)
3. Table existence validation
4. Column existence validation
5. Dangerous function detection
6. Query complexity assessment

**Error Severities:**
- `error`: Critical issues that prevent execution
- `warning`: Potential problems or dangerous patterns
- `info`: Informational messages about query characteristics

## Sample Database Schema

### Students Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Student name |
| gpa | REAL | Grade point average (0.0-4.0) |
| major | TEXT | Major field of study |
| enrollment_year | INTEGER | Year enrolled |
| email | TEXT | Student email |

### Professors Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Professor name |
| department | TEXT | Department |
| salary | REAL | Annual salary |
| hire_date | TEXT | Date hired |
| email | TEXT | Professor email |

### Courses Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| course_name | TEXT | Course title |
| credits | INTEGER | Credit hours |
| department | TEXT | Department |
| professor_id | INTEGER | Foreign key → professors.id |

### Enrollments Table
| Column | Type | Description |
|--------|------|-------------|
| student_id | INTEGER | Foreign key → students.id |
| course_id | INTEGER | Foreign key → courses.id |
| grade | TEXT | Letter grade (A, B, C, etc.) |
| semester | TEXT | Semester enrolled |
| enrollment_date | TEXT | Date enrolled |

## Usage Examples

### Example 1: Schema Introspection

```python
from database_manager import DatabaseManager

with DatabaseManager('database.db') as db:
    # Get all tables
    tables = db.get_table_names()
    print(f"Tables: {', '.join(tables)}")

    # Get columns for a table
    columns = db.get_column_names('students')
    print(f"Student columns: {', '.join(columns)}")

    # Get table statistics
    stats = db.get_table_stats('students')
    print(f"Row count: {stats['row_count']}")

    # Export schema as Markdown
    markdown = db.export_schema('markdown')
    print(markdown)
```

### Example 2: Safe Parameterized Queries

```python
# ❌ UNSAFE - SQL injection vulnerable
query = f"SELECT * FROM students WHERE name = '{user_input}'"

# ✅ SAFE - Parameterized query
query = "SELECT * FROM students WHERE name = :name"
params = {'name': user_input}
result = db.execute_query(query, params=params)
```

### Example 3: Validation with Error Handling

```python
from sql_validator import SQLValidator, format_validation_report

validator = SQLValidator(db_schema, read_only=True)

# Validate query
query = "SELECT * FROM students WHERE gpa > 3.5"
is_valid, errors = validator.validate(query)

# Generate formatted report
report = format_validation_report(is_valid, errors, query)
print(report)
```

### Example 4: Query Complexity Assessment

```python
metadata = validator.get_query_metadata(query)

print(f"Query Type: {metadata['query_type']}")
print(f"Complexity: {metadata['complexity']}")
print(f"Tables: {', '.join(metadata['tables'])}")
print(f"Has Joins: {metadata['has_joins']}")
print(f"Has Aggregation: {metadata['has_aggregation']}")
```

### Example 5: Transaction Management

```python
# For write operations (read_only=False)
db = DatabaseManager('database.db', read_only=False)

with db.transaction() as conn:
    conn.execute(text("INSERT INTO students (name, gpa) VALUES (:name, :gpa)"),
                 {'name': 'John Doe', 'gpa': 3.5})
    conn.execute(text("INSERT INTO enrollments (student_id, course_id) VALUES (:sid, :cid)"),
                 {'sid': 31, 'cid': 1})
    # Automatically commits if no errors, rolls back on exception
```

## Security Features

### 1. Read-Only Mode (Default)
```python
# Read-only: prevents all write operations
db = DatabaseManager('database.db', read_only=True)

# Attempting writes will raise PermissionError
db.execute_query("DELETE FROM students")  # ❌ Error!
```

### 2. SQL Injection Prevention
```python
# Always use parameterized queries
query = "SELECT * FROM students WHERE name = :name AND gpa > :min_gpa"
params = {'name': user_input, 'min_gpa': 3.0}
result = db.execute_query(query, params=params)
```

### 3. Forbidden Operation Detection
```python
validator = SQLValidator(schema, read_only=True)

# These are automatically blocked:
validator.validate("DROP TABLE students")      # ❌ Forbidden
validator.validate("DELETE FROM students")     # ❌ Forbidden
validator.validate("UPDATE students SET ...")  # ❌ Forbidden
validator.validate("INSERT INTO students ...") # ❌ Forbidden
```

### 4. Schema Validation
```python
# Validates table and column existence
validator.validate("SELECT invalid_column FROM students")  # ❌ Column not found
validator.validate("SELECT * FROM nonexistent_table")      # ❌ Table not found
```

## Query Complexity Levels

| Level | Characteristics | Example |
|-------|-----------------|---------|
| **SIMPLE** | Single table, no joins | `SELECT * FROM students WHERE gpa > 3.5` |
| **MODERATE** | Multiple tables or basic joins | `SELECT s.name FROM students s JOIN enrollments e ON s.id = e.student_id` |
| **COMPLEX** | Multiple joins, subqueries | `SELECT * FROM students WHERE id IN (SELECT student_id FROM enrollments)` |
| **ADVANCED** | CTEs, window functions, nested subqueries | `WITH top_students AS (...) SELECT * FROM top_students` |

## Best Practices

### 1. Always Validate Before Execution
```python
is_valid, errors = validator.validate(query)
if is_valid:
    result = db.execute_query(query)
else:
    # Handle validation errors
    for error in errors:
        print(f"Error: {error.message}")
```

### 2. Use Context Managers
```python
# Automatically closes connection
with DatabaseManager('database.db') as db:
    result = db.execute_query(query)
    # Connection closed automatically
```

### 3. Provide Schema Context to ML Models
```python
executor = SafeTextToSQLExecutor('database.db')

# Include in your model's prompt
schema_context = executor.get_schema_context()
sample_data = executor.get_sample_data_context()

prompt = f"""Given this schema:
{schema_context}

Sample data:
{sample_data}

Question: {user_question}
SQL:"""
```

### 4. Handle Errors Gracefully
```python
result = executor.execute_generated_query(question, generated_sql)

if result['success']:
    df = result['results']
    print(f"Found {len(df)} rows")
else:
    error = result['error']
    # Log error, retry with different prompt, etc.
    print(f"Query failed: {error}")
```

## Performance Considerations

### Indexes
The sample database includes indexes on:
- `students.major`
- `students.gpa`
- `courses.department`
- `enrollments.student_id`
- `enrollments.course_id`

### Query Optimization Tips
1. **Use specific columns** instead of `SELECT *`
2. **Add LIMIT clauses** to prevent excessive results
3. **Avoid leading wildcards** in LIKE patterns (`LIKE '%term'`)
4. **Use UNION** instead of multiple OR conditions
5. **Consider indexes** for frequently queried columns

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Database file not found | Check file path |
| `PermissionError` | Write operation in read-only mode | Set `read_only=False` or use SELECT |
| `ValidationError: table_not_found` | Table doesn't exist | Check table name spelling |
| `ValidationError: column_not_found` | Column doesn't exist | Check column name, use `get_column_names()` |
| `SQLAlchemyError` | SQL syntax error | Review query syntax, check validator |

## Testing

Run the demonstration scripts to verify installation:

```bash
# Test database creation
python3 src/create_sample_db.py

# Test database manager and validator
python3 src/demo_database_system.py

# Test Text-to-SQL integration
python3 src/example_text_to_sql_integration.py
```

## File Structure

```
proj/
├── src/
│   ├── database_manager.py          # Core database management
│   ├── sql_validator.py             # Query validation
│   ├── create_sample_db.py          # Database setup script
│   ├── demo_database_system.py      # System demonstration
│   └── example_text_to_sql_integration.py  # Integration example
├── data/
│   └── databases/
│       └── sample_university.db     # Sample database
└── DATABASE_SYSTEM_README.md        # This file
```

## Advanced Features

### Custom Validation Rules

```python
class CustomValidator(SQLValidator):
    def validate(self, query: str):
        is_valid, errors = super().validate(query)

        # Add custom rules
        if 'LIKE' in query.upper() and '%' not in query:
            errors.append(ValidationError(
                'custom_rule',
                'LIKE should include wildcard %',
                'warning'
            ))

        return is_valid, errors
```

### Multiple Database Support

```python
# Connect to multiple databases
db1 = DatabaseManager('database1.db')
db2 = DatabaseManager('database2.db')

# Query each independently
results1 = db1.execute_query("SELECT * FROM students")
results2 = db2.execute_query("SELECT * FROM employees")
```

### Schema Comparison

```python
db1 = DatabaseManager('old_schema.db')
db2 = DatabaseManager('new_schema.db')

schema1 = db1.get_schema()
schema2 = db2.get_schema()

# Compare tables
tables1 = set(schema1['tables'].keys())
tables2 = set(schema2['tables'].keys())

added_tables = tables2 - tables1
removed_tables = tables1 - tables2
```

## Contributing

When adding features:
1. Maintain backward compatibility
2. Add comprehensive error handling
3. Update validation rules as needed
4. Include docstrings and type hints
5. Test with sample database

## License

This database management system is part of the NLP Text-to-SQL project.

## Support

For issues or questions:
1. Check this README first
2. Run demo scripts to verify installation
3. Review error messages and validation reports
4. Check schema using `export_schema()`

---

**Remember**: Always validate queries before execution, use parameterized queries, and default to read-only mode for safety.
