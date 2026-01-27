# Model Integration Guide

## 🔌 Connecting Real Models to Streamlit UI

This guide shows how to replace mock functions in `app.py` with actual T5/BART model inference.

---

## 📋 Integration Checklist

- [ ] Create model inference module (`src/model_inference.py`)
- [ ] Update model loading function in `app.py`
- [ ] Replace SQL generation mock with real inference
- [ ] Connect to actual SQLite databases
- [ ] Test end-to-end pipeline
- [ ] Add error handling for edge cases

---

## 1️⃣ Model Inference Module

**Create: `src/model_inference.py`**

```python
"""
Model inference wrapper for Text-to-SQL generation
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Optional
import time


class Text2SQLModel:
    """
    Wrapper for T5-based Text-to-SQL model inference
    """

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize model and tokenizer

        Args:
            model_path: Path to fine-tuned model checkpoint
            tokenizer_path: Path to tokenizer (defaults to model_path)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    def prepare_input(self, natural_language: str, schema: Dict) -> str:
        """
        Prepare input by concatenating natural language with schema

        Args:
            natural_language: User's natural language query
            schema: Database schema dictionary

        Returns:
            Formatted input string for T5
        """
        # Format schema
        schema_str = " | ".join([
            f"{table}: {', '.join(columns)}"
            for table, columns in schema.items()
        ])

        # T5 format: "translate to SQL: <query> | Schema: <schema>"
        input_text = f"translate to SQL: {natural_language} | Schema: {schema_str}"

        return input_text

    def generate_sql(
        self,
        natural_language: str,
        schema: Dict,
        temperature: float = 0.3,
        num_beams: int = 4,
        max_length: int = 256
    ) -> Dict:
        """
        Generate SQL from natural language

        Args:
            natural_language: User's query in natural language
            schema: Database schema dictionary
            temperature: Sampling temperature (0.0 = deterministic)
            num_beams: Number of beams for beam search
            max_length: Maximum output length

        Returns:
            Dictionary with generated SQL and metadata
        """
        # Prepare input
        input_text = self.prepare_input(natural_language, schema)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=temperature > 0,
                early_stopping=True,
                num_return_sequences=1
            )

        generation_time = time.time() - start_time

        # Decode
        generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate confidence (using beam scores if available)
        # For simplicity, use inverse of generation time as proxy
        confidence = min(0.99, 1.0 - (generation_time / 5.0))

        return {
            "sql": generated_sql,
            "confidence": confidence,
            "tokens_generated": len(outputs[0]),
            "generation_time": generation_time,
            "beam_size": num_beams,
            "temperature": temperature
        }


# Global model instance (for caching)
_model_instance = None


def get_model(model_path: str) -> Text2SQLModel:
    """
    Get or create model instance (singleton pattern)

    Args:
        model_path: Path to model checkpoint

    Returns:
        Text2SQLModel instance
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = Text2SQLModel(model_path)

    return _model_instance
```

---

## 2️⃣ Update `app.py` Model Loading

**Replace the mock function:**

```python
# OLD (Mock)
@st.cache_resource
def load_model_placeholder():
    time.sleep(1)
    return {
        "model": "t5-base-finetuned",
        "status": "ready",
        "version": "1.0.0"
    }


# NEW (Real Model)
import sys
sys.path.append('./src')
from model_inference import get_model

@st.cache_resource
def load_text2sql_model():
    """
    Load fine-tuned T5 model for Text-to-SQL generation
    Cached globally to avoid reloading on each interaction
    """
    try:
        model_path = "./models/t5_finetuned"
        model = get_model(model_path)

        return {
            "model": model,
            "status": "ready",
            "version": "1.0.0",
            "model_name": "t5-base-finetuned"
        }
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None
```

**Update sidebar model loading:**

```python
# In sidebar
if not st.session_state.model_loaded:
    if st.button("🔄 Load Model"):
        with st.spinner("Loading T5 model... This may take a minute."):
            model_info = load_text2sql_model()

            if model_info:
                st.session_state.model_loaded = True
                st.session_state.model_info = model_info
                st.success("✓ Model loaded successfully!")
            else:
                st.error("✗ Failed to load model")
else:
    st.markdown('<span class="status-badge status-success">✓ Model Ready</span>', unsafe_allow_html=True)
    model_info = st.session_state.model_info
    st.markdown(f"**Model:** {model_info['model_name']}")
    st.markdown(f"**Version:** {model_info['version']}")
    st.markdown(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

---

## 3️⃣ Replace SQL Generation Mock

**OLD (Mock):**

```python
def generate_sql_mock(natural_language, database, temperature, beam_size):
    # Simple keyword matching...
    pass
```

**NEW (Real Inference):**

```python
def generate_sql_real(
    natural_language: str,
    database: str,
    temperature: float,
    beam_size: int
) -> Dict:
    """
    Generate SQL using the loaded T5 model

    Args:
        natural_language: User's natural language query
        database: Selected database name
        temperature: Model temperature
        beam_size: Beam search size

    Returns:
        Dictionary with SQL and generation metadata
    """
    # Get model from session state
    model_info = st.session_state.get('model_info')
    if not model_info or 'model' not in model_info:
        raise ValueError("Model not loaded. Please load model first.")

    model = model_info['model']

    # Get schema for selected database
    schema = DATABASE_SCHEMAS[database]['tables']

    # Generate SQL
    result = model.generate_sql(
        natural_language=natural_language,
        schema=schema,
        temperature=temperature,
        num_beams=beam_size,
        max_length=256
    )

    return result
```

**Update the generate button logic:**

```python
if generate_button and natural_language.strip():
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first from the sidebar")
    else:
        with st.spinner("🤖 Generating SQL query..."):
            try:
                result = generate_sql_real(
                    natural_language,
                    st.session_state.current_database,
                    temperature,
                    beam_size
                )

                st.session_state.generated_sql = result

                # Add to history
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'natural_language': natural_language,
                    'sql': result['sql'],
                    'database': st.session_state.current_database,
                    'confidence': result['confidence']
                })

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
```

---

## 4️⃣ Connect to Real Databases

**Create: `src/database_manager.py`**

```python
"""
Database connection and query execution
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class DatabaseManager:
    """
    Manages SQLite database connections and query execution
    """

    def __init__(self, db_dir: str = "./data/databases"):
        """
        Initialize database manager

        Args:
            db_dir: Directory containing SQLite database files
        """
        self.db_dir = Path(db_dir)

    def get_connection(self, database: str) -> sqlite3.Connection:
        """
        Get connection to specified database

        Args:
            database: Database name (without .db extension)

        Returns:
            SQLite connection object
        """
        db_path = self.db_dir / f"{database}.db"

        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        return sqlite3.connect(db_path)

    def execute_query(
        self,
        sql: str,
        database: str,
        timeout: int = 30
    ) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query and return results

        Args:
            sql: SQL query to execute
            database: Target database name
            timeout: Query timeout in seconds

        Returns:
            Tuple of (success, dataframe, error_message)
        """
        conn = None
        try:
            conn = self.get_connection(database)
            conn.execute(f"PRAGMA query_timeout = {timeout * 1000}")  # milliseconds

            # Execute query
            df = pd.read_sql_query(sql, conn)

            return True, df, None

        except sqlite3.Error as e:
            return False, None, f"Database error: {str(e)}"

        except pd.errors.DatabaseError as e:
            return False, None, f"Query execution error: {str(e)}"

        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

        finally:
            if conn:
                conn.close()

    def get_schema(self, database: str) -> dict:
        """
        Get schema information for database

        Args:
            database: Database name

        Returns:
            Dictionary mapping table names to column lists
        """
        conn = None
        try:
            conn = self.get_connection(database)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            schema = {}
            for table in tables:
                # Get columns for each table
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table] = columns

            return schema

        finally:
            if conn:
                conn.close()


# Global instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create DatabaseManager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
```

**Update `app.py` execution function:**

```python
# Add to imports
from src.database_manager import get_db_manager

def execute_sql_real(sql: str, database: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Execute SQL query on actual database

    Args:
        sql: SQL query to execute
        database: Target database name

    Returns:
        Tuple of (success, dataframe, error_message)
    """
    db_manager = get_db_manager()
    return db_manager.execute_query(sql, database)
```

**Replace in execute button:**

```python
if st.button("▶️ Execute Query", type="primary", use_container_width=True, disabled=not is_valid):
    with st.spinner("⚡ Executing query..."):
        start_time = time.time()

        try:
            success, df, error = execute_sql_real(
                result['sql'],
                st.session_state.current_database
            )
            execution_time = time.time() - start_time

            st.session_state.execution_results = {
                'success': success,
                'dataframe': df,
                'error': error,
                'execution_time': execution_time
            }

        except Exception as e:
            st.session_state.execution_results = {
                'success': False,
                'dataframe': None,
                'error': f"Execution error: {str(e)}",
                'execution_time': 0
            }
```

---

## 5️⃣ Create Sample Databases

**Script: `create_sample_databases.py`**

```python
"""
Create sample SQLite databases for demo
"""

import sqlite3
from pathlib import Path


def create_university_db(db_path: Path):
    """Create university sample database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gpa REAL,
            major TEXT,
            enrollment_year INTEGER
        )
    """)

    # Sample data
    students = [
        (1, 'Alice Johnson', 20, 3.8, 'Computer Science', 2022),
        (2, 'Bob Smith', 21, 3.9, 'Mathematics', 2021),
        (3, 'Carol White', 22, 3.7, 'Physics', 2022),
        (4, 'David Brown', 20, 3.6, 'Computer Science', 2023),
        (5, 'Emma Davis', 19, 3.95, 'Mathematics', 2023),
    ]
    cursor.executemany("INSERT INTO students VALUES (?, ?, ?, ?, ?, ?)", students)

    # Professors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS professors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            hire_date TEXT,
            salary INTEGER
        )
    """)

    professors = [
        (1, 'Dr. Smith', 'Computer Science', '2015-08-15', 95000),
        (2, 'Dr. Johnson', 'Mathematics', '2012-01-10', 92000),
        (3, 'Dr. Williams', 'Physics', '2018-09-01', 88000),
    ]
    cursor.executemany("INSERT INTO professors VALUES (?, ?, ?, ?, ?)", professors)

    # Courses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY,
            course_name TEXT NOT NULL,
            course_code TEXT,
            credits INTEGER,
            professor_id INTEGER,
            department TEXT,
            FOREIGN KEY (professor_id) REFERENCES professors(id)
        )
    """)

    courses = [
        (1, 'Database Systems', 'CS301', 3, 1, 'Computer Science'),
        (2, 'Machine Learning', 'CS401', 4, 1, 'Computer Science'),
        (3, 'Linear Algebra', 'MATH201', 3, 2, 'Mathematics'),
        (4, 'Quantum Mechanics', 'PHYS301', 4, 3, 'Physics'),
    ]
    cursor.executemany("INSERT INTO courses VALUES (?, ?, ?, ?, ?, ?)", courses)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    db_dir = Path("./data/databases")
    db_dir.mkdir(parents=True, exist_ok=True)

    create_university_db(db_dir / "university.db")
    print("✓ Created university.db")
```

**Run it:**

```bash
python create_sample_databases.py
```

---

## 6️⃣ Testing Pipeline

**Test script: `test_integration.py`**

```python
"""
Test end-to-end Text-to-SQL pipeline
"""

from src.model_inference import get_model
from src.database_manager import get_db_manager


def test_pipeline():
    """Test complete pipeline"""

    print("1. Loading model...")
    model = get_model("./models/t5_finetuned")
    print("✓ Model loaded")

    print("\n2. Loading database...")
    db_manager = get_db_manager()
    schema = db_manager.get_schema("university")
    print(f"✓ Schema loaded: {list(schema.keys())}")

    print("\n3. Generating SQL...")
    natural_language = "Show all students with GPA above 3.5"
    result = model.generate_sql(natural_language, schema)
    print(f"✓ Generated SQL:\n{result['sql']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Time: {result['generation_time']:.3f}s")

    print("\n4. Executing query...")
    success, df, error = db_manager.execute_query(result['sql'], "university")

    if success:
        print(f"✓ Query executed successfully")
        print(f"  Rows: {len(df)}")
        print(f"\nResults:\n{df}")
    else:
        print(f"✗ Query failed: {error}")


if __name__ == "__main__":
    test_pipeline()
```

---

## 7️⃣ Production Checklist

### Before Deployment

- [ ] **Model Performance**: Validate on test set (>80% execution accuracy)
- [ ] **Database Security**: Use read-only connections, no DROP/DELETE
- [ ] **Input Sanitization**: Validate natural language inputs
- [ ] **Error Handling**: Graceful failures with helpful messages
- [ ] **Timeout Limits**: Prevent long-running queries (30s max)
- [ ] **Rate Limiting**: Prevent abuse (10 queries/minute)
- [ ] **Logging**: Track queries, errors, generation time
- [ ] **Model Versioning**: Tag model checkpoints
- [ ] **Database Backups**: Regular backups of sample databases

### Performance Optimization

```python
# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use half precision for faster inference
if device.type == "cuda":
    model = model.half()

# Batch processing for multiple queries
def generate_batch(queries: List[str], schema: Dict):
    inputs = tokenizer(queries, padding=True, return_tensors="pt")
    outputs = model.generate(**inputs)
    return [tokenizer.decode(out) for out in outputs]
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Model Too Large for Memory

**Solution:**
```python
# Use quantized model (8-bit)
from transformers import T5ForConditionalGeneration, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = T5ForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Issue 2: Slow Generation Time

**Solution:**
```python
# Reduce beam size
num_beams = 2  # Instead of 4-8

# Use greedy decoding
outputs = model.generate(**inputs, do_sample=False, num_beams=1)

# Enable FP16 (if GPU available)
model = model.half()
```

### Issue 3: Database Locked Error

**Solution:**
```python
# Use timeout and retries
def execute_with_retry(query, database, max_retries=3):
    for attempt in range(max_retries):
        try:
            return db_manager.execute_query(query, database)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            raise
```

---

## 📊 Expected Performance Metrics

After integration, you should see:

| Metric | Target | Acceptable |
|--------|--------|-----------|
| Model Load Time | <10s | <30s |
| SQL Generation | <1s | <3s |
| Query Execution | <100ms | <1s |
| Execution Accuracy | >85% | >75% |
| Syntax Validity | >95% | >90% |

---

**Integration complete! Your Streamlit UI now uses real AI models.**
