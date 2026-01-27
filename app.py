"""
Text-to-SQL Interactive Demo
Professional Streamlit interface for natural language to SQL conversion
"""

import streamlit as st
import pandas as pd
import sqlite3
import sqlparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import os

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL System",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #3b82f6;
        --success: #22c55e;
        --error: #ef4444;
        --warning: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-light: #e2e8f0;
    }

    /* Headers */
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }

    .status-success {
        background: #22c55e;
        color: white;
    }

    .status-error {
        background: #ef4444;
        color: white;
    }

    .status-warning {
        background: #f59e0b;
        color: white;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Code blocks */
    .stCodeBlock {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }

    /* Example queries */
    .example-query {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }

    .example-query:hover {
        background: #334155;
        border-left-color: #22c55e;
        transform: translateX(4px);
    }

    /* Results table */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    /* Loading spinner */
    .stSpinner > div {
        border-color: #10b981 transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# Example queries with categories
EXAMPLE_QUERIES = {
    "Basic Queries": [
        "Show all students with GPA above 3.5",
        "List all courses offered in the Computer Science department",
        "Find students who are older than 20 years",
        "Display all employees hired after 2020"
    ],
    "Aggregations": [
        "What is the average salary by department?",
        "How many students are enrolled in each major?",
        "Count the number of courses per instructor",
        "Calculate the total revenue by product category"
    ],
    "Joins": [
        "List all professors and their courses",
        "Show students and their enrolled courses",
        "Display employees with their department names",
        "Find orders with customer details"
    ],
    "Complex Queries": [
        "Which departments have more than 10 employees?",
        "Find the top 5 highest-paid employees in each department",
        "Show courses with no enrolled students",
        "List students who have taken all required courses"
    ]
}


# Database schemas for different sample databases
DATABASE_SCHEMAS = {
    "university": {
        "description": "Academic database with students, courses, professors, and enrollments",
        "tables": {
            "students": ["id", "name", "age", "gpa", "major", "enrollment_year"],
            "professors": ["id", "name", "department", "hire_date", "salary"],
            "courses": ["id", "course_name", "course_code", "credits", "professor_id", "department"],
            "enrollments": ["id", "student_id", "course_id", "semester", "grade"]
        }
    },
    "company": {
        "description": "Corporate database with employees, departments, and projects",
        "tables": {
            "employees": ["id", "name", "position", "department_id", "salary", "hire_date"],
            "departments": ["id", "name", "location", "budget", "manager_id"],
            "projects": ["id", "name", "start_date", "end_date", "budget", "department_id"],
            "assignments": ["id", "employee_id", "project_id", "role", "hours"]
        }
    },
    "ecommerce": {
        "description": "E-commerce database with products, customers, and orders",
        "tables": {
            "customers": ["id", "name", "email", "address", "registration_date"],
            "products": ["id", "name", "category", "price", "stock", "supplier_id"],
            "orders": ["id", "customer_id", "order_date", "total_amount", "status"],
            "order_items": ["id", "order_id", "product_id", "quantity", "price"]
        }
    }
}


@st.cache_resource
def load_model_placeholder():
    """
    Placeholder for model loading - replace with actual model loading logic
    In production, this would load the fine-tuned T5/BART model
    """
    time.sleep(1)  # Simulate loading time
    return {
        "model": "t5-base-finetuned",
        "status": "ready",
        "version": "1.0.0"
    }


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_database' not in st.session_state:
        st.session_state.current_database = "university"
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = None


def generate_sql_mock(natural_language: str, database: str, temperature: float, beam_size: int) -> Dict:
    """
    Mock SQL generation - replace with actual model inference
    This demonstrates the expected interface for the real model
    """
    time.sleep(0.5)  # Simulate inference time

    # Simple keyword-based mock generation
    sql_query = ""
    confidence = 0.0

    nl_lower = natural_language.lower()

    if "gpa above" in nl_lower and "3.5" in nl_lower:
        sql_query = "SELECT * FROM students WHERE gpa > 3.5"
        confidence = 0.95
    elif "average salary" in nl_lower and "department" in nl_lower:
        sql_query = "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department"
        confidence = 0.92
    elif "professors" in nl_lower and "courses" in nl_lower:
        sql_query = "SELECT p.name as professor, c.course_name FROM professors p JOIN courses c ON p.id = c.professor_id"
        confidence = 0.88
    elif "students" in nl_lower and "enrolled" in nl_lower and "major" in nl_lower:
        sql_query = "SELECT major, COUNT(*) as student_count FROM students GROUP BY major"
        confidence = 0.90
    else:
        sql_query = f"-- SQL generation pending model implementation\n-- Input: {natural_language}"
        confidence = 0.50

    return {
        "sql": sql_query,
        "confidence": confidence,
        "tokens_generated": len(sql_query.split()),
        "generation_time": 0.5,
        "beam_size": beam_size,
        "temperature": temperature
    }


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """Validate SQL syntax using sqlparse"""
    try:
        # Parse SQL
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "Empty or invalid SQL statement"

        # Check for basic SQL keywords
        sql_upper = sql.upper()
        if not any(keyword in sql_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return False, "No valid SQL operation found"

        # Format SQL to check for syntax errors
        formatted = sqlparse.format(sql, reindent=True, keyword_case='upper')

        return True, "SQL syntax is valid"
    except Exception as e:
        return False, f"Syntax error: {str(e)}"


def execute_sql_mock(sql: str, database: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Mock SQL execution - replace with actual database execution
    Returns: (success, dataframe, error_message)
    """
    try:
        # Validate first
        is_valid, error_msg = validate_sql(sql)
        if not is_valid:
            return False, None, error_msg

        # Mock data for demonstration
        if "students" in sql.lower() and "gpa" in sql.lower():
            mock_data = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'name': ['Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown'],
                'age': [20, 21, 22, 20],
                'gpa': [3.8, 3.9, 3.7, 3.6],
                'major': ['Computer Science', 'Mathematics', 'Physics', 'Computer Science'],
                'enrollment_year': [2022, 2021, 2022, 2023]
            })
            return True, mock_data, None

        elif "department" in sql.lower() and "avg" in sql.lower():
            mock_data = pd.DataFrame({
                'department': ['Engineering', 'Sales', 'Marketing', 'HR', 'IT'],
                'avg_salary': [95000, 75000, 70000, 65000, 85000]
            })
            return True, mock_data, None

        elif "professor" in sql.lower() and "course" in sql.lower():
            mock_data = pd.DataFrame({
                'professor': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown'],
                'course_name': ['Database Systems', 'Machine Learning', 'Algorithms', 'Data Structures']
            })
            return True, mock_data, None

        elif "major" in sql.lower() and "count" in sql.lower():
            mock_data = pd.DataFrame({
                'major': ['Computer Science', 'Mathematics', 'Physics', 'Engineering'],
                'student_count': [45, 32, 28, 38]
            })
            return True, mock_data, None

        else:
            # Generic mock result
            mock_data = pd.DataFrame({
                'result': ['Query executed successfully'],
                'note': ['Replace with actual database connection']
            })
            return True, mock_data, None

    except Exception as e:
        return False, None, f"Execution error: {str(e)}"


def display_schema_info(database: str):
    """Display database schema information in an expandable section"""
    schema = DATABASE_SCHEMAS.get(database, {})

    st.markdown(f"### 📊 Database: {database.title()}")
    st.markdown(f"*{schema.get('description', 'No description available')}*")

    with st.expander("📋 View Schema Details", expanded=False):
        for table_name, columns in schema.get('tables', {}).items():
            st.markdown(f"**`{table_name}`**")
            cols_display = ", ".join([f"`{col}`" for col in columns])
            st.markdown(f"  {cols_display}")
            st.markdown("---")


def format_sql_with_syntax_highlighting(sql: str) -> str:
    """Format SQL with proper indentation and highlighting"""
    formatted = sqlparse.format(
        sql,
        reindent=True,
        keyword_case='upper',
        indent_width=2
    )
    return formatted


def main():
    """Main application logic"""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗄️ Text-to-SQL System</h1>
        <p>Transform natural language into executable SQL queries using AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Database selection
        selected_db = st.selectbox(
            "Select Database",
            options=list(DATABASE_SCHEMAS.keys()),
            format_func=lambda x: x.title(),
            key="database_selector"
        )
        st.session_state.current_database = selected_db

        st.markdown("---")

        # Model settings
        st.markdown("### 🤖 Model Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        beam_size = st.slider(
            "Beam Size",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of beams for beam search. Higher values may improve quality but increase computation time"
        )

        st.markdown("---")

        # Model status
        st.markdown("### 📊 Model Status")

        if not st.session_state.model_loaded:
            if st.button("🔄 Load Model"):
                with st.spinner("Loading model..."):
                    model_info = load_model_placeholder()
                    st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
                st.json(model_info)
        else:
            st.markdown('<span class="status-badge status-success">✓ Model Ready</span>', unsafe_allow_html=True)
            model_info = load_model_placeholder()
            st.markdown(f"**Model:** {model_info['model']}")
            st.markdown(f"**Version:** {model_info['version']}")

        st.markdown("---")

        # Query history
        if st.session_state.query_history:
            st.markdown("### 📜 Recent Queries")
            st.markdown(f"*{len(st.session_state.query_history)} queries in history*")

            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Schema information
        display_schema_info(st.session_state.current_database)
        st.markdown("---")

    with col2:
        # Quick stats
        st.markdown("### 📈 Statistics")

        metrics_html = f"""
        <div class="metric-card">
            <div class="metric-label">Queries Generated</div>
            <div class="metric-value">{len(st.session_state.query_history)}</div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

    # Example queries section
    st.markdown("## 💡 Example Queries")

    selected_category = st.selectbox(
        "Choose a category",
        options=list(EXAMPLE_QUERIES.keys()),
        label_visibility="collapsed"
    )

    # Display example queries as clickable buttons
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES[selected_category]):
        with cols[idx % 2]:
            if st.button(f"📝 {example}", key=f"example_{idx}", use_container_width=True):
                st.session_state.selected_example = example

    st.markdown("---")

    # Natural language input
    st.markdown("## 🎯 Enter Your Query")

    default_text = st.session_state.get('selected_example', '')
    natural_language = st.text_area(
        "Describe what you want to query in natural language",
        value=default_text,
        height=100,
        placeholder="e.g., Show all students with GPA above 3.5",
        key="nl_input"
    )

    # Generate SQL button
    col_gen, col_clear = st.columns([4, 1])

    with col_gen:
        generate_button = st.button("🔄 Generate SQL", type="primary", use_container_width=True)

    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.generated_sql = None
            st.session_state.execution_results = None
            st.session_state.selected_example = ''
            st.rerun()

    # Generate SQL
    if generate_button and natural_language.strip():
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first from the sidebar")
        else:
            with st.spinner("🤖 Generating SQL query..."):
                result = generate_sql_mock(
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
                    'database': st.session_state.current_database
                })

    # Display generated SQL
    if st.session_state.generated_sql:
        st.markdown("---")
        st.markdown("## 📝 Generated SQL")

        result = st.session_state.generated_sql

        # Validation status
        is_valid, validation_msg = validate_sql(result['sql'])

        if is_valid:
            st.markdown(
                f'<span class="status-badge status-success">✓ Valid SQL</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="status-badge status-error">✗ Invalid SQL</span>',
                unsafe_allow_html=True
            )
            st.error(validation_msg)

        # Display SQL with syntax highlighting
        formatted_sql = format_sql_with_syntax_highlighting(result['sql'])
        st.code(formatted_sql, language='sql', line_numbers=True)

        # Generation metrics
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with metric_cols[1]:
            st.metric("Generation Time", f"{result['generation_time']:.2f}s")
        with metric_cols[2]:
            st.metric("Tokens", result['tokens_generated'])
        with metric_cols[3]:
            st.metric("Beam Size", result['beam_size'])

        # Execute query button
        st.markdown("---")

        if st.button("▶️ Execute Query", type="primary", use_container_width=True, disabled=not is_valid):
            with st.spinner("⚡ Executing query..."):
                start_time = time.time()
                success, df, error = execute_sql_mock(result['sql'], st.session_state.current_database)
                execution_time = time.time() - start_time

                st.session_state.execution_results = {
                    'success': success,
                    'dataframe': df,
                    'error': error,
                    'execution_time': execution_time
                }

    # Display execution results
    if st.session_state.execution_results:
        st.markdown("---")
        st.markdown("## 📊 Query Results")

        results = st.session_state.execution_results

        if results['success']:
            st.markdown(
                f'<span class="status-badge status-success">✓ Query Executed Successfully</span>',
                unsafe_allow_html=True
            )

            # Display results table
            df = results['dataframe']
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # Result metrics
            metric_cols = st.columns(3)

            with metric_cols[0]:
                st.metric("Rows Returned", len(df))
            with metric_cols[1]:
                st.metric("Columns", len(df.columns))
            with metric_cols[2]:
                st.metric("Execution Time", f"{results['execution_time']:.3f}s")

            # Export options
            st.markdown("### 💾 Export Results")

            export_cols = st.columns(2)

            with export_cols[0]:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with export_cols[1]:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.markdown(
                f'<span class="status-badge status-error">✗ Query Failed</span>',
                unsafe_allow_html=True
            )
            st.error(results['error'])

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 2rem 0;">
        <p>Built with Streamlit • Powered by T5/BART Transformers</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            🚀 Text-to-SQL System v1.0.0 | Team: Eba, Mati, Nahom
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
