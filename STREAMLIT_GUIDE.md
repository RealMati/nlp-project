# Streamlit UI - Quick Start Guide

## 🚀 Launch the Application

```bash
# From project root
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ✨ Key Features

### 1. **Professional Design**
- Custom gradient theme (emerald green primary colors)
- Glassmorphic UI elements with shadows and hover effects
- Responsive layout that works on all screen sizes
- Dark theme optimized for developer aesthetics

### 2. **Sidebar Configuration**
- **Database Selection**: Choose between university, company, or ecommerce schemas
- **Model Settings**:
  - Temperature slider (0.0-1.0) for generation randomness
  - Beam size slider (1-10) for search quality
- **Model Status**: Load model and view version info
- **Query History**: Track recent queries with clear functionality

### 3. **Main Interface Flow**

```
Schema Display → Example Queries → Natural Language Input → Generate SQL → Validate → Execute → View Results → Export
```

### 4. **Example Queries**
Pre-built query templates organized by category:
- **Basic Queries**: Simple SELECT statements with WHERE clauses
- **Aggregations**: GROUP BY with AVG, COUNT, SUM functions
- **Joins**: Multi-table queries with JOIN operations
- **Complex Queries**: Advanced queries with subqueries and filters

Click any example to auto-fill the input box.

### 5. **SQL Generation**
- Real-time generation with loading spinner
- Confidence score display
- Token count and generation time metrics
- Syntax validation with visual badges (✓ Valid / ✗ Invalid)
- Formatted SQL with syntax highlighting

### 6. **Query Execution**
- Execute button (disabled if SQL is invalid)
- Results displayed in interactive dataframe
- Execution metrics (rows, columns, execution time)
- Export to CSV or JSON with timestamped filenames

### 7. **Session State Management**
- Query history persists during session
- Generated SQL cached between reruns
- Model loaded state preserved

## 🎨 UI Components

### Status Badges
```
✓ Valid SQL      → Green badge
✗ Invalid SQL    → Red badge
✓ Model Ready    → Success badge
```

### Metric Cards
Gradient cards with hover effects showing:
- Confidence percentage
- Generation time
- Token count
- Beam size
- Rows returned
- Execution time

### Interactive Elements
- Clickable example queries with hover transform
- Primary/secondary button styling
- Expandable schema details
- Syntax-highlighted code blocks

## 🔧 Customization

### Colors (in app.py CSS)
```css
--primary: #10b981       /* Emerald green */
--primary-dark: #059669  /* Dark emerald */
--secondary: #3b82f6     /* Blue */
--success: #22c55e       /* Green */
--error: #ef4444         /* Red */
--warning: #f59e0b       /* Orange */
```

### Layout
- Wide mode enabled by default
- 2:1 column ratio for main content
- Sidebar expanded by default
- Responsive grid for metrics

## 📊 Database Schemas

### University (Default)
```
students: id, name, age, gpa, major, enrollment_year
professors: id, name, department, hire_date, salary
courses: id, course_name, course_code, credits, professor_id, department
enrollments: id, student_id, course_id, semester, grade
```

### Company
```
employees: id, name, position, department_id, salary, hire_date
departments: id, name, location, budget, manager_id
projects: id, name, start_date, end_date, budget, department_id
assignments: id, employee_id, project_id, role, hours
```

### E-commerce
```
customers: id, name, email, address, registration_date
products: id, name, category, price, stock, supplier_id
orders: id, customer_id, order_date, total_amount, status
order_items: id, order_id, product_id, quantity, price
```

## 🔌 Integration Points (TODO)

The app currently uses mock functions. Replace these with actual implementations:

### 1. Model Loading
```python
@st.cache_resource
def load_model_placeholder():
    # TODO: Replace with actual model loading
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    model = T5ForConditionalGeneration.from_pretrained("./models/t5_finetuned")
    tokenizer = T5Tokenizer.from_pretrained("./models/tokenizer")

    return {"model": model, "tokenizer": tokenizer}
```

### 2. SQL Generation
```python
def generate_sql_mock(natural_language, database, temperature, beam_size):
    # TODO: Replace with actual inference
    # Use model from load_model_placeholder()
    # Concatenate schema with natural language
    # Generate SQL with beam search
    pass
```

### 3. SQL Execution
```python
def execute_sql_mock(sql, database):
    # TODO: Connect to actual SQLite database
    import sqlite3

    conn = sqlite3.connect(f"./data/databases/{database}.db")
    df = pd.read_sql_query(sql, conn)
    conn.close()

    return True, df, None
```

## 🎯 Testing Checklist

- [ ] Load model successfully
- [ ] Select different databases
- [ ] Adjust temperature and beam size
- [ ] Click example queries
- [ ] Generate SQL from natural language
- [ ] Validate SQL syntax
- [ ] Execute query
- [ ] View results table
- [ ] Export to CSV
- [ ] Export to JSON
- [ ] Check query history
- [ ] Clear history
- [ ] Test responsive layout
- [ ] Verify hover effects
- [ ] Check loading spinners
- [ ] Test error states

## 📱 Responsive Design

The UI is optimized for:
- Desktop (1920x1080+): Full wide layout with side-by-side columns
- Laptop (1366x768): Optimized metric cards
- Tablet (768x1024): Stacked layout
- Mobile (375x667): Single column, touch-friendly buttons

## 🚨 Error Handling

The app handles:
- Invalid SQL syntax (pre-execution validation)
- Empty queries (button disabled states)
- Model not loaded (warning messages)
- Execution failures (error badges with messages)
- Empty results (graceful display)

## 💡 Pro Tips

1. **Faster Iteration**: Lower beam size for faster generation during development
2. **Better Quality**: Higher beam size (8-10) for production-quality SQL
3. **Deterministic Output**: Set temperature to 0.1 for consistent results
4. **Creative Queries**: Set temperature to 0.5-0.7 for more varied SQL
5. **Quick Testing**: Use example queries to test different SQL patterns
6. **Export Results**: Download results as CSV for further analysis

## 📚 Next Steps

1. Integrate actual T5/BART model (replace mock functions)
2. Connect to real SQLite databases in `./data/databases/`
3. Add query explanation feature (SQL → Natural Language)
4. Implement query refinement (multi-turn conversations)
5. Add authentication for multi-user support
6. Deploy to Streamlit Cloud or custom server

---

**Built with Streamlit 1.28+ | Requires Python 3.9+**
