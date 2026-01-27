# Quick Start Guide

## 🚀 Run the App (30 seconds)

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py

# 3. Open browser
# → http://localhost:8501
```

---

## 📁 File Structure

```
proj/
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Python dependencies
├── STREAMLIT_GUIDE.md         ← Full UI documentation
├── UI_COMPONENTS.md           ← Design system reference
├── MODEL_INTEGRATION.md       ← How to connect real models
├── QUICK_START.md             ← This file
│
├── data/
│   ├── databases/             ← SQLite database files (.db)
│   ├── spider/                ← Training dataset
│   └── preprocessed/          ← Processed training data
│
├── models/
│   ├── t5_finetuned/         ← Fine-tuned T5 model
│   └── tokenizer/            ← Tokenizer artifacts
│
└── src/
    ├── model_inference.py     ← Model wrapper (to be created)
    ├── database_manager.py    ← DB connections (to be created)
    ├── sql_validator.py       ← SQL validation
    └── utils.py               ← Helper functions
```

---

## 🎯 Current Status

### ✅ Implemented (Ready to Use)

- **Professional UI**: Dark theme with emerald accents, responsive layout
- **Database Schemas**: 3 sample schemas (university, company, ecommerce)
- **Example Queries**: 16+ pre-built queries across 4 categories
- **SQL Validation**: Syntax checking with sqlparse
- **Mock Generation**: Keyword-based SQL generation (demo mode)
- **Mock Execution**: Sample data display
- **Export Features**: CSV and JSON downloads
- **Session State**: Query history, caching

### 🔧 To Be Implemented (Integration Required)

- **Model Loading**: Replace `load_model_placeholder()` with real T5 model
- **SQL Generation**: Replace `generate_sql_mock()` with model inference
- **Database Execution**: Replace `execute_sql_mock()` with real SQLite queries
- **Sample Databases**: Create `.db` files in `data/databases/`

See `MODEL_INTEGRATION.md` for step-by-step integration guide.

---

## 🎨 UI Features

### Sidebar

```
⚙️ Configuration
  ├─ Database Selector (university/company/ecommerce)
  ├─ Temperature Slider (0.0 - 1.0)
  ├─ Beam Size Slider (1 - 10)
  ├─ Load Model Button
  ├─ Model Status Badge
  └─ Query History Counter
```

### Main Panel

```
🗄️ Header Hero
  ↓
📊 Database Schema (expandable)
  ↓
💡 Example Queries (4 categories, 16+ examples)
  ↓
🎯 Natural Language Input (text area)
  ↓
🔄 Generate SQL Button
  ↓
📝 Generated SQL Code Block
  ├─ Syntax Highlighting
  ├─ Validation Badge
  └─ Generation Metrics
  ↓
▶️ Execute Query Button
  ↓
📊 Results Table
  ├─ Interactive Dataframe
  ├─ Execution Metrics
  └─ Export Buttons (CSV/JSON)
```

---

## 🎮 How to Use

### Demo Mode (Current - No Model Required)

1. **Launch app**: `streamlit run app.py`
2. **Select database**: Choose from sidebar dropdown
3. **Try example query**: Click any example query button
4. **Generate SQL**: Click "🔄 Generate SQL" button
5. **View result**: See generated SQL with syntax highlighting
6. **Execute**: Click "▶️ Execute Query" to see mock data
7. **Export**: Download results as CSV or JSON

### Production Mode (After Integration)

1. **Load model**: Click "🔄 Load Model" in sidebar (first time only)
2. **Wait for loading**: Model loads once, cached for session
3. **Enter query**: Type natural language or use examples
4. **Generate**: AI generates SQL based on selected database schema
5. **Validate**: Automatic syntax validation
6. **Execute**: Run query on actual SQLite database
7. **Analyze**: View real results, export data

---

## 🎨 Visual Design

### Color Scheme

```
Primary:   Emerald Green (#10b981)
Dark BG:   Navy (#0f172a)
Cards:     Slate (#1e293b)
Success:   Green (#22c55e)
Error:     Red (#ef4444)
Warning:   Orange (#f59e0b)
```

### Unique Elements

- **Gradient hero header** with emerald background
- **Status badges** (pill-shaped, colored by state)
- **Metric cards** with hover lift animation
- **Example cards** with left border accent and slide effect
- **SQL code blocks** with dark theme and line numbers
- **Primary buttons** with gradient + glow effect
- **Results table** with rounded corners and shadow

### Animations

- Buttons: Hover → lift up 2px + enhanced glow
- Cards: Hover → lift up 2px
- Examples: Hover → slide right 4px + border color change
- Spinner: Emerald color override
- All transitions: 0.2s smooth

---

## 📊 Example Queries by Category

### Basic Queries (SELECT + WHERE)
- "Show all students with GPA above 3.5"
- "List all courses offered in the Computer Science department"
- "Find students who are older than 20 years"
- "Display all employees hired after 2020"

### Aggregations (GROUP BY)
- "What is the average salary by department?"
- "How many students are enrolled in each major?"
- "Count the number of courses per instructor"
- "Calculate the total revenue by product category"

### Joins (Multi-table)
- "List all professors and their courses"
- "Show students and their enrolled courses"
- "Display employees with their department names"
- "Find orders with customer details"

### Complex Queries (Advanced)
- "Which departments have more than 10 employees?"
- "Find the top 5 highest-paid employees in each department"
- "Show courses with no enrolled students"
- "List students who have taken all required courses"

---

## 🔧 Configuration Options

### Model Settings (Sidebar)

**Temperature** (0.0 - 1.0)
- **0.0 - 0.2**: Deterministic, consistent output
- **0.3 - 0.5**: Balanced (recommended)
- **0.6 - 1.0**: Creative, varied output

**Beam Size** (1 - 10)
- **1**: Greedy decoding (fastest)
- **4**: Balanced quality/speed (recommended)
- **8-10**: Best quality (slower)

### Database Schemas

**University** (default)
- Tables: students, professors, courses, enrollments
- Use case: Academic queries

**Company**
- Tables: employees, departments, projects, assignments
- Use case: Corporate analytics

**E-commerce**
- Tables: customers, products, orders, order_items
- Use case: Sales and inventory queries

---

## 📦 Dependencies

Already in `requirements.txt`:

```
torch>=2.0.0              # Deep learning framework
transformers>=4.35.0      # Hugging Face models
streamlit>=1.28.0         # Web interface
pandas>=2.0.0             # Data manipulation
sqlparse>=0.4.4           # SQL parsing/validation
datasets>=2.14.0          # Dataset loading
accelerate>=0.24.0        # Model optimization
```

---

## 🐛 Troubleshooting

### App won't start

```bash
# Check Streamlit version
streamlit --version

# Reinstall dependencies
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

### Port already in use

```bash
# Use different port
streamlit run app.py --server.port 8502
```

### CSS not loading

```bash
# Hard refresh browser
Ctrl+Shift+R (Windows/Linux)
Cmd+Shift+R (Mac)
```

### Session state issues

```bash
# Clear browser cookies
# Or restart Streamlit server
```

---

## 🚀 Next Steps

1. **Test Demo Mode**: Run app and explore UI features
2. **Create Sample Databases**: See `MODEL_INTEGRATION.md` section 5
3. **Integrate Model**: Follow `MODEL_INTEGRATION.md` steps 1-4
4. **Test Production Mode**: End-to-end with real model
5. **Deploy**: Streamlit Cloud or custom server

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `STREAMLIT_GUIDE.md` | Complete UI feature documentation |
| `UI_COMPONENTS.md` | Design system and component library |
| `MODEL_INTEGRATION.md` | Step-by-step model integration |
| `QUICK_START.md` | This file (30-second setup) |
| `README.md` | Project overview and architecture |

---

## 💡 Pro Tips

1. **Use examples first**: Click example queries to understand expected input format
2. **Check schema**: Expand schema details to see available tables/columns
3. **Start simple**: Test basic queries before complex ones
4. **Adjust beam size**: Lower for speed during dev, higher for quality in production
5. **Monitor history**: Sidebar shows query count, clear when needed
6. **Export results**: Use CSV for Excel, JSON for APIs

---

## 🎯 Success Criteria

After running the app, you should see:

✅ Professional dark theme with emerald accents
✅ Responsive layout (works on mobile too)
✅ Clickable example queries
✅ Syntax-highlighted SQL code blocks
✅ Status badges for validation
✅ Interactive results table
✅ CSV/JSON export buttons
✅ Smooth animations and transitions
✅ Query history tracking

---

**App ready to run! Launch with: `streamlit run app.py`**

🚀 **Demo Mode**: Works immediately (mock data)
🔌 **Production Mode**: Requires model integration (see `MODEL_INTEGRATION.md`)
