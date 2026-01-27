# Text-to-SQL System
**Natural Language to SQL Query Converter**

## Project Overview
A production-grade Text-to-SQL system that translates natural language questions into executable SQL queries using fine-tuned transformer models (T5/BART).

## Team Members
- Eba Adisu (UGR/2749/14)
- Mati Milkessa (UGR/0949/14)
- Nahom Garefo (UGR/6739/14)

## Features
- **Schema-Aware Generation**: Understands database structure to generate contextually accurate SQL
- **Multi-Operation Support**: SELECT, WHERE, JOINs, aggregations (COUNT, AVG, SUM, etc.)
- **Syntax Validation**: Pre-execution SQL validation to prevent errors
- **Interactive UI**: Real-time Streamlit interface for query testing
- **Performance Metrics**: Execution accuracy and exact-match evaluation

## Architecture
```
┌─────────────────┐
│  User Input     │ "Show me students with highest grades"
│  (Natural Lang) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  T5/BART Model  │ Fine-tuned on Spider/WikiSQL
│  + Schema       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQL Validator  │ Syntax checking & schema validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQLite Engine  │ Query execution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Results        │ Formatted table output
└─────────────────┘
```

## Tech Stack
- **Language**: Python 3.9+
- **Framework**: PyTorch
- **Models**: Hugging Face Transformers (T5-base, BART)
- **Dataset**: Spider (multi-domain, complex queries)
- **Database**: SQLite3
- **UI**: Streamlit
- **Libraries**: transformers, datasets, sqlparse, pandas, torch

## Project Structure
```
text-to-sql/
├── data/
│   ├── spider/              # Spider dataset
│   ├── databases/           # Sample SQLite databases
│   └── preprocessed/        # Processed training data
├── models/
│   ├── t5_finetuned/       # Fine-tuned model checkpoints
│   └── tokenizer/          # Tokenizer artifacts
├── src/
│   ├── data_preparation.py  # Dataset loading & preprocessing
│   ├── model_training.py    # Training pipeline
│   ├── model_inference.py   # Inference wrapper
│   ├── sql_validator.py     # SQL syntax validation
│   ├── database_manager.py  # SQLite connection & execution
│   └── utils.py             # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── app.py                   # Streamlit application
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd text-to-sql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Spider dataset
python src/data_preparation.py --download
```

## Usage

### Training
```bash
# Fine-tune T5 model on Spider dataset
python train.py \
  --model_name t5-base \
  --dataset spider \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 3e-5
```

### Evaluation
```bash
# Evaluate model performance
python evaluate.py \
  --model_path ./models/t5_finetuned \
  --test_data ./data/spider/dev.json
```

### Web Interface
```bash
# Launch Streamlit app
streamlit run app.py
```

## Model Performance

| Metric | Score |
|--------|-------|
| Execution Accuracy | TBD% |
| Exact Match Accuracy | TBD% |
| Component Match | TBD% |

## Example Queries

```
Input:  "Show me all students with GPA above 3.5"
Output: SELECT * FROM students WHERE gpa > 3.5

Input:  "What is the average salary by department?"
Output: SELECT department, AVG(salary) FROM employees GROUP BY department

Input:  "List professors and their courses"
Output: SELECT p.name, c.course_name FROM professors p JOIN courses c ON p.id = c.professor_id
```

## Challenges & Solutions

1. **Schema Awareness**: Concatenated schema information with query for context
2. **Complex JOINs**: Used Spider dataset with multi-table queries
3. **Validation**: Built SQL parser to catch syntax errors pre-execution
4. **Generalization**: Fine-tuned on diverse domains (academia, business, etc.)

## Future Enhancements
- [ ] Support for subqueries and CTEs
- [ ] Multi-turn conversations (query refinement)
- [ ] Support for UPDATE, DELETE, INSERT operations
- [ ] Integration with PostgreSQL/MySQL
- [ ] Query explanation (SQL → Natural Language)

## References
- [Spider Dataset](https://yale-lily.github.io/spider)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [BART Paper](https://arxiv.org/abs/1910.13461)

## License
MIT

---
**Built with J.A.R.V.I.S. orchestration**
