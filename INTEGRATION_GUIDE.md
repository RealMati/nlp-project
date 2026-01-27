# Text-to-SQL System - Integration & Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Training the Model](#training-the-model)
4. [Integrating Model with UI](#integrating-model-with-ui)
5. [Deployment Options](#deployment-options)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Fastest path from zero to running application:**

```bash
# 1. Setup environment
./quickstart.sh

# 2. Download dataset (choose 'y' when prompted)
python src/data_preparation.py --download

# 3. Train model (lightweight test run)
python src/model_training.py \
    --model_name t5-small \
    --epochs 3 \
    --output_dir ./models/t5_test

# 4. Launch UI
streamlit run app.py
```

---

## Step-by-Step Setup

### 1. Environment Preparation

```bash
# Clone/navigate to project
cd /Users/leul/Documents/NLP/proj

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

### 2. Data Preparation

**Option A: Use Hugging Face Hub (Recommended)**
```bash
python src/data_preparation.py --download
```

**Option B: Manual Spider Download**
```bash
# Download from https://yale-lily.github.io/spider
wget https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m
unzip spider.zip -d data/spider/
```

**Preprocess data:**
```python
from src.data_preparation import SpiderDataProcessor

processor = SpiderDataProcessor(
    data_dir="./data/spider",
    schema_format="verbose"  # or "compact", "natural"
)

# Load dataset
dataset = processor.load_spider_from_huggingface()

# Tokenize for training
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenized = processor.create_tokenized_dataset(dataset, tokenizer)

# Save preprocessed
tokenized.save_to_disk("./data/preprocessed/spider_t5")
```

### 3. Create Sample Database (for testing)

```bash
python src/demo_database_system.py
```

This creates `data/databases/sample_university.db` with realistic academic data.

---

## Training the Model

### Basic Training (T5-Small for quick testing)

```bash
python src/model_training.py \
    --model_name t5-small \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir ./models/t5_small_finetuned \
    --eval_steps 500 \
    --save_steps 1000
```

**Expected time:** ~2-3 hours on GPU, ~12 hours on CPU

### Production Training (T5-Base)

```bash
python src/model_training.py \
    --model_name t5-base \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --fp16 \
    --output_dir ./models/t5_base_finetuned \
    --eval_steps 500 \
    --save_steps 1000 \
    --early_stopping_patience 3
```

**Expected time:** ~12-16 hours on GPU (V100/A100)

### Advanced Training (T5-Large)

```bash
python src/model_training.py \
    --model_name t5-large \
    --epochs 15 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --fp16 \
    --output_dir ./models/t5_large_finetuned \
    --max_input_length 768 \
    --max_target_length 512
```

**Hardware requirements:**
- GPU: 24GB+ VRAM (A100, RTX 3090)
- RAM: 32GB+
- Storage: 50GB+ for checkpoints

**Training parameters explained:**
- `--batch_size`: Samples per GPU per step (reduce if OOM)
- `--gradient_accumulation_steps`: Effective batch size = batch_size × this value
- `--fp16`: Use mixed precision (2x faster, half memory)
- `--warmup_ratio`: Gradual learning rate increase (prevents early divergence)
- `--early_stopping_patience`: Stop if no eval improvement for N evaluations

### Monitor Training

```bash
# Real-time logs
tail -f models/t5_base_finetuned/trainer_log.txt

# TensorBoard (if enabled)
tensorboard --logdir models/t5_base_finetuned/runs
```

---

## Integrating Model with UI

### Step 1: Verify Model Training

```bash
python evaluate.py \
    --model_path ./models/t5_base_finetuned \
    --test_data ./data/spider/dev.json \
    --database_dir ./data/spider/database \
    --max_samples 100
```

**Expected output:**
```
Evaluation Results:
  Exact Match Accuracy: 68.5%
  Execution Accuracy: 75.2%
  Token Accuracy: 82.1%
```

### Step 2: Test Inference

```python
from src.model_inference import Text2SQLInference

inference = Text2SQLInference(
    model_path="./models/t5_base_finetuned",
    device="cuda"  # or "cpu"
)

result = inference.predict(
    question="Show all students with GPA above 3.5",
    db_path="./data/databases/sample_university.db"
)

print(result.sql)
print(f"Confidence: {result.confidence:.2%}")
```

### Step 3: Update Streamlit App

**Replace mock functions in `app.py`:**

```python
# OLD (lines ~180-200)
def generate_sql_mock(natural_language, database, temperature, beam_size):
    # Mock implementation
    return "SELECT * FROM students WHERE gpa > 3.5"

# NEW
from src.model_inference import Text2SQLInference, GenerationConfig

@st.cache_resource
def load_model():
    return Text2SQLInference(
        model_path="./models/t5_base_finetuned",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def generate_sql(natural_language, database, temperature, beam_size):
    model = load_model()

    # Get database schema
    db_path = f"./data/databases/{database}.db"
    schema = get_database_schema(db_path)  # Implement this

    # Configure generation
    config = GenerationConfig(
        temperature=temperature,
        num_beams=beam_size,
        max_length=256
    )

    # Generate SQL
    result = model.predict(
        question=natural_language,
        db_path=db_path,
        validate=True,
        generation_config=config
    )

    return result.sql, result.confidence
```

**Integrate database execution:**

```python
# Replace execute_sql_mock (lines ~220-240)
from src.database_manager import DatabaseManager

def execute_sql(sql: str, database: str):
    try:
        db_path = f"./data/databases/{database}.db"
        db_manager = DatabaseManager(db_path, read_only=True)

        # Execute query
        result_df = db_manager.execute_query(sql, return_dataframe=True)

        # Get stats
        rows = len(result_df)
        columns = len(result_df.columns)

        return True, result_df, f"Query executed successfully. {rows} rows × {columns} columns"

    except Exception as e:
        return False, None, f"Execution error: {str(e)}"
```

### Step 4: Add Real Databases

```python
# Create company database
import sqlite3
import pandas as pd

conn = sqlite3.connect('./data/databases/company.db')

# Create tables
conn.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary REAL,
        hire_date TEXT
    )
''')

conn.execute('''
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT,
        budget REAL,
        manager_id INTEGER
    )
''')

# Insert sample data
employees_data = [
    (1, 'Alice Johnson', 'Engineering', 95000, '2020-01-15'),
    (2, 'Bob Smith', 'Sales', 75000, '2019-06-20'),
    # ... add more
]

conn.executemany('INSERT INTO employees VALUES (?,?,?,?,?)', employees_data)
conn.commit()
conn.close()
```

### Step 5: Launch Integrated App

```bash
streamlit run app.py
```

---

## Deployment Options

### Option 1: Docker (Recommended)

**Build image:**
```bash
docker build -t text2sql-app .
```

**Run container:**
```bash
docker run -p 8501:8501 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    text2sql-app
```

**Access:** http://localhost:8501

### Option 2: Docker Compose

```bash
docker-compose up -d
```

**Check logs:**
```bash
docker-compose logs -f
```

**Stop:**
```bash
docker-compose down
```

### Option 3: Cloud Deployment

#### Streamlit Cloud

1. Push code to GitHub
2. Connect repo at https://streamlit.io/cloud
3. Set environment variables
4. Deploy

**Note:** Large models may exceed free tier limits. Consider model compression.

#### AWS EC2

```bash
# Launch instance (g4dn.xlarge for GPU)
ssh -i key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install docker.io docker-compose -y

# Clone and run
git clone your-repo.git
cd your-repo
docker-compose up -d
```

#### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/text2sql

# Deploy
gcloud run deploy text2sql \
    --image gcr.io/PROJECT_ID/text2sql \
    --platform managed \
    --memory 4Gi \
    --timeout 300
```

### Option 4: Production Server Setup

**Using Nginx + Gunicorn (for Flask version):**

```bash
# Install Nginx
sudo apt install nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/text2sql

# Nginx config:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/text2sql /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## Performance Optimization

### Model Optimization

**1. Quantization (Reduce model size by 4x)**

```python
from transformers import AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("./models/t5_base_finetuned")

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save
quantized_model.save_pretrained("./models/t5_base_quantized")
```

**2. ONNX Export (Faster inference)**

```bash
pip install optimum[onnxruntime]

optimum-cli export onnx \
    --model ./models/t5_base_finetuned \
    --task text2text-generation \
    ./models/t5_base_onnx
```

**3. Model Distillation (Smaller, faster model)**

```python
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments

teacher = AutoModelForSeq2SeqLM.from_pretrained("./models/t5_base_finetuned")
student = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Train student to mimic teacher outputs
# (Implementation in src/model_distillation.py - to be created)
```

### Application Optimization

**1. Caching**

```python
import streamlit as st
from functools import lru_cache

@st.cache_resource
def load_model():
    return Text2SQLInference("./models/t5_base_finetuned")

@lru_cache(maxsize=128)
def get_schema(db_path):
    return DatabaseManager(db_path).get_schema()
```

**2. Batch Processing**

```python
# For multiple queries
results = model.predict_batch(
    questions=["query1", "query2", "query3"],
    schemas=[schema1, schema2, schema3],
    batch_size=8
)
```

**3. GPU Optimization**

```python
# Use mixed precision
inference = Text2SQLInference(
    model_path="./models/t5_base_finetuned",
    device="cuda",
    fp16=True  # Enable half precision
)
```

### Database Optimization

**1. Indexing**

```sql
-- Add indexes for common queries
CREATE INDEX idx_students_gpa ON students(gpa);
CREATE INDEX idx_students_major ON students(major);
CREATE INDEX idx_enrollments_student ON enrollments(student_id);
```

**2. Query Caching**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=256)
def execute_cached(sql_hash, db_path):
    sql = # retrieve original SQL
    return DatabaseManager(db_path).execute_query(sql)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) During Training

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
--batch_size 2 --gradient_accumulation_steps 8

# Use smaller model
--model_name t5-small

# Enable CPU offloading
--fp16 --gradient_checkpointing
```

#### 2. Slow Inference

**Problem:** Query generation takes >10 seconds

**Solutions:**
- Use quantized model (4x faster)
- Reduce beam search: `--num_beams 3` instead of 5
- Use GPU if available
- Enable model caching in Streamlit

#### 3. Low Accuracy on Custom Database

**Problem:** Model generates incorrect SQL for your database

**Solutions:**
- Fine-tune on custom examples (create training data from your DB)
- Use schema serialization format that matches training
- Add few-shot examples in prompt
- Increase model size (t5-base → t5-large)

#### 4. SQL Validation Failures

**Error:** `ValidationError: Table 'xyz' not found in schema`

**Solutions:**
```python
# Ensure schema is properly loaded
db_manager = DatabaseManager(db_path)
schema = db_manager.get_schema(force_refresh=True)

# Check schema format matches training
processor = SpiderDataProcessor(schema_format="verbose")
```

#### 5. Streamlit App Crashes

**Error:** `BrokenPipeError` or app hangs

**Solutions:**
```bash
# Increase server limits
streamlit run app.py \
    --server.maxUploadSize 1000 \
    --server.maxMessageSize 1000 \
    --server.enableXsrfProtection false
```

#### 6. Docker Build Fails

**Error:** `ERROR: Could not find a version that satisfies the requirement torch`

**Solutions:**
```dockerfile
# Update Dockerfile base image
FROM python:3.11-slim

# Or install from specific source
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Debugging Tools

**1. Enable verbose logging**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**2. Profile model inference**

```python
import time

start = time.time()
result = model.predict(question, schema)
print(f"Inference time: {time.time() - start:.2f}s")
```

**3. Inspect tokenization**

```python
from src.data_preparation import SpiderDataProcessor

processor = SpiderDataProcessor()
input_text = processor.preprocess_example(example, schemas)
tokens = tokenizer(input_text["input_text"])
print(f"Token count: {len(tokens['input_ids'])}")
```

### Getting Help

- **GitHub Issues:** [Your repo URL]
- **Documentation:** See README.md, STREAMLIT_GUIDE.md
- **Spider Dataset:** https://yale-lily.github.io/spider
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers

---

## Production Checklist

Before deploying to production:

- [ ] Model trained and evaluated (>70% accuracy)
- [ ] All mock functions replaced with real implementations
- [ ] Database connections tested
- [ ] SQL validation enabled and tested
- [ ] Error handling implemented for all user inputs
- [ ] Rate limiting configured (if public)
- [ ] HTTPS enabled (for cloud deployment)
- [ ] Model caching implemented
- [ ] Database backups configured
- [ ] Monitoring/logging setup (Sentry, CloudWatch, etc.)
- [ ] Load testing completed
- [ ] Security review completed (SQL injection prevention)
- [ ] Documentation updated with deployment URL
- [ ] Demo video created (optional)

---

## Next Steps

1. **Train your first model:**
   ```bash
   python src/model_training.py --model_name t5-small --epochs 3
   ```

2. **Evaluate performance:**
   ```bash
   python evaluate.py --model_path ./models/t5_small_finetuned
   ```

3. **Integrate with UI:**
   - Update `app.py` with real model (see Step 3 above)
   - Add your own databases to `data/databases/`

4. **Deploy:**
   ```bash
   docker-compose up -d
   ```

5. **Monitor and iterate:**
   - Collect user feedback
   - Fine-tune on problematic queries
   - Add more example databases

---

**Built by:** Eba Adisu, Mati Milkessa, Nahom Garefo
**Project:** Text-to-SQL Translation System
**Version:** 1.0.0
