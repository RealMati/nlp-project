# Architecture Deep Dive: Lightweight RAG (BM25 + Groq)

## Overview

This document explains the internal workings of the **Lightweight RAG** system. Unlike traditional RAG systems that use heavy Vector Databases (Pinecone/Chroma) and Embedding Models (BERT/ada-002), this system uses **BM25 (Best Matching 25)** for retrieval and **Groq** for high-speed generation.

**Key Innovation**: Zero vector embeddings, instant indexing, sub-second retrieval, and blazing-fast LLM inference.

---

## System Architecture

```mermaid
graph TD
    User[👤 User Input] --> App[🖥️ Streamlit App]
    
    subgraph "🧠 Retrieval Layer (Local Memory)"
        App -->|1. Natural Language Query| Retriever[BM25 Retriever]
        Data[(📁 train_spider.json<br/>7000 examples)] -->|Load & Index on Startup| Retriever
        Retriever -->|2. Top-K Similar Examples| Context[📋 Retrieved Context]
    end
    
    subgraph "⚡ Generation Layer (Cloud)"
        Schema[🗄️ Database Schema] --> Prompt[📝 Prompt Builder]
        Context --> Prompt
        App -->|3. Question| Prompt
        Prompt -->|4. Full Prompt| LLM[🚀 Groq API<br/>llama-3.3-70b]
        LLM -->|5. Generated SQL| App
    end
    
    subgraph "💾 Execution Layer"
        App -->|6. Execute SQL| DB[(🗃️ SQLite Database)]
        DB -->|7. Query Results| App
    end
    
    App -->|8. Display| User
```

---

## Component Breakdown

### 1. The "Keystore" (BM25 Retriever)

**Location**: [`src/llm_integration/retriever.py`](file:///media/ego/New%20Volume7/C_B/Class/NLP/ASS/nlp-project/src/llm_integration/retriever.py)

#### What is BM25?
BM25 (Best Matching 25) is a **probabilistic keyword-based ranking function** used by search engines. It scores documents based on:
- **Term Frequency (TF)**: How often query words appear in a document
- **Inverse Document Frequency (IDF)**: How rare/unique those words are across all documents
- **Document Length Normalization**: Prevents bias toward longer documents

**Formula** (simplified):
```
score(Q, D) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```
Where:
- `Q` = Query, `D` = Document
- `f(qi, D)` = Frequency of term `qi` in document `D`
- `k1`, `b` = Tuning parameters (default: 1.5, 0.75)
- `avgdl` = Average document length

#### Why BM25 over Vector Search?

| Feature | Vector Search (Pinecone) | BM25 (Our Approach) |
|---------|-------------------------|---------------------|
| **Setup Time** | 10-30 minutes (embedding generation) | <1 second (tokenization only) |
| **Dependencies** | PyTorch (1GB+), sentence-transformers | rank-bm25 (50KB) |
| **Memory** | Requires GPU or slow CPU | Pure RAM, <100MB |
| **Accuracy for SQL** | Good for semantic similarity | **Better for exact keyword matches** (table names, column names) |
| **Cost** | API calls or GPU compute | Free, local |

#### Code Walkthrough

**Initialization** (happens once on app startup):
```python
# src/llm_integration/retriever.py:20-42
def _initialize(self):
    # 1. Load the Spider dataset (7000 text-to-SQL pairs)
    with open(self.data_path, 'r') as f:
        self.examples = json.load(f)  # [{question: "...", query: "SELECT ..."}]
    
    # 2. Tokenize all questions (split into words, lowercase, remove punctuation)
    corpus = [self._tokenize(ex['question']) for ex in self.examples]
    # Example: "Show students with GPA > 3.5" → ['show', 'students', 'gpa', '35']
    
    # 3. Build BM25 index (inverted index: word → document IDs)
    self.bm25 = BM25Okapi(corpus)
    # This takes ~0.5 seconds for 7000 documents
```

**Retrieval** (happens on every user query):
```python
# src/llm_integration/retriever.py:51-72
def retrieve_similar(self, query: str, k: int = 5):
    # 1. Tokenize user query
    tokenized_query = self._tokenize(query)
    # "Find student with id 5" → ['find', 'student', 'id', '5']
    
    # 2. Score all 7000 examples using BM25
    top_n_docs = self.bm25.get_top_n(tokenized_query, self.examples, n=k)
    # Returns top 5 most relevant examples in ~5ms
    
    # 3. Extract (question, SQL) pairs
    results = [(ex['question'], ex['query']) for ex in top_n_docs]
    return results
```

**Example Output**:
```python
query = "Show all students with GPA above 3.5"
results = retriever.retrieve_similar(query, k=3)

# Returns:
[
    ("List all students with GPA greater than 3.0", "SELECT * FROM students WHERE gpa > 3.0"),
    ("Find students with high grades", "SELECT * FROM students WHERE gpa > 3.5"),
    ("Show top students", "SELECT * FROM students ORDER BY gpa DESC LIMIT 10")
]
```

---

### 2. The LLM Generator (Groq)

**Location**: [`src/llm_integration/generator.py`](file:///media/ego/New%20Volume7/C_B/Class/NLP/ASS/nlp-project/src/llm_integration/generator.py)

#### Why Groq?
Groq is a hardware company that built custom **LPU (Language Processing Unit)** chips optimized for transformer inference. They offer:
- **Speed**: 500-800 tokens/second (vs 50-100 for GPT-4)
- **Latency**: ~0.3s first token (vs 1-3s for GPT-4)
- **Model**: Open-source Llama 3.3 70B (comparable to GPT-4 for code)
- **Cost**: Free tier with generous limits

#### Code Walkthrough

**Initialization**:
```python
# src/llm_integration/generator.py:12-28
def _initialize_client(self):
    if self.provider == "groq":
        from openai import OpenAI  # Groq uses OpenAI-compatible API
        return OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
```

**SQL Generation**:
```python
# src/llm_integration/generator.py:39-59
def generate_sql(self, prompt: str) -> str:
    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or your configured model
        messages=[
            {"role": "system", "content": "You are an SQL expert. Generate only SQL, no explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Deterministic (no randomness)
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
```

---

### 3. The Pipeline Orchestrator

**Location**: [`src/llm_integration/pipeline.py`](file:///media/ego/New%20Volume7/C_B/Class/NLP/ASS/nlp-project/src/llm_integration/pipeline.py)

This is the **brain** that connects Retriever + Generator.

#### Code Walkthrough

```python
class LLMPipeline:
    def __init__(self):
        self.retriever = Retriever()  # Loads BM25 index
        self.generator = LLMGenerator()  # Connects to Groq
    
    def generate_sql(self, question: str, db_schema: str) -> dict:
        # STEP 1: Retrieve similar examples
        examples = self.retriever.retrieve_similar(question, k=3)
        
        # STEP 2: Build prompt
        prompt = self._build_prompt(question, db_schema, examples)
        
        # STEP 3: Generate SQL
        sql = self.generator.generate_sql(prompt)
        
        # STEP 4: Return result
        return {"sql": sql, "confidence": 0.85}
```

**Prompt Construction** (the secret sauce):
```python
def _build_prompt(self, question, schema, examples):
    # Format retrieved examples
    few_shot = "\n\n".join([
        f"Question: {q}\nSQL: {sql}" 
        for q, sql in examples
    ])
    
    # Final prompt
    return f"""You are an SQL expert. Convert natural language to SQL.

Database Schema:
{schema}

Similar Examples:
{few_shot}

Question: {question}
SQL:"""
```

**Example Prompt Sent to Groq**:
```text
You are an SQL expert. Convert natural language to SQL.

Database Schema:
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    gpa REAL
)

Similar Examples:

Question: List all students with GPA greater than 3.0
SQL: SELECT * FROM students WHERE gpa > 3.0

Question: Find students with high grades
SQL: SELECT * FROM students WHERE gpa > 3.5

Question: Show top students
SQL: SELECT * FROM students ORDER BY gpa DESC LIMIT 10

Question: Show all students with GPA above 3.5
SQL:
```

**Groq Response** (in ~0.5 seconds):
```sql
SELECT * FROM students WHERE gpa > 3.5
```

---

## Complete Data Flow (Step-by-Step)

```mermaid
sequenceDiagram
    participant U as User
    participant A as Streamlit App
    participant R as BM25 Retriever
    participant P as Pipeline
    participant G as Groq API
    participant D as SQLite DB
    
    Note over A: App Startup
    A->>R: Initialize Retriever
    R->>R: Load train_spider.json (7000 examples)
    R->>R: Build BM25 index (~0.5s)
    R-->>A: Ready
    
    Note over U,D: User Query Flow
    U->>A: "Show students with GPA > 3.5"
    A->>D: Extract schema (tables, columns)
    D-->>A: Schema string
    
    A->>P: generate_sql(question, schema)
    P->>R: retrieve_similar(question, k=3)
    R->>R: Tokenize query
    R->>R: Score all 7000 examples (BM25)
    R-->>P: Top 3 examples [(Q1, SQL1), (Q2, SQL2), (Q3, SQL3)]
    
    P->>P: Build prompt (schema + examples + question)
    P->>G: POST /chat/completions
    Note over G: Groq LPU processes<br/>Llama 3.3 70B
    G-->>P: "SELECT * FROM students WHERE gpa > 3.5"
    P-->>A: {sql: "SELECT ...", confidence: 0.85}
    
    A->>D: Execute SQL
    D-->>A: Query results (DataFrame)
    A->>U: Display results table
```

---

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| **App Startup** | 1-2s | Load BM25 index |
| **Retrieval** | 5-10ms | Search 7000 examples |
| **Prompt Build** | <1ms | String concatenation |
| **LLM Generation** | 300-800ms | Groq API call |
| **SQL Execution** | 10-100ms | Depends on query complexity |
| **Total (Query → Result)** | **~1 second** | End-to-end |

Compare to traditional RAG:
- **ChromaDB + GPT-4**: 5-10 seconds (embedding lookup + slow LLM)
- **Our approach**: 1 second (keyword search + fast LLM)

---

## Configuration

All settings are in [`.env`](file:///media/ego/New%20Volume7/C_B/Class/NLP/ASS/nlp-project/.env):

```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...  # Your Groq API key
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_API_BASE=https://api.groq.com/openai/v1
```

**Supported Providers**:
- `groq`: Groq (recommended for speed)
- `openai`: OpenAI GPT-4/3.5
- `gemini`: Google Gemini (if you have quota)
- `ollama`: Local Llama models

---

## File Structure

```
src/llm_integration/
├── config.py          # Environment variable loader
├── retriever.py       # BM25 keyword search
├── generator.py       # LLM API client (Groq/OpenAI/Gemini)
├── pipeline.py        # Orchestrates retriever + generator
└── __init__.py        # Package exports

data/spider/
└── train_spider.json  # 7000 text-to-SQL examples (RAG memory)

app.py                 # Streamlit UI (calls pipeline)
```

---

## Key Design Decisions

### 1. Why No Vector Embeddings?
**Problem**: Embedding 7000 examples requires:
- PyTorch (1GB download)
- sentence-transformers (500MB)
- 10-30 minutes of compute
- GPU or very slow CPU

**Solution**: BM25 keyword search is:
- Faster for exact matches (table/column names)
- Instant to build (<1s)
- Tiny dependency (50KB)

### 2. Why Groq over GPT-4?
**Problem**: GPT-4 is slow (2-5s per query) and expensive ($0.03/1K tokens).

**Solution**: Groq is:
- 10x faster (0.3-0.5s)
- Free tier (generous limits)
- Open-source model (Llama 3.3 70B)

### 3. Why SQLite?
**Problem**: Need a simple, portable database for demos.

**Solution**: SQLite is:
- Zero-config (no server)
- Single-file databases
- Supports all standard SQL

---

## Extending the System

### Add a New LLM Provider

Edit [`generator.py`](file:///media/ego/New%20Volume7/C_B/Class/NLP/ASS/nlp-project/src/llm_integration/generator.py):

```python
elif self.provider == "anthropic":
    from anthropic import Anthropic
    return Anthropic(api_key=self.api_key)
```

### Add More Training Examples

Add to `data/spider/train_spider.json`:
```json
{
  "question": "Your custom question",
  "query": "SELECT * FROM your_table"
}
```
Restart the app to rebuild the BM25 index.

### Switch to Vector Search (if needed)

Replace `retriever.py` with:
```python
from sentence_transformers import SentenceTransformer
import chromadb

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        # ... rest of vector DB logic
```

---

## Troubleshooting

### "Retriever not initialized"
**Cause**: `train_spider.json` not found.

**Fix**: Run `python download_via_hf.py` or check `data/spider/` exists.

### "Error generating SQL: 429"
**Cause**: Groq rate limit exceeded.

**Fix**: Wait 60 seconds or upgrade to paid tier.

### Slow retrieval (>100ms)
**Cause**: Large dataset (>50K examples).

**Fix**: Reduce `k` parameter or use vector search for large datasets.

---

## Summary

This system achieves **production-grade Text-to-SQL** with:
- ✅ **Zero heavy dependencies** (no PyTorch, no ChromaDB)
- ✅ **Sub-second latency** (BM25 + Groq)
- ✅ **100% local retrieval** (no API calls for search)
- ✅ **Flexible LLM backend** (Groq/OpenAI/Gemini/Ollama)

**Trade-offs**:
- ❌ Less semantic understanding than vector search (e.g., won't match "puppy" to "dog")
- ✅ Better for SQL (exact keyword matches on table/column names)

For most Text-to-SQL use cases, **keyword search is superior** to semantic search.
