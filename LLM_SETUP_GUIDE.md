# LLM Text-to-SQL Setup Guide

Use this guide to configure and run the new Retrieval-Augmented LLM Text-to-SQL system.

## 1. Environment Setup
The installation of dependencies is currently running in the background. Ensure it completes before proceeding.

Required packages:
- `openai`
- `sentence-transformers`
- `chromadb`
- `python-dotenv`
- (And standard project requirements like `torch`, `streamlit`)

## 2. API Configuration
1.  Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and add your OpenAI API Key:
    ```
    LLM_API_KEY=sk-proj-...
    LLM_PROVIDER=openai
    LLM_MODEL_NAME=gpt-4o
    ```

## 3. Build Vector Store
1.  **Download Spider Data** (if not present):
    ```bash
    ~/.venvs/nlp-project/bin/pip install datasets # Temp install
    ~/.venvs/nlp-project/bin/python3 download_via_hf.py
    ```

2.  **Populate Embeddings**:
    ```bash
    ~/.venvs/nlp-project/bin/python3 src/llm_integration/build_vector_store.py
    ```
    *Note: If using Gemini Free Tier, this script runs slowly (~20 mins) to respect rate limits.*

## 4. Run the Application
Start the Streamlit app:

```bash
~/.venvs/nlp-project/bin/streamlit run app.py
```
- In the sidebar, under **Model Selection**, choose **LLM (In-Context Learning)**.
- Load the model and start querying!

## 5. Evaluation
To evaluate the LLM approach on the Spider development set:

```bash
~/.venvs/nlp-project/bin/python3 evaluate_llm.py
```
Results will be saved to `llm_evaluation_results.json`.
