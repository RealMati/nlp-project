import json
import re
import torch
import sys
import os
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from natsql2sql.natsql_parser import create_sql_from_natSQL
from natsql2sql.natsql2sql import Args

MODEL_PATH = "./t5-natsql-adapter-final2"  # Your local folder
# MODEL_PATH = "google/flan-t5-small"
BASE_MODEL_ID = "google/flan-t5-small"   # Fallback for tokenizer

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
except OSError:
    print("Error loading local model. Please ensure 'model.safetensors' is in the folder.")
    sys.exit(1)
    

# Load Tokenizer (Try local first, then fallback to internet)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except:
    print("Local tokenizer files missing. Downloading standard tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)


# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")
model.to(device)
model.eval()

# Load Spider Data
print("Loading Spider Dev Data...")
with open('data/dev.json', 'r') as f:
    dev_data = json.load(f)
with open('data/tables.json', 'r') as f:
    schema_tables = json.load(f)

db_schemas = {t['db_id']: t for t in schema_tables}

# Load NatSQL table metadata (required for NatSQL -> SQL conversion)
tables_for_natsql_path = None
for candidate in [
    'NatSQLv1_6_1/tables_for_natsql.json',
    'NatSQLv1_6/tables_for_natsql.json',
    'tables_for_natsql.json',
]:
    if os.path.exists(candidate):
        tables_for_natsql_path = candidate
        break

if tables_for_natsql_path:
    with open(tables_for_natsql_path, 'r') as f:
        natsql_tables = json.load(f)
    natsql_table_dict = {t['db_id']: t for t in natsql_tables}
    print(f"Using NatSQL table metadata from: {tables_for_natsql_path}")
else:
    natsql_table_dict = {}
    print("⚠️  NatSQL table metadata not found. NatSQL → SQL conversion may be skipped.")


def serialize_schema(db_schema):
    """Converts schema dictionary to the string format the model expects."""
    text_parts = []
    table_names = db_schema['table_names_original']
    column_names = db_schema['column_names_original']
    table_to_cols = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0: table_to_cols[table_idx].append(col_name)
    for table_idx, table_name in enumerate(table_names):
        text_parts.append(f"{table_name} : {' , '.join(table_to_cols[table_idx])}")
    return " | ".join(text_parts)



# 4. GENERATION LOOP
# ==========================================
predicted_sqls = []
predicted_rows = []

max_questions = 10
subset = dev_data[:max_questions]

print(f"Generating SQL for {len(subset)} questions...")

for item in tqdm(subset):
    question = item['question']
    db_id = item['db_id']
    
    # 1. Format Input
    schema_text = serialize_schema(db_schemas[db_id])
    input_text = f"Translate to NatSQL: {question} | {schema_text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # 2. Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            num_beams=2 # Use 2 or 4. 2 is faster.
        )
    
    natsql_pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if len(predicted_sqls) < 3:
        print(f"\n[DEBUG] db_id={db_id}")
        print(f"[DEBUG] natsql_pred: {repr(natsql_pred)}")
        print(f"[DEBUG] length: {len(natsql_pred)}")
        if not natsql_pred:
            token_ids = outputs[0][:10].tolist()
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            print(f"[DEBUG] first_token_ids: {token_ids}")
            print(f"[DEBUG] first_token_strs: {token_strs}")
    
    # 3. Convert NatSQL -> SQL (NatSQL official parser)
    try:
        table_json = natsql_table_dict.get(db_id)
        if not table_json:
            raise ValueError(f"Missing NatSQL table metadata for db_id={db_id}")

        natsql2sql_args = Args()
        remove_groupby_from_natsql = False
        if remove_groupby_from_natsql:
            natsql2sql_args.not_infer_group = False
        else:
            natsql2sql_args.not_infer_group = True

        db_path = os.path.join('data', 'database', db_id, f"{db_id}.sqlite")
        final_sql, _, _ = create_sql_from_natSQL(
            natsql_pred,
            db_id,
            db_path,
            table_json,
            sq=None,
            remove_values=False,
            remove_groupby_from_natsql=remove_groupby_from_natsql,
            args=natsql2sql_args,
        )
        if not final_sql:
            final_sql = natsql_pred
    except Exception:
        # Fallback if conversion fails
        final_sql = natsql_pred 
        
    final_sql = final_sql.strip() if final_sql else ""
    predicted_sqls.append(final_sql)
    predicted_rows.append((question, final_sql))


output_file = 'predicted_sql.txt'
gold_output_file = 'dev_gold_with_db.sql'
metrics_output_file = 'predicted_metrics.json'
with open(output_file, 'w') as f:
    for question, sql in predicted_rows:
        f.write(f"{question}\t{sql}\n")

print(f"\n✅ Predictions saved to {output_file}")

# Build a gold file with db_id for evaluation (required by evaluation.py)
with open(gold_output_file, 'w') as f:
    for item in subset:
        gold_sql = item.get('query', '').strip()
        db_id = item.get('db_id', '').strip()
        if gold_sql and db_id:
            f.write(f"{gold_sql}\t{db_id}\n")

def normalize_sql(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

# Build a metrics JSON with question, prediction, gold, and exact-match metrics
metrics_rows = []
exact_match_count = 0
exact_match_normalized_count = 0
for item, (_, pred_sql) in zip(subset, predicted_rows):
    question = item.get('question', '').strip()
    gold_sql = item.get('query', '').strip()
    pred_sql = pred_sql.strip() if pred_sql else ""

    exact_match = pred_sql == gold_sql
    exact_match_normalized = normalize_sql(pred_sql) == normalize_sql(gold_sql)
    exact_match_count += int(exact_match)
    exact_match_normalized_count += int(exact_match_normalized)

    metrics_rows.append({
        "question": question,
        "prediction": pred_sql,
        "gold": gold_sql,
        "exact_match": exact_match,
        "exact_match_normalized": exact_match_normalized,
    })

metrics_summary = {
    "count": len(metrics_rows),
    "exact_match": exact_match_count,
    "exact_match_normalized": exact_match_normalized_count,
    "exact_match_rate": exact_match_count / max(len(metrics_rows), 1),
    "exact_match_normalized_rate": exact_match_normalized_count / max(len(metrics_rows), 1),
}

with open(metrics_output_file, 'w') as f:
    json.dump({"summary": metrics_summary, "rows": metrics_rows}, f, indent=2)

print("Running official evaluation...")


# Call the evaluation script from the spider folder
# We check for exact match first
os.system(f"python ./evaluation.py --gold {gold_output_file} --pred {output_file} --db data/database --table data/tables.json --etype match")
