# NatSQL — Local Extensions

This README documents how this workspace extends the original NatSQL repository and what was added locally.

## How this builds on the original NatSQL

The upstream NatSQL codebase provides:

- Data preprocessing and NatSQL annotation generation via `check_and_preprocess.sh`.
- NatSQL ↔ SQL conversion utilities under `natsql2sql/`.
- Evaluation scripts for exact‑match and execution accuracy.

This workspace **builds directly on those components** by adding a training + inference path for a **T5 NatSQL adapter** and lightweight utilities to prepare data and evaluate predictions.

## What was added here

### 1) T5 NatSQL adapter model

- A T5 fine‑tuned model under:
  - `t5-natsql-adapter-final/`
  - `t5-natsql-adapter-final2/`
- The model was trained on **Google Colab (T4 GPU)**.
- Training data was produced by the original NatSQL script:
  - `check_and_preprocess.sh`

### 2) Training data formatter

- `format_data.py` converts NatSQL‑annotated Spider examples into a **T5 source/target** format.
- It serializes the schema and produces entries like:
  - **source**: `Translate to NatSQL: <question> | <db_id> | <schema>`
  - **target**: `<natsql>`
- Output file: `ready_to_train.json`

### 3) Inference + evaluation driver

- `test_model_drive.py` loads the local T5 adapter and generates NatSQL for a subset of Spider dev questions.
- It optionally converts NatSQL → SQL using the official NatSQL parser.
- Outputs:
  - `predicted_sql.txt` (question + predicted SQL)
  - `predicted_metrics.json` (exact‑match summary)
  - `dev_gold_with_db.sql` (gold SQL + db_id for evaluation)

## End‑to‑end flow (local)

1. **Prepare NatSQL data** (upstream):
   - Run `check_and_preprocess.sh`
   - Produces NatSQL annotations in `NatSQLv1_6/`

2. **Format for T5** (local):
   - Run `format_data.py`
   - Produces `ready_to_train.json`

3. **Train adapter (Colab, T4 GPU)**
   - Use `ready_to_train.json` to fine‑tune T5
   - Save the adapter to `t5-natsql-adapter-final2/`

4. **Run inference + evaluate** (local):
   - Run `test_model_drive.py`
   - Generates predictions and evaluation artifacts

## Notes

- The NatSQL conversion logic is unchanged; the local work focuses on **model training and usage**.
- If you update schema serialization or prompt format, retrain the adapter for consistent results.


## Examples 
Real SQL (Hard): 
``SELECT T1.name FROM instructor AS T1 JOIN advisor AS T2 ON T1.ID = T2.i_ID JOIN student AS T3 ON T2.s_ID = T3.ID ORDER BY T3.tot_cred DESC LIMIT 1
``

NatSQL (Your Data): 
``select instructor.name from advisor where @.@ join advisor.* order by student.tot_cred desc limit 1
``
