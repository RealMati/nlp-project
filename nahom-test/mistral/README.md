# Mistral 7B LoRA Notebooks

This folder contains three notebooks that implement the same end‑to‑end Text‑to‑SQL pipeline using LoRA fine‑tuning on Mistral‑7B. They are intentionally similar so you can compare training settings and dataset sizes without changing the overall methodology.

## What these notebooks do

Each notebook follows a staged pipeline:

1. **Environment & constraints**
   - Fast iteration on a single GPU.
   - 4‑bit quantization + LoRA for memory efficiency.
   - Small dataset subset for early experiments.

2. **Dataset preparation (WikiSQL)**
   - Load WikiSQL from Hugging Face.
   - Select a fixed‑size subset for train/val/test.
   - Build a prompt that includes schema and question.
   - Concatenate SQL as the target continuation.

3. **Model setup**
   - Load Mistral‑7B and tokenizer.
   - Configure 4‑bit quantization (BitsAndBytes).
   - Freeze base weights and attach LoRA adapters.

4. **Training loop**
   - Tokenize to a fixed max length.
   - Use causal LM data collator.
   - Train with small batch sizes + gradient accumulation.
   - Save only LoRA adapters.

5. **Inference**
   - Rebuild the same prompt format.
   - Generate SQL with low temperature.
   - Stop at EOS / newline.

6. **Execution evaluation**
   - Conceptual stage describing accuracy by execution, not string match.

7. **Analysis & scaling**
   - Identify failure modes (columns, WHERE, aggregates).
   - Scale data and epochs only after pipeline is stable.

## Files and how they differ

- **mistral_7b_LoRA.ipynb**
  - Baseline, fastest run.
  - Smaller subsets (train ~1k, val/test ~200).
  - Short max length (256).
  - A conservative training configuration (short run).

- **mistral_7b_LoRA-2.ipynb**
  - Slight adjustments to training schedule and batches.
  - Similar subset sizes to the baseline.
  - Intended for a second pass with moderate tweaks.

- **mistral_7b_LoRA copy.ipynb**
  - Extended run variant.
  - Larger subsets (train ~8k, val/test ~1k).
  - Longer max length (512).
  - Designed for comparison against the baseline.

## Key implementation details

- **Prompt format**
  - A fixed template with:
    - Database schema section
    - Question section
    - SQL section (target continuation)
  - Keeping this identical across notebooks is critical for fair comparisons.

- **Tokenizer settings**
  - Right padding.
  - Pad token set to EOS (Mistral has no pad token by default).

- **LoRA modules**
  - Targets: attention projections and FFN (q/k/v/o, gate/up/down).
  - Low rank + small dropout for fast, stable training.

- **Quantization**
  - 4‑bit NF4 with double quantization.
  - Compute dtype set to fp16.

## Expected outputs

- **Training artifacts**
  - Trainer logs in the output directory.
  - LoRA adapter weights for reuse in inference.

- **Qualitative checks**
  - Sample SQL generations after training.
  - Manual inspection for hallucinated columns or invalid syntax.

## When to use which notebook

- Use **mistral_7b_LoRA.ipynb** to validate that the pipeline runs end‑to‑end.
- Use **mistral_7b_LoRA-2.ipynb** to test training tweaks without scaling data.
- Use **mistral_7b_LoRA copy.ipynb** for longer runs and broader coverage.

## Troubleshooting tips

- **Out of memory**
  - Reduce max length or batch size.
  - Increase gradient accumulation instead of batch size.

- **Bad SQL quality**
  - Check prompt format consistency.
  - Reduce temperature and max generation tokens.
  - Inspect dataset preprocessing for schema mismatch.

- **Slow training**
  - Reduce dataset size.
  - Lower evaluation frequency.
  - Reduce max steps/epochs.

## Consistency rule

If you change any prompt, preprocessing, or schema formatting logic, update all three notebooks to keep the comparisons meaningful.
