# Level 2: Production Text-to-SQL

## Run BOTH in Parallel on Kaggle:

### **V2:** `Text2SQL_V2_OPTIMIZED.ipynb` — Conservative approach
### **V3:** `Text2SQL_V3_AGGRESSIVE.ipynb` — Aggressive approach

Pick the best performer after training.

---

## What's In This Folder

| File | Status | Description |
|------|--------|-------------|
| **Text2SQL_V2_OPTIMIZED.ipynb** | ✅ RUN THIS | LR=1e-4, 20 epochs, cosine schedule |
| **Text2SQL_V3_AGGRESSIVE.ipynb** | ✅ RUN THIS | LR=3e-4, 15 epochs, linear schedule |
| Text2SQL_FINAL.ipynb | ⚠️ Baseline | 11.6% accuracy (LR too low) |
| Text2SQL_Production_Bulletproof.ipynb | ❌ Old | Previous attempt |
| Text2SQL_Production.ipynb | ❌ Old | Original version |

---

## Quick Start (Run Both in Parallel)

### Session 1: V2 (Conservative)
1. **Upload** `Text2SQL_V2_OPTIMIZED.ipynb` to Kaggle
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Run All**: Run all cells
4. **Wait**: ~2-3 hours
5. **Download**: `t2sql_v2_model.zip` + `report_v2.json`

### Session 2: V3 (Aggressive)
1. **Open new Kaggle session** (different tab)
2. **Upload** `Text2SQL_V3_AGGRESSIVE.ipynb`
3. **Enable GPU**: Settings → Accelerator → GPU T4 x2
4. **Run All**: Run all cells
5. **Wait**: ~1.5-2.5 hours
6. **Download**: `t2sql_v3_model.zip` + `report_v3.json`

**Compare results, submit the better one.**

---

## Version Comparison

| Setting | V2 (Conservative) | V3 (Aggressive) |
|---------|------------------|-----------------|
| Learning Rate | 1e-4 | 3e-4 |
| Epochs | 20 | 15 |
| LR Schedule | Cosine | Linear |
| Effective Batch | 32 | 64 |
| Warmup | 5% | 10% |
| Expected Time | 2-3 hours | 1.5-2.5 hours |
| Target Accuracy | 30-45% | 30-45% |

Both auto-select T5-Base if GPU has 15+ GB memory.

---

## Features

- ✅ Loads real Spider dataset (7,000 examples)
- ✅ Advanced schema serialization with types
- ✅ Safe tokenization (removes all non-tensor columns)
- ✅ Overflow protection in metrics
- ✅ Pre-training verification
- ✅ 10 epochs with cosine LR schedule
- ✅ Label smoothing (0.1)
- ✅ Early stopping
- ✅ Beam search (4 beams)

---

## Cells Overview

| Cell | Purpose | Time |
|------|---------|------|
| 1-2 | Setup & imports | 1 min |
| 3 | Load Spider dataset | 2 min |
| 4-5 | Schema & preprocessing | 3 min |
| 6-7 | Tokenization | 2 min |
| 8-9 | Load model & config | 1 min |
| 10-11 | Metrics & trainer | 30 sec |
| 12 | Verification checks | 30 sec |
| **13** | **TRAINING** | **6-8 hours** |
| 14-15 | Evaluate & save | 5 min |
| 16-17 | Test & report | 2 min |

---

## Troubleshooting

**"Dataset not found"**
```
Spider dataset moved. Try manually downloading from:
https://yale-lily.github.io/spider
```

**"Out of memory"**
```
Reduce batch size in Cell 9:
per_device_train_batch_size=4  # Instead of 8
```

**"Session timeout"**
```
Kaggle: 9-hour limit per session
Solution: Save checkpoints (already configured)
Resume with: trainer.train(resume_from_checkpoint="./text2sql_model/checkpoint-XXX")
```

---

## Kaggle vs Colab

| Platform | Can close browser? | Session limit | GPU quota |
|----------|-------------------|---------------|-----------|
| **Kaggle** | ✅ YES | 9 hours | 30h/week |
| Colab Free | ❌ NO | Variable | ~12h/week |
| Colab Pro | ✅ YES | 24 hours | Higher |

**Recommendation: Use Kaggle**

---

## After Training

### From V2:
1. `t2sql_v2_model.zip` - Zipped model
2. `t2sql_final_v2/` - Model folder
3. `report_v2.json` - Training metrics

### From V3:
1. `t2sql_v3_model.zip` - Zipped model
2. `t2sql_final_v3/` - Model folder
3. `report_v3.json` - Training metrics

**Compare both reports, submit the higher accuracy model.**

---

**Run both now. Come back to results.**
