# HuggingFace Model Storage Configuration

## Problem
By default, HuggingFace Transformers downloads models to `~/.cache/huggingface/` which can fill up the root partition (typically smaller).

## Solution: Store Models in Project Directory

### Option 1: Project-Local Cache (Recommended)

Store models within the project to avoid root partition issues:

```bash
# Set environment variable before running
export HF_HOME="/home/bs_thesis/shift (Copy)/Fact-or-Fiction/models/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

# Then run normally
conda run -n fact_check_env python3 scripts/run_component1_pv_esm.py --split train --limit_claims 20
```

**Benefits**:
- ✅ No root partition consumption
- ✅ Models stored with project (portable)
- ✅ Easy to clean up (`rm -rf models/`)
- ✅ Multiple projects can have separate caches

### Option 2: Custom Location on Larger Partition

If you have a `/data` or `/mnt` partition with more space:

```bash
export HF_HOME="/data/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
```

### Option 3: Permanent Configuration

Add to your `~/.bashrc` or create a `.env` file:

```bash
# In ~/.bashrc or project .env
export HF_HOME="/home/bs_thesis/shift (Copy)/Fact-or-Fiction/models/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
```

## Update .gitignore

Add to `.gitignore` to exclude from version control:

```
# HuggingFace model cache (large files)
models/huggingface/
```

## Runner Script Integration

The runner script automatically respects `HF_HOME` and `TRANSFORMERS_CACHE` environment variables - no code changes needed!

## Manual Pre-Download (Optional)

If you want to pre-download the model once:

```bash
export HF_HOME="/home/bs_thesis/shift (Copy)/Fact-or-Fiction/models/huggingface"

conda run -n fact_check_env python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print('✅ Model downloaded successfully!')
"
```

## Disk Space Check

Check current usage:
```bash
# Root partition
df -h /

# Project location
df -h "/home/bs_thesis/shift (Copy)/Fact-or-Fiction"
```

## Model Size Reference

- **DeBERTa-v3-large**: ~1.4 GB
- **Cache overhead**: ~200 MB
- **Total**: ~1.6 GB per model

---

**Recommendation for Your Case**: Use Option 1 (project-local cache) to keep models in your home directory where you have more space.
