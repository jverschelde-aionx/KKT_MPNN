# Model Scaling Experiments - Sweep Guide

This guide explains how to run comprehensive experiments comparing model variants across problem sizes.

## Overview

The hyperparameter sweep tests **12 model variants** across multiple **problem sizes** to measure:
1. JEPA self-supervised pre-training impact
2. Model scaling behavior (MLP vs GNN)
3. Feature normalization effects
4. Training mode comparisons (EMA vs SimSiam)

## Model Variants (12 total)

### Baselines (4 variants)
- `mlp_baseline_norm` - MLP with normalized features
- `mlp_baseline_unnorm` - MLP with raw features
- `gnn_baseline_norm` - GNN with normalized features
- `gnn_baseline_unnorm` - GNN with raw features

### JEPA Models (8 variants)
**MLP with EMA** (Mode 3: Pre-train → Joint):
- `mlp_jepa_ema_norm` - MLP + JEPA EMA + normalized
- `mlp_jepa_ema_unnorm` - MLP + JEPA EMA + unnormalized

**MLP with SimSiam** (Mode 4: Joint from start):
- `mlp_jepa_simsiam_norm` - MLP + JEPA SimSiam + normalized
- `mlp_jepa_simsiam_unnorm` - MLP + JEPA SimSiam + unnormalized

**GNN with EMA**:
- `gnn_jepa_ema_norm` - GNN + JEPA EMA + normalized
- `gnn_jepa_ema_unnorm` - GNN + JEPA EMA + unnormalized

**GNN with SimSiam**:
- `gnn_jepa_simsiam_norm` - GNN + JEPA SimSiam + normalized
- `gnn_jepa_simsiam_unnorm` - GNN + JEPA SimSiam + unnormalized

## Problem Configurations

**Problem Types**:
- `RND` - Random Linear Programs
- `CA` - Combinatorial Auction

**Problem Sizes**: 2, 5, 10, 20, 50, 100

**Total Experiments**: 12 model variants × 2 problem types × 6 sizes = **144 experiments**

## Usage

### 1. Test Locally (Single Configuration)

Test one configuration before launching full sweep:

```bash
cd src
python hyperparameter_sweep.py --mode local
```

This runs a quick 2-epoch test with MLP baseline on RND size 2.

### 2. Launch Full Sweep

**Option A: Run all model variants** (144 experiments):
```bash
python hyperparameter_sweep.py --mode sweep \
  --problem-types RND CA \
  --problem-sizes 2 5 10 20 50 100 \
  --epochs 50
```

**Option B: Test subset of models** (e.g., baselines only):
```bash
python hyperparameter_sweep.py --mode sweep \
  --model-variants mlp_baseline_norm gnn_baseline_norm \
  --problem-types RND \
  --problem-sizes 2 5 10 \
  --epochs 20
```

**Option C: Test one problem type** (72 experiments):
```bash
python hyperparameter_sweep.py --mode sweep \
  --problem-types RND \
  --problem-sizes 2 5 10 20 50 100 \
  --epochs 50
```

**Option D: Start with small sizes** (24 experiments):
```bash
python hyperparameter_sweep.py --mode sweep \
  --problem-types RND CA \
  --problem-sizes 2 5 \
  --epochs 30
```

### 3. Launch Multiple Parallel Agents

For faster execution, run multiple agents in parallel:

```bash
# Terminal 1
python hyperparameter_sweep.py --mode sweep --epochs 50

# The above will print a sweep ID like: wandb agent aionx/kkt_model_scaling/abc123

# Terminal 2-4 (launch additional agents with same sweep ID)
wandb agent aionx/kkt_model_scaling/abc123
```

### 4. Export Results to CSV

After experiments complete, download results:

```bash
python hyperparameter_sweep.py --mode export \
  --export-file scaling_results.csv \
  --project kkt_model_scaling
```

This creates a CSV with columns:
- Problem configuration (type, size)
- Model variant (architecture, JEPA mode, normalization)
- Training metrics (losses, KKT components)
- Runtime and metadata

## Prerequisites

### Generate Problem Instances

Before running the sweep, ensure instances exist for desired sizes:

```bash
cd src
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graph-aug

# Generate RND instances (sizes 2, 5, 10, 20, 50, 100)
for size in 2 5 10 20 50 100; do
  python generate_instances.py --problems RND --rnd_sizes $size \
    --n_instances 1000 --normalize_features true
done

# Generate CA instances
for size in 2 5 10 20 50 100; do
  python generate_instances.py --problems CA --ca_sizes $size \
    --n_instances 1000 --normalize_features true
done
```

**Note**: For unnormalized variants, the sweep will use the same instances but skip normalization during data loading (controlled by `normalize_features` flag in model config).

## Monitoring Progress

### WandB Dashboard

View real-time progress at: https://wandb.ai/YOUR_USERNAME/kkt_model_scaling

**Key Metrics to Monitor**:
- `valid/loss` - Validation KKT loss (primary metric)
- `train/loss_kkt` - Training KKT loss
- `train/loss_jepa` - JEPA self-supervised loss (for JEPA models)
- `valid/primal` - Primal feasibility violation
- `valid/dual` - Dual feasibility violation
- `valid/stationarity` - Stationarity condition violation
- `valid/complementary_slackness` - Complementary slackness violation

### Sweep Progress

Check sweep status:
```bash
wandb sweep list
```

## Analysis

### CSV Export Structure

The exported CSV contains:

**Configuration Columns**:
- `problem_type` - RND, CA
- `problem_size` - 2, 5, 10, 20, 50, 100
- `model_variant` - Full model variant name
- `architecture` - MLP or GNN
- `normalized` - True/False (feature normalization)
- `use_jepa` - True/False
- `jepa_mode` - ema, simsiam, or empty

**Results Columns**:
- `final_valid_loss` - Final validation KKT loss
- `best_valid_loss` - Best validation loss during training
- `final_train_loss_kkt` - Final training KKT loss
- `final_train_loss_jepa` - Final JEPA loss (if applicable)
- `final_primal`, `final_dual`, `final_stationarity`, `final_comp_slack` - KKT components
- `runtime_seconds` - Training duration

### Example Analysis (Python/Pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('scaling_results.csv')

# Compare baselines vs JEPA
baseline = df[df['use_jepa'] == False]
jepa = df[df['use_jepa'] == True]

# Plot scaling behavior
for variant in df['model_variant'].unique():
    subset = df[df['model_variant'] == variant]
    plt.plot(subset['problem_size'], subset['best_valid_loss'],
             marker='o', label=variant)

plt.xlabel('Problem Size')
plt.ylabel('Best Validation Loss')
plt.legend()
plt.title('Model Scaling Behavior')
plt.savefig('scaling_comparison.png')

# Compare JEPA impact per architecture
mlp_baseline = df[(df['architecture'] == 'MLP') & (df['use_jepa'] == False)]
mlp_jepa = df[(df['architecture'] == 'MLP') & (df['use_jepa'] == True)]

print("MLP Baseline avg loss:", mlp_baseline['best_valid_loss'].mean())
print("MLP + JEPA avg loss:", mlp_jepa['best_valid_loss'].mean())
print("JEPA improvement:",
      (mlp_baseline['best_valid_loss'].mean() - mlp_jepa['best_valid_loss'].mean()) /
      mlp_baseline['best_valid_loss'].mean() * 100, "%")
```

## Computational Requirements

### Estimated Runtimes (per experiment)

**Small sizes (2, 5)**:
- MLP: ~5-10 minutes (50 epochs)
- GNN: ~10-15 minutes (50 epochs)

**Medium sizes (10, 20)**:
- MLP: ~10-20 minutes
- GNN: ~20-30 minutes

**Large sizes (50, 100)**:
- MLP: ~30-60 minutes
- GNN: ~60-120 minutes

**Total sweep time** (144 experiments, sequential): ~30-50 hours
**With 4 parallel agents**: ~8-12 hours

### GPU Memory

**MLP models**:
- Batch size 256: ~4-6 GB GPU memory
- Works on most GPUs (GTX 1080+)

**GNN models**:
- Batch size 128: ~6-10 GB GPU memory
- Recommend RTX 2080 or better for large sizes

## Troubleshooting

### Issue: SCIP version mismatch during instance generation

**Error**: `ImportError: libscip.so.8.0: cannot open shared object file`

**Solution**: Use existing instances or fix SCIP version (see Sprint 1 integration test report).

### Issue: Out of memory during training

**Solution**: Reduce batch size in sweep config:
```python
# In hyperparameter_sweep.py, modify create_sweep_config():
batch_size_mlp=128,  # default: 256
batch_size_gnn=64,   # default: 128
```

### Issue: Training too slow

**Solutions**:
1. Reduce epochs: `--epochs 20` (instead of 50)
2. Use smaller problem sizes first: `--problem-sizes 2 5 10`
3. Test subset of models: `--model-variants mlp_baseline_norm gnn_baseline_norm`
4. Run multiple parallel agents

## Tips

1. **Start Small**: Test with `--problem-sizes 2 5` and `--epochs 20` first
2. **Prioritize**: Focus on baselines + one JEPA mode (e.g., EMA) initially
3. **Parallel Execution**: Launch 2-4 agents for faster completion
4. **Monitor Early**: Check first few runs to ensure setup is correct
5. **Export Regularly**: Export CSV periodically to track progress

## References

- Sprint 1 Implementation Report: `src/tests/SPRINT1_INTEGRATION_TESTS.md`
- JEPA Configuration Examples: `src/configs/config_jepa_*.yml`
- Training Script: `src/train.py`
