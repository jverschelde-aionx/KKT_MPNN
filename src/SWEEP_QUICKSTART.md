# Hyperparameter Sweep - Quick Start Guide

Simple, streamlined workflow for running model scaling experiments.

## Single Command Workflow

The sweep script automatically:
1. 🚀 Launches WandB sweep with all model variants
2. 📊 Exports results to Excel when complete

## Usage

### Basic Usage (All 12 models, all sizes)
```bash
cd src
python hyperparameter_sweep.py \
  --problem-types RND CA \
  --problem-sizes 2 5 10 20 50 100 \
  --epochs 50
```

**Result**:
- Creates 144 experiments (12 models × 2 types × 6 sizes)
- Automatically exports to `sweep_results.xlsx`

### Start Small (24 experiments)
```bash
python hyperparameter_sweep.py \
  --problem-types RND CA \
  --problem-sizes 2 5 \
  --epochs 30
```

### Test Specific Models
```bash
python hyperparameter_sweep.py \
  --model-variants mlp_baseline_norm gnn_baseline_norm \
  --problem-sizes 2 5 10 \
  --epochs 20
```

### Custom Output File
```bash
python hyperparameter_sweep.py \
  --export-file my_results.xlsx \
  --epochs 50
```

### Disable Auto-Export
```bash
python hyperparameter_sweep.py \
  --no-export \
  --epochs 50
```

## Available Model Variants

**Baselines (4):**
- `mlp_baseline_norm`
- `mlp_baseline_unnorm`
- `gnn_baseline_norm`
- `gnn_baseline_unnorm`

**JEPA with EMA (4):**
- `mlp_jepa_ema_norm`
- `mlp_jepa_ema_unnorm`
- `gnn_jepa_ema_norm`
- `gnn_jepa_ema_unnorm`

**JEPA with SimSiam (4):**
- `mlp_jepa_simsiam_norm`
- `mlp_jepa_simsiam_unnorm`
- `gnn_jepa_simsiam_norm`
- `gnn_jepa_simsiam_unnorm`

## Parallel Execution

Speed up experiments by running multiple agents:

```bash
# Terminal 1 - Launch sweep
python hyperparameter_sweep.py --epochs 50
# Note the sweep ID (e.g., abc123xyz)

# Terminal 2-4 - Launch additional agents
wandb agent YOUR_USERNAME/kkt_model_scaling/abc123xyz
wandb agent YOUR_USERNAME/kkt_model_scaling/abc123xyz
wandb agent YOUR_USERNAME/kkt_model_scaling/abc123xyz
```

**4 agents = 4× faster!**

## Excel Output Format

The exported Excel file contains:

| Column | Description |
|--------|-------------|
| `problem_type` | RND, CA |
| `problem_size` | 2, 5, 10, 20, 50, 100 |
| `model_variant` | Full model name |
| `architecture` | MLP or GNN |
| `normalized` | Features normalized? |
| `use_jepa` | JEPA enabled? |
| `jepa_mode` | ema or simsiam |
| `best_valid_loss` | ⭐ Key metric for comparison |
| `final_primal` | Primal feasibility |
| `final_dual` | Dual feasibility |
| `final_stationarity` | Stationarity condition |
| `final_comp_slack` | Complementary slackness |
| `runtime_seconds` | Training duration |

## Quick Analysis

Load in Python/Pandas:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_excel('sweep_results.xlsx')

# Compare baselines vs JEPA
baselines = df[df['use_jepa'] == False]
jepa_models = df[df['use_jepa'] == True]

print("Baseline avg loss:", baselines['best_valid_loss'].mean())
print("JEPA avg loss:", jepa_models['best_valid_loss'].mean())

# Plot scaling behavior
for variant in df['model_variant'].unique():
    subset = df[df['model_variant'] == variant]
    plt.plot(subset['problem_size'], subset['best_valid_loss'],
             marker='o', label=variant)

plt.xlabel('Problem Size')
plt.ylabel('Best Validation Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## CLI Reference

```
python hyperparameter_sweep.py [OPTIONS]

Options:
  --problem-types      Problem types to test (default: RND CA)
  --problem-sizes      Sizes to test (default: 2 5 10 20 50 100)
  --epochs             Training epochs per run (default: 50)
  --batch-size         Batch size (default: 256)
  --model-variants     Models to test (default: all 12)
  --export-file        Output Excel file (default: sweep_results.xlsx)
  --no-export          Skip auto-export
  --project            WandB project (default: kkt_model_scaling)
  --entity             WandB username (default: your account)
```

## Tips

✅ **Start small** - Test with `--problem-sizes 2 5` first
✅ **Use parallel agents** - 4 agents = 4× speedup
✅ **Monitor WandB** - Check progress at wandb.ai
✅ **Check Excel early** - Export can run even with incomplete sweep

---

**See also:** `SWEEP_GUIDE.md` for detailed documentation
