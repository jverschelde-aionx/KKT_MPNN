# KKT_MPNN

## Installation and Setup

### Using Docker (Recommended)

The easiest way to run this project is with Docker, which handles all dependencies automatically:

```bash
# Build and run the Docker container
docker compose up --build
```

For GPU support, uncomment the GPU-related sections in the compose.yaml file.

### Manual Installation

If you prefer to install dependencies manually, we recommend using conda:

```bash
# Create conda environment
conda env create -f requirements.yml
conda activate graph-aug

# Install PyTorch Geometric extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
```

## JEPA Self-Supervised Pre-training (Sprint 1 Feature)

KKT_MPNN now supports **JEPA (Joint-Embedding Predictive Architecture)** for self-supervised representation learning, enabling improved training through pre-training on the structure of LP problems.

### What is JEPA?

JEPA is a self-supervised learning method that learns representations by predicting embeddings of clean/lightly-masked inputs from heavily-masked inputs. Unlike reconstruction-based methods (like MAE), JEPA predicts in latent space, making it:
- **More efficient**: 2.5-10x faster than reconstruction methods
- **More effective**: Learns semantic structure rather than low-level details
- **Flexible**: Works with both MLP and GNN architectures

**Key References:**
- I-JEPA: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (Assran et al., CVPR 2023)
- BYOL: "Bootstrap Your Own Latent" (Grill et al., NeurIPS 2020)
- SimSiam: "Exploring Simple Siamese Representation Learning" (Chen & He, CVPR 2021)

### Training Modes

JEPA supports **4 distinct training workflows** controlled by `jepa_pretrain_epochs` and `jepa_weight`:

#### Mode 1: Baseline (KKT-only)
No JEPA - traditional KKT training for comparison.
```bash
cd src
python train.py --batch_size 8 --epochs 20
# No JEPA flags needed
```

#### Mode 2: JEPA Pre-train → KKT Fine-tune
Traditional SSL workflow: pre-train representations, then fine-tune for KKT.
```bash
python train.py --batch_size 8 --epochs 20 \
  --use_jepa --jepa_mode ema \
  --jepa_pretrain_epochs 3 --jepa_weight 0
```
- First 3 epochs: JEPA-only (representation learning)
- Remaining epochs: KKT-only (task-specific fine-tuning)

#### Mode 3: JEPA Pre-train → Joint Training (Recommended)
Pre-train then maintain representations with auxiliary JEPA loss.
```bash
python train.py --batch_size 8 --epochs 20 \
  --use_jepa --jepa_mode ema \
  --jepa_pretrain_epochs 3 --jepa_weight 0.2
```
- First 3 epochs: JEPA-only
- Remaining epochs: KKT + 0.2×JEPA (prevents representation collapse)

#### Mode 4: Joint Training from Start
Combined JEPA+KKT from epoch 0 (faster but may learn less general representations).
```bash
python train.py --batch_size 8 --epochs 20 \
  --use_jepa --jepa_mode ema \
  --jepa_pretrain_epochs 0 --jepa_weight 0.2
```

### MLP vs GNN Usage

**MLP Architecture** (default):
```bash
python train.py --use_jepa --jepa_mode ema \
  --jepa_pretrain_epochs 3 --jepa_weight 0.2 \
  --jepa_mask_entry_online 0.40 \
  --jepa_mask_row_online 0.20 \
  --jepa_mask_col_online 0.20
```

**GNN Architecture** (bipartite graphs):
```bash
python train.py --use_bipartite_graphs --use_jepa --jepa_mode ema \
  --jepa_pretrain_epochs 3 --jepa_weight 0.2 \
  --jepa_mask_ratio_nodes 0.3
```

### LP-Aware Masking Strategy (MLP Only)

For MLP models, JEPA uses **structure-aware masking** that respects LP semantics:

**Three masking types:**
1. **Row masking** (constraints): Masks entire constraint `A[i,:] + b[i]`
2. **Column masking** (variables): Masks entire variable `A[:,j] + c[j]`
3. **Entry masking**: Individual coefficients `A[i,j]`

**Asymmetric views:**
- **Online view** (context, heavier mask):
  - 40% entries + 20% rows + 20% cols → ~60-70% of A masked
  - Forces model to infer structure from partial information
- **Target view** (lighter or clean):
  - 10% entries + 5% rows + 5% cols → ~15-20% of A masked
  - Or completely clean (all ratios = 0)

**Safety guarantees:**
- Always keeps ≥1 unmasked row AND ≥1 unmasked column
- Only masks within real problem region (respects padding)
- Maintains LP semantic coherence through tied masking

**Why this works for LPs:**
- Row masking: Forces model to infer constraint structure
- Column masking: Forces model to infer variable relationships
- Entry masking: Adds fine-grained perturbations
- Tied masking: Maintains semantic coherence (constraint = A row + b value)

### EMA vs SimSiam Mode

**EMA Mode (Recommended)**:
```bash
--jepa_mode ema --ema_momentum 0.996
```
- Uses separate target encoder updated via exponential moving average
- More stable, better accuracy, prevents collapse better
- Requires 2× memory (online + target models)
- Best for research and when memory allows

**SimSiam Mode**:
```bash
--jepa_mode simsiam
```
- Shares encoder, uses stop-gradient instead of EMA
- Lighter weight, lower memory usage
- Sufficient for large models on standard problems
- Best for production/resource-constrained settings

### Configuration Options

**Core JEPA settings:**
```yaml
jepa:
  use_jepa: true              # Enable JEPA training
  jepa_mode: "ema"            # "ema" or "simsiam"
  jepa_weight: 0.2            # Weight for JEPA loss in joint training
  jepa_pretrain_epochs: 3     # JEPA-only epochs before joint training
  ema_momentum: 0.996         # EMA momentum (ema mode only)
```

**MLP masking (LP-aware):**
```yaml
  # Online view (heavier mask - context)
  jepa_mask_entry_online: 0.40
  jepa_mask_row_online: 0.20
  jepa_mask_col_online: 0.20

  # Target view (lighter mask or clean)
  jepa_mask_entry_target: 0.10  # Set to 0 for completely clean target
  jepa_mask_row_target: 0.05
  jepa_mask_col_target: 0.05
```

**GNN masking (node-level):**
```yaml
  jepa_mask_ratio_nodes: 0.3  # Fraction of nodes to mask
```

**Optional augmentations:**
```yaml
  jepa_noisy_mask: false      # Add Gaussian noise at masked positions (vs zeros)
  jepa_row_scaling: false     # Apply row scaling s_i ~ LogUniform(0.5, 2.0)
```

### Recommended Defaults

- **JEPA weight**: 0.2 (Mode 3/4)
- **EMA momentum**: 0.996 (standard for vision tasks)
- **Pre-training epochs**: 3 (quick exploration) or 10-50 (thorough pre-training)
- **MLP online mask**: 40% entries, 20% rows, 20% cols
- **MLP target mask**: 10% entries, 5% rows, 5% cols (or 0 for clean)
- **GNN mask ratio**: 0.3 (node-level)
- **Mode**: Mode 3 (pre-train → joint) for best results

### Monitoring Training

JEPA training logs separate losses to WandB:
- `train/loss_jepa`: Self-supervised JEPA loss
- `train/loss_kkt`: Task-specific KKT loss
- `train/loss`: Combined loss (during joint training)

During pre-training epochs, only `train/loss_jepa` is optimized. After pre-training, both losses are logged.

### Checkpointing

Checkpoints automatically save target model state when using EMA mode:
```python
checkpoint = {
    "model": online_model.state_dict(),
    "target_model": target_model.state_dict(),  # Saved for EMA mode
    "optimizer": optimizer.state_dict(),
    # ...
}
```

SimSiam mode does not save a separate target model (encoder is shared).

## Optional Feature Normalization

Control whether problem instance features undergo min-max normalization:

**Enabled (default)** - scales features to [0, 1]:
```bash
python generate_instances.py --problems CA --ca_sizes 5 --n_instances 1000
# or explicitly:
python generate_instances.py --normalize_features true
```

**Disabled** - preserves raw feature values:
```bash
python generate_instances.py --normalize_features false
```

**When to disable normalization:**
- Reproducing papers that don't normalize
- Features already on similar scales
- Studying impact of scale information on learning
- Comparing with non-neural optimization methods

**When to keep normalization (default):**
- Features have vastly different scales
- Using standard neural network architectures
- General-purpose training (recommended for most use cases)

## Method Implementation
We utilize the bipartite graph convolution available on GitHub1 (Han et al., 2023), as the architecture for our MPNN. Two
iterations of the process shown in Figure 2(a) are applied, resulting in two constraint-side and two variable-side convolutions.
Our proposed model is implemented using the Transformer encoder code from GitHub2 (Wu et al., 2021), maintaining the
same configuration. We developed two MPNN-based baselines, M MLP and M CNN. M MLP consists of four MLP layers
with a hidden size of 128 and tanh activation, while M CNN includes four CNN layers followed by an MLP layer with
ReLU activation. We utilized the positional encoding module from GitHub3 (Gorishniy et al., 2022).
All ML models were trained using the proposed learning algorithm (Algorithm 1) with RMSprop (learning rate = 1e-4,
epsilon = 1e-5, alpha = 0.99, weight decay = 1e-3). They were trained concurrently on 64 different instances with 5,000
parameter updates for the results in Tables 1 and 3, and 10,000 for Table 2. Our RL algorithm is built upon the Actor-Critic
implementation in PyTorch4 (Kostrikov, 2018), modified to be tailored for MILP problems.

## References
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- https://github.com/yandex-research/rtdl-num-embeddings
- https://github.com/ucbrise/graphtrans
- https://github.com/sribdcn/Predict-and-Search_MILP_method

##
Test scalability (problems scale in size)
Differt problem types (different types)
compare with traditional solvers in runtime


# IP And WA problem instances
https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view