# Implementation Plan: Sprint 1 Features (JEPA Pre-training + Optional Normalization)

## Overview

This plan covers the implementation of two distinct features for the KKT_MPNN project:

1. **JEPA Self-Supervised Pre-training**: Add Joint-Embedding Predictive Architecture (JEPA) for representation learning with both EMA (BYOL/I-JEPA style) and SimSiam modes, supporting both MLP and GNN architectures.

2. **Optional Feature Normalization**: Allow users to control whether problem instance features undergo min-max normalization during data generation, maintaining backward compatibility with current default behavior.

These features will enable researchers to explore improved representation learning and compare model performance with/without normalization.

---

## Requirements Summary

### Feature 1: JEPA Self-Supervised Pre-training

**Core Requirements:**
- Flexible training modes supporting 4 distinct workflows:
  1. **Baseline KKT-only**: Traditional training without JEPA (control group)
  2. **JEPA pre-train → KKT fine-tune**: SSL pre-training then task-specific fine-tuning (`jepa_pretrain_epochs > 0, jepa_weight = 0`)
  3. **JEPA pre-train → Joint training**: Pre-training then joint optimization (`jepa_pretrain_epochs > 0, jepa_weight > 0`)
  4. **Joint from start**: Combined JEPA+KKT from epoch 0 (`jepa_pretrain_epochs = 0, jepa_weight > 0`)
- Multiple architecture support: Both KKTNetMLP and GNNPolicy
- Configurable training strategy: EMA mode (target encoder with momentum) and SimSiam mode (stop-gradient, no EMA)
- Masking strategies: Instance-level for MLP, node-level for GNN
- Adjustable training schedule: Configurable pre-training epochs and loss weighting via `jepa_weight` parameter
- Checkpoint compatibility: Save/restore both online and target models
- Enable with single `--use_jepa` flag

**Success Criteria:**
- Pre-training runs successfully for configured epochs before joint training
- Joint training combines JEPA + KKT losses with configurable weighting
- Both EMA and SimSiam modes work without crashes
- Checkpoints save/restore all states correctly
- Training logs show separate JEPA and KKT loss terms
- Measurable improvement in KKT loss convergence or solution quality
- Works for both MLP and GNN architectures

**Recommended Defaults:**
- JEPA weight: 0.2
- EMA momentum: 0.996
- MLP masking (online view): 40% entries, 20% rows, 20% columns
- MLP masking (target view): 10% entries, 5% rows, 5% columns (or 0 for clean)
- GNN mask ratio: 0.3 (node-level)
- Pre-training epochs: 3
- Noisy mask: false (use hard zero masking)
- Row scaling: false (can enable for augmentation)

### Feature 2: Optional Feature Normalization

**Core Requirements:**
- Configuration flag to enable/disable feature normalization
- Default behavior maintains backward compatibility (normalization ON)
- Applied consistently across all feature types
- Clear documentation in generated data metadata

**Success Criteria:**
- Users can disable normalization with flag (e.g., `--normalize_features false`)
- Generated instances respect the flag
- Default behavior unchanged (normalization enabled)
- Models train successfully on both normalized and unnormalized data
- No errors with unnormalized data

---

## Research Findings

### Best Practices: JEPA Architecture

**Key Insights from 2024 Research:**

1. **Masking Strategy** (Critical Design Choice):
   - Sample target blocks with sufficiently large scale (semantic information)
   - Use sufficiently informative (spatially distributed) context blocks
   - Large masked regions force the model to learn complex world representations
   - Random patch sampling makes task too easy and limits learning

2. **Latent Space Prediction**:
   - Predict in abstract representation space, not pixel/input space
   - Allows model to focus on higher-level conceptual information
   - More computationally efficient than generative approaches
   - I-JEPA is 2.5x faster than iBOT and 10x more efficient than MAE

3. **Avoiding Collapse**:
   - Incorporate VICReg regularization strategies when learning from large/diverse datasets
   - EMA teacher helps prevent collapse, especially important with smaller models
   - SimSiam can work without EMA by using stop-gradient, but EMA boosts performance
   - Noise reduction from EMA averaging enables larger learning rates

4. **EMA vs SimSiam Trade-offs**:
   - **EMA (BYOL/I-JEPA)**: More stable, better accuracy, prevents collapse better, but adds memory/computation overhead
   - **SimSiam**: Lighter weight, shares encoder weights, sufficient for large models on standard benchmarks
   - **Recommendation**: Start with EMA for research; SimSiam for production efficiency

5. **Architecture Components**:
   - Encoder: Transforms inputs to abstract representations
   - Projector: Maps encoder output to embedding space (normalized)
   - Predictor: Online-only, predicts target embeddings from context embeddings
   - Target encoder (EMA only): Slow-moving average of online encoder

### Best Practices: Feature Normalization

**Min-Max vs Standardization:**

1. **Min-Max Normalization** (Current Implementation):
   - Scales features to [0, 1] range
   - Formula: `(x - min) / (max - min)`
   - **Advantages**:
     - Bounds values to fixed range suitable for neural network activations
     - Preserves zero entries (important for sparse constraint matrices)
     - Typical choice for neural networks requiring 0-1 scale data
   - **Disadvantages**:
     - Sensitive to outliers (extreme values crush other values)
     - Loses information about original problem scale

2. **When to Skip Normalization**:
   - Comparing with papers using unnormalized features (reproducibility)
   - Preserving original problem structure and scale information
   - When features already on similar scales
   - Studying impact of preprocessing on model performance

3. **Impact on Optimization**:
   - Gradient descent converges faster with feature scaling
   - Improves numerical conditioning of optimization problem
   - Without scaling, certain weights update faster than others
   - Essential for good training behavior in deep networks

**Recommendation**: Maintain min-max normalization as default (current behavior), but allow researchers to disable for experimental flexibility.

---

## Reference Implementations

### JEPA Reference Implementations (External)

1. **Meta I-JEPA** (Official):
   - Repository: https://github.com/facebookresearch/ijepa
   - Language: PyTorch
   - Features: Vision transformers, block masking, EMA teacher
   - Key patterns: Target encoder momentum update, cosine similarity loss, multi-block masking strategy

2. **Related Papers**:
   - "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023)
   - "Exploring Simple Siamese Representation Learning" (SimSiam, 2021)
   - "Bootstrap Your Own Latent" (BYOL, 2020)

### Internal Reference Points

**Codebase Patterns to Follow:**

1. **Optional Feature Toggle** (like `use_bipartite_graphs`):
   - Location: `src/train.py:88`
   - Pattern: Boolean action flag in config + conditional model instantiation
   - Apply to: `--use_jepa` flag

2. **Loss Function Modularity** (like KKT loss):
   - Location: `src/models/losses.py:61-100`
   - Pattern: Individual term functions + weighted combination + return dict for logging
   - Apply to: JEPA loss functions

3. **Configuration Management**:
   - Location: `src/config.yml`
   - Pattern: Section-based YAML config with CLI override via configargparse
   - Apply to: New `jepa` section in config

4. **Model Architecture with Argument Registration**:
   - Location: `src/models/models.py:44-73` (GNNPolicy.add_args)
   - Pattern: Static method to register model-specific CLI arguments
   - Apply to: JEPA configuration arguments

5. **Normalization Control**:
   - Location: `src/data/generators.py:108-114` (_minmax_normalization function)
   - Location: `src/data/common.py:19-20` (Settings dataclass)
   - Pattern: Conditional normalization based on Settings flags
   - Apply to: New `normalize_features` field in Settings

---

## Technology Decisions

### For JEPA Implementation

**1. Architecture Wrapper vs Integrated**:
- **Decision**: Create separate `jepa.py` module with utility functions, integrate directly into existing models
- **Rationale**: Keeps existing model classes clean, modular code, easier to toggle on/off
- **Alternative Considered**: Wrapper class around models - rejected due to complexity with checkpointing and forward passes

**2. Loss Combination Strategy**:
- **Decision**: Weighted sum of JEPA + KKT losses with configurable scheduling
- **Rationale**: Simple, interpretable, allows gradual transition from pre-training to joint training
- **Alternative Considered**: Multi-task loss balancing (e.g., uncertainty weighting) - rejected as over-engineering for first version

**3. Masking Implementation**:
- **Decision**: MLP: zero-out masked entries with optional small Gaussian noise; GNN: zero-out node features
- **Rationale**: Simple, effective, preserves graph structure for GNN, low computational overhead
- **Alternative Considered**: Learnable mask tokens - rejected to keep implementation lightweight

**4. Embedding Normalization**:
- **Decision**: L2-normalize embeddings before computing cosine similarity loss
- **Rationale**: Standard practice in contrastive/JEPA methods, numerical stability, bounded gradients
- **Implementation**: `torch.nn.functional.normalize(z, dim=-1)`

### For Optional Normalization

**1. Configuration Approach**:
- **Decision**: Add boolean flag to Settings dataclass, propagate through entire pipeline
- **Rationale**: Follows existing pattern (`add_positional_features`, `normalize_positional_features`), clear semantics
- **Alternative Considered**: Enum for normalization type (none/minmax/standardize) - deferred to future work

**2. Default Value**:
- **Decision**: `normalize_features: bool = True`
- **Rationale**: Backward compatibility, normalization is generally beneficial, matches current behavior

---

## Implementation Tasks

### Phase 1: JEPA Foundation (Feature 1)

#### Task 1.1: Create JEPA Utility Functions
**Description**: Implement core JEPA utilities for EMA update, loss computation, and LP-aware view generation

**Files to create:**
- `src/models/jepa_utils.py`

**Implementation Details:**
```python
# Functions to implement:
# 1. ema_update(target, online, momentum) - EMA parameter update
# 2. cosine_pred_loss(pred, target) - Cosine similarity loss: 2 - 2*cos(pred, target)
# 3. jepa_loss_mlp(online, target, x_on, x_tg, mode) - MLP JEPA loss with asymmetric views
# 4. jepa_loss_gnn(online, target, ctx, tgt, masks, mode) - GNN JEPA loss
# 5. make_lp_jepa_views(A, b, c, mask_m, mask_n, config) - LP-aware masking with:
#    - Row masking (constraints): mask A[i,:] + b[i]
#    - Column masking (variables): mask A[:,j] + c[j]
#    - Entry masking (individual coefficients)
#    - Online view (heavy): 40% entries, 20% rows, 20% cols
#    - Target view (light): 10% entries, 5% rows, 5% cols (or clean)
#    - Optional row scaling augmentation
#    - Respect padding, guarantee at least 1 unmasked row/col
# 6. make_gnn_views(batch_graph, mask_ratio) - GNN node masking
```

**Key Implementation Requirements:**
- **Structure-aware masking**: When row i is masked, set M_b[i]=1 and M_A[i,:]=1
- **Tied masking**: When column j is masked, set M_c[j]=1 and M_A[:,j]=1
- **Composition**: M_A = M_row ∨ M_col ∨ M_entry
- **Padding safety**: Only mask within real region (honor mask_m, mask_n)
- **Context guarantee**: Keep at least 1 unmasked row and 1 unmasked column
- **Asymmetric views**: Online gets heavier mask, target gets light/clean mask
- **Optional noise**: ε ~ N(0, 0.01 * median|coefficients|) at masked positions

**Dependencies**: None (pure utility functions)

**Estimated Effort**: 4-5 hours (increased due to LP-specific masking complexity)

**Reference**:
- Specification: `/home/joachim-verschelde/Repos/KKT_MPNN/PRPs/context/JEPA_integration_analysis.md:113-208`
- Pattern: Modular loss functions in `src/models/losses.py:4-100`

---

#### Task 1.2: Add JEPA Components to KKTNetMLP
**Description**: Extend MLP model with projector/predictor heads and embedding API

**Files to modify:**
- `src/models/models.py` (class KKTNetMLP)

**Implementation Details:**
- Add projector network in `__init__`: `self.jepa_proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, embed_dim))`
- Add predictor network in `__init__`: `self.jepa_pred = nn.Sequential(nn.Linear(embed_dim, pred_dim), nn.ReLU(), nn.Linear(pred_dim, embed_dim))`
- Implement `encode_trunk(flat_input)` method - returns hidden representation before task heads
- Implement `jepa_embed(flat_input)` method - returns L2-normalized projection: `normalize(self.jepa_proj(encode_trunk(x)), dim=-1)`
- Implement `jepa_pred(z)` method - returns L2-normalized prediction: `normalize(self.jepa_pred(z), dim=-1)`
- Keep existing `forward()` and heads unchanged

**Architecture Flow:**
```
Online: x_on → encode_trunk → jepa_proj → jepa_pred → p_online (normalized)
Target: x_tg → encode_trunk → jepa_proj → z_target (normalized, stop-grad)
```

**Code insertion point**: `src/models/models.py:23` (after self.net definition)

**Dependencies**: None

**Estimated Effort**: 1.5 hours

**Reference**: Specification lines 45-75 in JEPA_integration_analysis.md

---

#### Task 1.3: Add JEPA Components to GNNPolicy
**Description**: Extend GNN model with per-node projectors/predictors and embedding API

**Files to modify:**
- `src/models/models.py` (class GNNPolicy)

**Implementation Details:**
- Add projectors for constraint and variable nodes in `__init__` (near heads)
- Add predictors for constraint and variable nodes (online only)
- Implement `jepa_embed_nodes(constraint_features, edge_indices, edge_features, variable_features)` method
- Returns tuple of L2-normalized embeddings: (constraint_embeddings, variable_embeddings)
- Keep existing `encode()` and heads unchanged

**Code insertion point**: Near existing heads in GNNPolicy.__init__

**Dependencies**: None

**Estimated Effort**: 1-1.5 hours

**Reference**: Specification lines 78-110 in JEPA_integration_analysis.md

---

#### Task 1.4: Add JEPA Configuration Arguments
**Description**: Register JEPA-related command-line arguments and config options

**Files to modify:**
- `src/train.py` (argument parser setup)
- `src/config.yml` (add jepa section)

**Implementation Details:**

Add to train.py after line 103 (in training argument group):
```python
t.add_argument("--use_jepa", action="store_true", help="Enable JEPA training")
t.add_argument("--jepa_mode", choices=["ema", "simsiam"], default="ema",
              help="EMA teacher (BYOL/I-JEPA) or SimSiam (no EMA)")
t.add_argument("--jepa_weight", type=float, default=0.2, help="Weight for JEPA loss")
t.add_argument("--jepa_pretrain_epochs", type=int, default=3,
              help="JEPA-only epochs before joint KKT+JEPA")

# LP-aware masking for MLP (online/context view - heavier)
t.add_argument("--jepa_mask_entry_online", type=float, default=0.40,
              help="MLP online view: fraction of A entries masked")
t.add_argument("--jepa_mask_row_online", type=float, default=0.20,
              help="MLP online view: fraction of constraint rows masked")
t.add_argument("--jepa_mask_col_online", type=float, default=0.20,
              help="MLP online view: fraction of variable columns masked")

# LP-aware masking for MLP (target view - lighter or clean)
t.add_argument("--jepa_mask_entry_target", type=float, default=0.10,
              help="MLP target view: fraction of A entries masked (0 for clean)")
t.add_argument("--jepa_mask_row_target", type=float, default=0.05,
              help="MLP target view: fraction of constraint rows masked (0 for clean)")
t.add_argument("--jepa_mask_col_target", type=float, default=0.05,
              help="MLP target view: fraction of variable columns masked (0 for clean)")

# GNN masking (node-level)
t.add_argument("--jepa_mask_ratio_nodes", type=float, default=0.3,
              help="GNN: fraction of nodes masked")

# Augmentation options
t.add_argument("--jepa_noisy_mask", action="store_true",
              help="Add Gaussian noise at masked positions (vs zeros)")
t.add_argument("--jepa_row_scaling", action="store_true",
              help="Apply row scaling augmentation (s_i ~ LogUniform(0.5, 2.0))")

t.add_argument("--ema_momentum", type=float, default=0.996,
              help="Momentum for EMA teacher (ema mode only)")
```

Add to config.yml:
```yaml
jepa:
  use_jepa: false
  jepa_mode: "ema"
  jepa_weight: 0.2
  jepa_pretrain_epochs: 3

  # LP-aware masking ratios (MLP)
  # Online view (heavier mask - context)
  jepa_mask_entry_online: 0.40
  jepa_mask_row_online: 0.20
  jepa_mask_col_online: 0.20

  # Target view (lighter mask or clean)
  jepa_mask_entry_target: 0.10
  jepa_mask_row_target: 0.05
  jepa_mask_col_target: 0.05

  # GNN masking
  jepa_mask_ratio_nodes: 0.3

  # Augmentation options
  jepa_noisy_mask: false
  jepa_row_scaling: false

  ema_momentum: 0.996
```

**Dependencies**: None

**Estimated Effort**: 30 minutes

**Reference**:
- Specification lines 20-37 in JEPA_integration_analysis.md
- Pattern: `src/train.py:84-103` (training args), `src/models/models.py:44-73` (GNNPolicy.add_args)

---

### Phase 2: JEPA Training Integration (Feature 1)

#### Task 2.1: Create Optional EMA Target Model
**Description**: Instantiate target encoder when using EMA mode, set up no-grad parameters

**Files to modify:**
- `src/train.py` (in `train()` function, after model creation)

**Implementation Details:**
- Import `deepcopy` from copy module (add at top)
- After model instantiation (around line 251-255), add:
```python
target_model = None
if args.use_jepa and args.jepa_mode == "ema":
    from copy import deepcopy
    target_model = deepcopy(model)
    for p in target_model.parameters():
        p.requires_grad_(False)
```

**Code insertion point**: `src/train.py:255` (right after model instantiation)

**Dependencies**: Tasks 1.2, 1.3, 1.4 must be complete

**Estimated Effort**: 30 minutes

**Reference**: Specification lines 214-223 in JEPA_integration_analysis.md

---

#### Task 2.2: Update TrainingState for Multi-Loss Tracking
**Description**: Extend TrainingState class to track JEPA and KKT losses separately

**Files to modify:**
- `src/train.py` (TrainingState class)

**Implementation Details:**
- Add `jepa_loss_sum` field to `__init__`
- Add method `add_jepa_loss(loss)` for JEPA loss tracking
- Extend `finish_epoch()` to return both losses
- Reset JEPA loss in `_reset_training_state()`

**Code location**: `src/train.py:21-60`

**Dependencies**: None

**Estimated Effort**: 30 minutes

**Reference**: Pattern from existing `training_loss_sum` tracking in TrainingState

---

#### Task 2.3: Extend Checkpoint Save/Load for Target Model
**Description**: Save and restore target model state when using EMA mode

**Files to modify:**
- `src/train.py` (checkpoint creation and loading)

**Implementation Details:**

When creating checkpoint (around line 308-321):
```python
ckpt = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "args": vars(args),
}
if target_model is not None:
    ckpt["target_model"] = target_model.state_dict()
```

When loading checkpoint (if resumption is implemented):
```python
if args.use_jepa and args.jepa_mode == "ema" and "target_model" in ckpt:
    target_model.load_state_dict(ckpt["target_model"])
```

**Code location**: Around `src/train.py:308-321`

**Dependencies**: Task 2.1 complete

**Estimated Effort**: 30 minutes

**Reference**: Specification lines 226-245 in JEPA_integration_analysis.md

---

#### Task 2.4: Update train_epoch Function Signature
**Description**: Add args and target_model parameters to train_epoch

**Files to modify:**
- `src/train.py` (train_epoch function definition and call site)

**Implementation Details:**

Update function signature:
```python
def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    training_state: TrainingState,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    args=None,           # NEW
    target_model=None,   # NEW
) -> float:
```

Update call site (where train_epoch is invoked):
```python
train_loss = train_epoch(
    model=model,
    loader=train_loader,
    optimizer=optimizer,
    device=device,
    training_state=training_state,
    primal_weight=args.primal_weight,
    dual_weight=args.dual_weight,
    stationarity_weight=args.stationarity_weight,
    complementary_slackness_weight=args.complementary_slackness_weight,
    args=args,                # NEW
    target_model=target_model, # NEW
)
```

**Code locations**: Function definition in train.py, call site in train() function

**Dependencies**: Task 2.1 complete

**Estimated Effort**: 15 minutes

**Reference**: Specification lines 248-284 in JEPA_integration_analysis.md

---

#### Task 2.5: Integrate JEPA Loss Computation in Training Loop
**Description**: Add JEPA loss calculation, combine with KKT loss, implement pre-training schedule

**Files to modify:**
- `src/train.py` (inside train_epoch, within batch loop)

**Implementation Details:**

After computing KKT loss (find where `loss_kkt, _ = kkt_loss(...)` is called):

```python
loss = loss_kkt

if args and args.use_jepa:
    # Import utility functions at top of file
    from models.jepa_utils import (jepa_loss_mlp, jepa_loss_gnn,
                                    make_lp_jepa_views, make_gnn_views)

    # Detect model type and compute JEPA loss
    if isinstance(batch[0], torch_geometric.data.Batch):
        # GNN path
        batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes = batch
        ctx, tgt, m_cons, m_vars = make_gnn_views(batch_graph, args.jepa_mask_ratio_nodes)
        online_ref = model
        target_ref = target_model if args.jepa_mode == "ema" else model
        loss_jepa = jepa_loss_gnn(online_ref, target_ref, ctx, tgt, m_cons, m_vars, args.jepa_mode)
    else:
        # MLP path - LP-aware structured masking
        model_input, A, b, c, mask_m, mask_n = batch
        # Create asymmetric views: heavy online mask, light/clean target mask
        x_on, x_tg = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=args.jepa_mask_entry_online,
            r_row_on=args.jepa_mask_row_online,
            r_col_on=args.jepa_mask_col_online,
            r_entry_tg=args.jepa_mask_entry_target,
            r_row_tg=args.jepa_mask_row_target,
            r_col_tg=args.jepa_mask_col_target,
            noisy_mask=args.jepa_noisy_mask,
            row_scaling=args.jepa_row_scaling
        )
        online_ref = model
        target_ref = target_model if args.jepa_mode == "ema" else model
        loss_jepa = jepa_loss_mlp(online_ref, target_ref, x_on, x_tg, args.jepa_mode)

    # Pre-training schedule: JEPA-only for first N epochs
    jepa_only = training_state.get_epoch() < (args.jepa_pretrain_epochs or 0)
    loss = (loss_jepa if jepa_only else (loss_kkt + args.jepa_weight * loss_jepa))

    # Log JEPA loss separately
    if (training_state.get_step() % training_state.log_every) == 0:
        wandb.log({"train/loss_jepa": float(loss_jepa)}, step=training_state.get_step())
        if not jepa_only:
            wandb.log({"train/loss_kkt": float(loss_kkt)}, step=training_state.get_step())
```

**Code location**: Inside train_epoch batch loop, after KKT loss computation

**Dependencies**: Tasks 1.1, 2.1, 2.2, 2.4 complete

**Estimated Effort**: 2 hours

**Reference**: Specification lines 287-326 in JEPA_integration_analysis.md

---

#### Task 2.6: Add EMA Update After Optimizer Step
**Description**: Update target encoder with EMA after each optimizer step when using EMA mode

**Files to modify:**
- `src/train.py` (inside train_epoch, after optimizer.step())

**Implementation Details:**

After `optimizer.step()` in train_epoch:
```python
if args and args.use_jepa and args.jepa_mode == "ema" and (target_model is not None):
    from models.jepa_utils import ema_update
    ema_update(target_model, model, m=args.ema_momentum)
```

**Code location**: Right after `optimizer.step()` in train_epoch

**Dependencies**: Task 2.1, 2.5 complete

**Estimated Effort**: 15 minutes

**Reference**: Specification lines 329-332 in JEPA_integration_analysis.md

---

### Phase 3: Optional Normalization (Feature 2)

#### Task 3.1: Add normalize_features to Settings Dataclass
**Description**: Add configuration field to control feature normalization

**Files to modify:**
- `src/data/common.py` (Settings dataclass)

**Implementation Details:**

Add field to Settings:
```python
@dataclass(frozen=True)
class Settings:
    add_positional_features: bool = True
    normalize_positional_features: bool = False
    normalize_features: bool = True  # NEW: control normalization (default True for backward compatibility)
```

**Code location**: `src/data/common.py:19-20`

**Dependencies**: None

**Estimated Effort**: 10 minutes

**Reference**: Existing pattern with `add_positional_features`, `normalize_positional_features`

---

#### Task 3.2: Add CLI Argument for Normalization Control
**Description**: Add command-line flag to enable/disable normalization

**Files to modify:**
- `src/generate_instances.py` (argument parser)

**Implementation Details:**

Add to data argument group (after line 93):
```python
d.add_argument(
    "--normalize_features",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Apply min-max normalization to features (default: True)"
)
```

Update Settings instantiation to use the flag:
```python
settings = Settings(
    add_positional_features=args.add_positional_features,
    normalize_positional_features=args.normalize_positional_features,
    normalize_features=args.normalize_features,  # NEW
)
```

**Code location**: `src/generate_instances.py` (data arguments section, Settings instantiation)

**Dependencies**: Task 3.1 complete

**Estimated Effort**: 15 minutes

**Reference**: Existing pattern in generate_instances.py for other Settings fields

---

#### Task 3.3: Update get_bipartite_graph Signature
**Description**: Add normalize_features parameter to graph generation function

**Files to modify:**
- `src/data/generators.py` (get_bipartite_graph function)

**Implementation Details:**

Update function signature (around line 524):
```python
def get_bipartite_graph(
    lp_path: Path,
    add_pos_feat: bool = True,
    normalize_pos_feat: bool = False,
    normalize_features: bool = True,  # NEW
) -> Tuple:
```

**Code location**: `src/data/generators.py:524`

**Dependencies**: Task 3.1 complete

**Estimated Effort**: 5 minutes

**Reference**: Specification in JEPA_integration_analysis.md (normalization section)

---

#### Task 3.4: Implement Conditional Normalization Logic
**Description**: Apply normalization conditionally based on normalize_features flag

**Files to modify:**
- `src/data/generators.py` (get_bipartite_graph function, normalization section)

**Implementation Details:**

Update normalization section (around lines 732-745):
```python
if normalize_features:
    # Existing normalization logic
    if add_pos_feat and not normalize_pos_feat:
        v_num, v_bits = v_nodes[:, :6], v_nodes[:, 6:]
        v_num = _minmax_normalization(v_num).clamp_(1e-5, 1.0)
        v_nodes = torch.cat([v_num, v_bits], dim=1)
    else:
        v_nodes = _minmax_normalization(v_nodes).clamp_(1e-5, 1.0)

    c_nodes = _minmax_normalization(c_nodes).clamp_(1e-5, 1.0) if c_nodes.numel() > 0 else c_nodes
else:
    # Skip normalization - use raw features
    logger.info("Normalization disabled - using raw features")
```

**Code location**: `src/data/generators.py:732-745`

**Dependencies**: Task 3.3 complete

**Estimated Effort**: 30 minutes

**Reference**: Existing normalization code at lines 732-745 in generators.py

---

#### Task 3.5: Propagate normalize_features Through Call Chain
**Description**: Update all call sites to pass normalize_features parameter

**Files to modify:**
- `src/generate_instances.py` (where get_bipartite_graph is called)
- Any other files calling get_bipartite_graph

**Implementation Details:**

Update call site (around line 116-117):
```python
graph_data = get_bipartite_graph(
    lp_path,
    settings.add_positional_features,
    settings.normalize_positional_features,
    settings.normalize_features,  # NEW
)
```

Use Grep to find all call sites:
```bash
grep -r "get_bipartite_graph" src/ --include="*.py"
```

**Dependencies**: Task 3.3, 3.4 complete

**Estimated Effort**: 30 minutes

**Reference**: Existing call pattern in generate_instances.py:116-117

---

### Phase 4: Integration Testing & Validation

#### Task 4.1: Test JEPA with MLP + EMA Mode
**Description**: Verify JEPA works correctly with MLP architecture using EMA target encoder

**Test Commands:**
```bash
# Joint training (no pre-training)
cd src && python train.py --devices 0 --batch_size 8 --epochs 5 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 0 \
  --jepa_weight 0.2 --jepa_mask_ratio_entries 0.5 --ema_momentum 0.996

# With 2 epochs pre-training
cd src && python train.py --devices 0 --batch_size 8 --epochs 5 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 2 \
  --jepa_weight 0.2 --jepa_mask_ratio_entries 0.5 --ema_momentum 0.996
```

**Validation Checklist:**
- [ ] Training starts without errors
- [ ] WandB logs show `train/loss_jepa` metric
- [ ] During pre-training epochs, only JEPA loss is optimized
- [ ] After pre-training, both losses appear in logs
- [ ] Checkpoint saved successfully
- [ ] Checkpoint contains `target_model` key
- [ ] Model can load from checkpoint

**Dependencies**: All Phase 1 and Phase 2 tasks complete

**Estimated Effort**: 1 hour

---

#### Task 4.2: Test JEPA with MLP + SimSiam Mode
**Description**: Verify JEPA works with SimSiam (no EMA) on MLP architecture

**Test Commands:**
```bash
cd src && python train.py --devices 0 --batch_size 8 --epochs 5 \
  --use_jepa --jepa_mode simsiam --jepa_pretrain_epochs 0 \
  --jepa_weight 0.2 --jepa_mask_ratio_entries 0.5
```

**Validation Checklist:**
- [ ] Training starts without errors
- [ ] No target_model is created (verify in code)
- [ ] WandB logs show `train/loss_jepa` metric
- [ ] Loss values are reasonable (not NaN or exploding)
- [ ] Checkpoint does NOT contain `target_model` key
- [ ] Memory usage lower than EMA mode

**Dependencies**: All Phase 1 and Phase 2 tasks complete

**Estimated Effort**: 45 minutes

---

#### Task 4.3: Test JEPA with GNN + EMA Mode
**Description**: Verify JEPA works correctly with GNN architecture using EMA target encoder

**Test Commands:**
```bash
cd src && python train.py --devices 0 --batch_size 4 --epochs 5 --use_bipartite_graphs \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 1 \
  --jepa_weight 0.2 --jepa_mask_ratio_nodes 0.3 --ema_momentum 0.996
```

**Validation Checklist:**
- [ ] Training starts without errors
- [ ] Node masking works correctly for both constraint and variable nodes
- [ ] WandB logs show `train/loss_jepa` metric
- [ ] Pre-training epoch completes successfully
- [ ] Joint training works after pre-training
- [ ] Checkpoint saved with target_model

**Dependencies**: All Phase 1 and Phase 2 tasks complete

**Estimated Effort**: 1 hour

---

#### Task 4.4: Test JEPA with GNN + SimSiam Mode
**Description**: Verify JEPA works with SimSiam on GNN architecture

**Test Commands:**
```bash
cd src && python train.py --devices 0 --batch_size 4 --epochs 5 --use_bipartite_graphs \
  --use_jepa --jepa_mode simsiam --jepa_pretrain_epochs 0 \
  --jepa_weight 0.2 --jepa_mask_ratio_nodes 0.3
```

**Validation Checklist:**
- [ ] Training starts without errors
- [ ] No target_model created
- [ ] Loss computation works correctly
- [ ] Training completes successfully

**Dependencies**: All Phase 1 and Phase 2 tasks complete

**Estimated Effort**: 45 minutes

---

#### Task 4.5: Test Baseline (No JEPA) Still Works
**Description**: Verify that training without JEPA flag reproduces original behavior

**Test Commands:**
```bash
# MLP baseline
cd src && python train.py --devices 0 --batch_size 8 --epochs 5

# GNN baseline
cd src && python train.py --devices 0 --batch_size 4 --epochs 5 --use_bipartite_graphs
```

**Validation Checklist:**
- [ ] Training works exactly as before
- [ ] No JEPA-related code executes
- [ ] Loss values match historical baselines
- [ ] WandB logs only show KKT loss components
- [ ] Checkpoints do not contain target_model

**Dependencies**: All Phase 2 tasks complete

**Estimated Effort**: 30 minutes

---

#### Task 4.6: Test Feature Normalization - Disabled
**Description**: Generate instances without normalization and verify raw features

**Test Commands:**
```bash
cd src && python generate_instances.py --problems CA --ca_sizes 5 \
  --n_instances 100 --normalize_features false
```

**Validation Checklist:**
- [ ] Instances generate without errors
- [ ] Features are NOT in [0, 1] range (check by loading and inspecting)
- [ ] Constraint features, variable features, edge features all unnormalized
- [ ] Log message confirms normalization was skipped

**Dependencies**: All Phase 3 tasks complete

**Estimated Effort**: 30 minutes

---

#### Task 4.7: Test Feature Normalization - Enabled (Default)
**Description**: Verify default behavior maintains backward compatibility

**Test Commands:**
```bash
# Implicit default
cd src && python generate_instances.py --problems CA --ca_sizes 5 --n_instances 100

# Explicit default
cd src && python generate_instances.py --problems CA --ca_sizes 5 \
  --n_instances 100 --normalize_features true
```

**Validation Checklist:**
- [ ] Instances generate without errors
- [ ] Features are in [0, 1] range (verify by inspection)
- [ ] Behavior identical to pre-implementation baseline
- [ ] No warnings or errors in logs

**Dependencies**: All Phase 3 tasks complete

**Estimated Effort**: 30 minutes

---

#### Task 4.8: Train Model on Unnormalized Data
**Description**: Verify models can train successfully on unnormalized features

**Test Commands:**
```bash
# First generate unnormalized data
cd src && python generate_instances.py --problems CA --ca_sizes 5 \
  --n_instances 500 --normalize_features false

# Then train on it
python train.py --devices 0 --batch_size 8 --epochs 3 --data_root ./data/instances
```

**Validation Checklist:**
- [ ] Model loads data without errors
- [ ] Training completes without NaN losses
- [ ] Gradients are stable (no exploding/vanishing)
- [ ] Validation metrics are computed correctly

**Dependencies**: Tasks 4.6, 4.7 complete

**Estimated Effort**: 1 hour

---

#### Task 4.9: Compare JEPA vs Baseline Performance
**Description**: Run controlled experiments across all 4 training modes to measure JEPA impact on convergence and solution quality

**Experiment Design (All 4 Training Modes):**
```bash
# Mode 1: Baseline (KKT-only, no JEPA)
python train.py --devices 0 --batch_size 8 --epochs 20 --seed 42

# Mode 2: JEPA pre-train → KKT fine-tune (traditional SSL workflow)
python train.py --devices 0 --batch_size 8 --epochs 20 --seed 42 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 3 --jepa_weight 0

# Mode 3: JEPA pre-train → Joint training (representation maintenance)
python train.py --devices 0 --batch_size 8 --epochs 20 --seed 42 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 3 --jepa_weight 0.2

# Mode 4: Joint training from start (no pre-training phase)
python train.py --devices 0 --batch_size 8 --epochs 20 --seed 42 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 0 --jepa_weight 0.2
```

**Metrics to Compare:**
- Validation KKT loss at epochs 5, 10, 15, 20
- Training convergence speed (epochs to reach threshold)
- Final primal feasibility, dual feasibility, stationarity, complementary slackness
- Training time per epoch

**Success Criterion**: JEPA shows measurable improvement in at least one metric

**Dependencies**: Tasks 4.1-4.5 complete

**Estimated Effort**: 2-3 hours (including experiment runtime)

---

### Phase 5: Documentation & Polish

#### Task 5.1: Update README with JEPA Usage Examples
**Description**: Document JEPA feature in main README

**Files to modify:**
- `README.md` or relevant documentation file

**Content to Add:**
- Brief explanation of JEPA self-supervised pre-training
- Usage examples for MLP and GNN
- Configuration options and recommended defaults
- When to use EMA vs SimSiam mode

**Dependencies**: All testing complete

**Estimated Effort**: 1 hour

---

#### Task 5.2: Update README with Normalization Options
**Description**: Document optional normalization feature

**Files to modify:**
- `README.md` or relevant documentation file

**Content to Add:**
- Explanation of normalization toggle
- When to use/skip normalization
- Example commands for both modes
- Impact on reproducibility

**Dependencies**: Tasks 4.6-4.8 complete

**Estimated Effort**: 30 minutes

---

#### Task 5.3: Add Inline Code Comments for JEPA
**Description**: Add docstrings and comments to JEPA utility functions

**Files to modify:**
- `src/models/jepa_utils.py`

**Details**:
- Docstring for each function explaining purpose, parameters, returns
- Inline comments for non-obvious logic (masking, EMA update, loss computation)
- References to relevant papers/algorithms

**Dependencies**: Phase 1, 2 complete

**Estimated Effort**: 1 hour

---

#### Task 5.4: Create Example Configuration Files
**Description**: Provide sample config.yml files for common use cases

**Files to create:**
- `config_jepa_mlp_ema.yml` - MLP with EMA pre-training
- `config_jepa_gnn_simsiam.yml` - GNN with SimSiam joint training
- `config_baseline.yml` - Standard training (no JEPA)

**Content**: Commented configurations with recommended hyperparameters for each scenario

**Dependencies**: Testing complete

**Estimated Effort**: 45 minutes

---

## Codebase Integration Points

### Files to Modify (JEPA Feature)

1. **`src/train.py`**
   - Lines 1-18: Add imports (deepcopy, jepa_utils functions)
   - Lines 21-60: Extend TrainingState class for JEPA loss tracking
   - Lines 84-103: Add JEPA configuration arguments
   - After line 255: Create optional target_model
   - Around line 308-321: Extend checkpoint save/load
   - train_epoch function: Update signature, integrate JEPA loss, add EMA update

2. **`src/models/models.py`**
   - KKTNetMLP class (lines 9-40): Add projector/predictor, embedding methods
   - GNNPolicy class: Add per-node projectors/predictors, embedding methods

3. **`src/config.yml`**
   - Add new `jepa:` section with configuration defaults

### Files to Create (JEPA Feature)

1. **`src/models/jepa_utils.py`** (NEW)
   - Core JEPA functionality: EMA update, loss computation, LP-aware masking
   - ~300-400 lines of code (increased due to sophisticated LP-aware masking logic)
   - Functions:
     - `ema_update()`: ~10 lines
     - `cosine_pred_loss()`: ~5 lines
     - `make_lp_jepa_views()`: ~150-200 lines (row/col/entry masking, safety checks, augmentations)
     - `make_gnn_views()`: ~50 lines
     - `jepa_loss_mlp()`: ~30 lines
     - `jepa_loss_gnn()`: ~30 lines
   - No external dependencies beyond torch, torch_geometric

### Files to Modify (Normalization Feature)

1. **`src/data/common.py`**
   - Lines 19-20: Add `normalize_features` field to Settings dataclass

2. **`src/data/generators.py`**
   - Line 524: Update get_bipartite_graph signature
   - Lines 732-745: Conditional normalization based on flag

3. **`src/generate_instances.py`**
   - Add CLI argument for --normalize_features
   - Update Settings instantiation to pass flag
   - Update get_bipartite_graph call sites

### No Changes Required

- `src/models/losses.py` - KKT loss unchanged
- `src/data/datasets.py` - Dataset classes unchanged
- `src/test_model.py` - Testing script unchanged

---

## Existing Patterns to Follow

### 1. Configuration Management
**Pattern**: YAML config + CLI override with configargparse
**Location**: `src/train.py:79-150`, `src/config.yml`
**Apply to**: JEPA configuration section

### 2. Optional Feature Toggle
**Pattern**: Boolean action flag + conditional instantiation
**Example**: `use_bipartite_graphs` flag at `src/train.py:88`
**Apply to**: `use_jepa` flag

### 3. Argument Registration
**Pattern**: Argument groups with add_argument_group
**Location**: `src/train.py:84-103` (training group), `src/models/models.py:44-73` (GNNPolicy.add_args)
**Apply to**: JEPA arguments in training group

### 4. Loss Function Structure
**Pattern**: Individual term functions + weighted combination + return dict
**Location**: `src/models/losses.py:4-100`
**Apply to**: JEPA loss functions in jepa_utils.py

### 5. Import Organization
**Pattern**: stdlib → third-party → local, no inline imports
**Location**: `src/train.py:1-18`
**Apply to**: All modified and new files

### 6. Model State Management
**Pattern**: state_dict() save/load in checkpoint dict
**Location**: `src/train.py:308-321`
**Apply to**: Target model checkpointing

### 7. WandB Logging
**Pattern**: Log metrics with step parameter, define custom step metric
**Location**: `src/train.py:170-177` (init), `src/train.py:33` (logging)
**Apply to**: JEPA loss logging

### 8. Settings Dataclass
**Pattern**: Frozen dataclass with default values for configuration
**Location**: `src/data/common.py:19-20`
**Apply to**: normalize_features field

---

## Technical Design

### Architecture Diagram: JEPA Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Load Batch      │
                    │  (A, b, c)       │
                    └──────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐          ┌───────────────┐
        │ Original      │          │ Masked        │
        │ Input (x_tgt) │          │ Input (x_ctx) │
        └───────────────┘          └───────────────┘
                │                           │
                ▼                           ▼
        ┌───────────────┐          ┌───────────────┐
        │ Target Model  │          │ Online Model  │
        │ (EMA/Shared)  │          │               │
        └───────────────┘          └───────────────┘
                │                           │
                ▼                           ▼
        ┌───────────────┐          ┌───────────────┐
        │ Encoder       │          │ Encoder       │
        │ z_t = f(x)    │          │ z_o = f(x)    │
        └───────────────┘          └───────────────┘
                │                           │
                ▼                           ▼
        ┌───────────────┐          ┌───────────────┐
        │ Projector     │          │ Projector     │
        │ proj(z_t)     │          │ proj(z_o)     │
        └───────────────┘          └───────────────┘
                │                           │
                │                           ▼
                │                  ┌───────────────┐
                │                  │ Predictor     │
                │                  │ pred(proj)    │
                │                  └───────────────┘
                │                           │
                └─────────┬─────────────────┘
                          ▼
                 ┌─────────────────┐
                 │  JEPA Loss      │
                 │  cosine(p, z_t) │
                 └─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌───────┐      ┌──────────┐      ┌──────────┐
    │ Epoch │      │  Epoch   │      │  Epoch   │
    │ < N   │      │  >= N    │      │  >= N    │
    │       │      │          │      │          │
    │ loss= │      │  KKT     │      │ Total =  │
    │ JEPA  │      │  Loss    │      │ KKT +    │
    │ only  │      │          │      │ λ*JEPA   │
    └───────┘      └──────────┘      └──────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Backprop       │
                 │  Optimizer Step │
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  EMA Update     │
                 │  (if mode=ema)  │
                 └─────────────────┘
```

### Data Flow: Masking Strategies

**MLP (LP-Aware Structured Masking):**

The masking strategy respects the structure of Linear Programming problems where `x = [vec(A), b, c]`:

```
Input: A ∈ R^(M×N), b ∈ R^M, c ∈ R^N, mask_m, mask_n (real sizes)

Step 1: Generate Structure-Aware Masks
  ├─► Row masks M_row: Randomly select rows to mask (constraints)
  │   - Ties: M_A[i,:] = 1 and M_b[i] = 1 (mask entire constraint)
  │   - Ratios: 20% online, 5% target (or 0 for clean target)
  │   - Safety: Keep at least 1 unmasked row
  │
  ├─► Column masks M_col: Randomly select columns to mask (variables)
  │   - Ties: M_A[:,j] = 1 and M_c[j] = 1 (mask entire variable)
  │   - Ratios: 20% online, 5% target (or 0 for clean target)
  │   - Safety: Keep at least 1 unmasked column
  │
  └─► Entry masks M_entry: Random individual A coefficients
      - Applied where neither row nor column is already masked
      - Ratios: 40% online, 10% target (or 0 for clean target)

Step 2: Compose Final Masks
  M_A = M_row ∨ M_col ∨ M_entry
  M_b = M_row
  M_c = M_col

Step 3: Create Two Asymmetric Views

  Online View (context, heavier mask):
    A_on = A ⊙ (1 - M_A_on) + ε_A ⊙ M_A_on
    b_on = b ⊙ (1 - M_b_on) + ε_b ⊙ M_b_on
    c_on = c ⊙ (1 - M_c_on) + ε_c ⊙ M_c_on
    where ε ~ 0 (hard mask) or N(0, 0.01 * median|coefficients|) (noisy mask)

  Target View (lighter mask or clean):
    A_tg = A ⊙ (1 - M_A_tg) + ε_A' ⊙ M_A_tg
    b_tg = b ⊙ (1 - M_b_tg) + ε_b' ⊙ M_b_tg
    c_tg = c ⊙ (1 - M_c_tg) + ε_c' ⊙ M_c_tg
    (Can be clean: M_tg = 0 everywhere)

Step 4: Optional Row Scaling Augmentation
  Sample s_i ~ LogUniform(0.5, 2.0) per constraint
  Scale: A[i,:] ← s_i * A[i,:], b[i] ← s_i * b[i]
  (Preserves feasible set: Ax ≤ b ⟺ (SA)x ≤ Sb for diagonal S ≻ 0)

Step 5: Flatten to Model Input
  x_on = [vec(A_on), b_on, c_on]  →  Online encoder
  x_tg = [vec(A_tg), b_tg, c_tg]  →  Target encoder (EMA, stop-grad)

**Why This Works for LPs:**
- **Row masking**: Forces encoder to infer constraint structure from visible constraints
- **Column masking**: Forces encoder to infer variable relationships from visible variables
- **Entry masking**: Adds fine-grained perturbations to coefficient matrix
- **Tied masking**: Maintains LP semantic coherence (constraint = A row + b value)
- **Row scaling**: Teaches invariance to equivalent problem formulations
- **Asymmetric views**: MAE-style: predict clean/lightly-masked from heavily-masked
```

**GNN (Node-Level Masking):**
```
Input: BipartiteGraph (constraint_features, edge_index, edge_attr, variable_features)
  │
  ├─► Original (target) → All features unchanged
  │
  └─► Context:
      1. Select k_c = mask_ratio * num_constraints random constraint nodes
      2. Select k_v = mask_ratio * num_variables random variable nodes
      3. Zero-out features: ctx_c[mask_c] = 0, ctx_v[mask_v] = 0
      4. Edge features and topology unchanged
      Result: Graph with masked node features
```

### API Endpoints (Configuration)

**Command-Line Interface:**

```bash
# Enable JEPA with EMA (default LP-aware masking)
--use_jepa --jepa_mode ema --ema_momentum 0.996

# Enable JEPA with SimSiam
--use_jepa --jepa_mode simsiam

# Configure loss weighting
--jepa_weight 0.2 --primal_weight 0.1 --dual_weight 0.1 --stationarity_weight 0.6

# Pre-training schedule
--jepa_pretrain_epochs 3  # JEPA-only for 3 epochs, then joint training

# LP-aware masking ratios (MLP)
# Online view (heavier mask)
--jepa_mask_entry_online 0.40 --jepa_mask_row_online 0.20 --jepa_mask_col_online 0.20
# Target view (lighter mask or clean)
--jepa_mask_entry_target 0.10 --jepa_mask_row_target 0.05 --jepa_mask_col_target 0.05
# For clean target view, set all target masks to 0:
--jepa_mask_entry_target 0 --jepa_mask_row_target 0 --jepa_mask_col_target 0

# GNN masking (node-level)
--jepa_mask_ratio_nodes 0.3

# Optional augmentations
--jepa_noisy_mask       # Add Gaussian noise at masked positions
--jepa_row_scaling      # Apply row scaling (s_i ~ LogUniform(0.5, 2.0))

# Disable normalization
--normalize_features false
```

**Configuration File (config.yml):**

```yaml
training:
  use_bipartite_graphs: false
  batch_size: 256
  epochs: 200
  lr: 0.001
  # KKT loss weights
  primal_weight: 0.1
  dual_weight: 0.1
  stationarity_weight: 0.6
  complementary_slackness_weight: 0.2

jepa:
  use_jepa: false
  jepa_mode: "ema"  # or "simsiam"
  jepa_weight: 0.2
  jepa_pretrain_epochs: 3

  # LP-aware masking ratios (MLP)
  # Online view (heavier mask - context)
  jepa_mask_entry_online: 0.40
  jepa_mask_row_online: 0.20
  jepa_mask_col_online: 0.20

  # Target view (lighter mask or clean)
  # Set to 0 for completely clean target
  jepa_mask_entry_target: 0.10
  jepa_mask_row_target: 0.05
  jepa_mask_col_target: 0.05

  # GNN masking (node-level)
  jepa_mask_ratio_nodes: 0.3

  # Augmentation options
  jepa_noisy_mask: false       # Use hard zero masking (vs Gaussian noise)
  jepa_row_scaling: false      # Enable row scaling augmentation

  ema_momentum: 0.996

data:
  problems: ["CA"]
  ca_sizes: [5]
  n_instances: 7000
  data_root: ./data/instances
  normalize_features: true
```

---

## Dependencies and Libraries

### Existing (No New Dependencies Required)

**Core ML Stack:**
- `torch` (2.0+) - Neural network framework, used for all model components
- `torch_geometric` - Graph neural networks, used for BipartiteGraphConvolution
- `rtdl_num_embeddings` - Periodic/PWL embeddings for numeric features

**Utilities:**
- `configargparse` - Configuration management (CLI + YAML)
- `wandb` - Experiment tracking and logging
- `loguru` - Logging
- `numpy` - Numerical operations

**Python Standard Library:**
- `copy.deepcopy` - For creating target model in EMA mode
- `dataclasses` - For Settings configuration
- `typing` - Type hints

### Optional (For Future Enhancements)

- `torch.distributed` - Multi-GPU training (JEPA is well-suited for data parallelism)
- `timm` - If adding vision transformer encoders in future

---

## Testing Strategy

### Unit Tests

**To Be Created:**

1. **`test_jepa_utils.py`** - Test JEPA utility functions
   - `test_ema_update()` - Verify momentum update math
   - `test_cosine_pred_loss()` - Verify loss computation
   - `test_make_mlp_views()` - Verify masking produces correct shapes and mask ratios
   - `test_make_gnn_views()` - Verify node masking for graphs
   - `test_jepa_loss_mlp()` - End-to-end MLP JEPA loss
   - `test_jepa_loss_gnn()` - End-to-end GNN JEPA loss

2. **`test_models.py`** (extend existing) - Test model extensions
   - `test_kkt_mlp_jepa_methods()` - Verify encode_trunk, jepa_embed methods
   - `test_gnn_policy_jepa_methods()` - Verify jepa_embed_nodes method
   - `test_model_output_shapes()` - Ensure forward passes unchanged

3. **`test_normalization.py`** - Test normalization feature
   - `test_normalization_enabled()` - Verify features in [0,1] range
   - `test_normalization_disabled()` - Verify raw features preserved
   - `test_settings_dataclass()` - Verify Settings field handling

### Integration Tests

**To Be Run Manually (Documented in Phase 4):**

1. **JEPA Training Tests**:
   - MLP + EMA mode (joint and pre-training)
   - MLP + SimSiam mode
   - GNN + EMA mode (joint and pre-training)
   - GNN + SimSiam mode
   - Baseline (no JEPA) regression test

2. **Normalization Tests**:
   - Generate with normalization enabled (default)
   - Generate with normalization disabled
   - Train on unnormalized data

3. **Checkpoint Tests**:
   - Save and load with EMA target model
   - Save and load with SimSiam (no target model)
   - Resume training from checkpoint

### Edge Cases to Cover

1. **JEPA Edge Cases**:
   - Mask ratio = 0.0 (no masking)
   - Mask ratio = 1.0 (full masking)
   - Empty batches
   - Very small batch sizes (< 4)
   - Switching modes (EMA ↔ SimSiam) mid-training

2. **Normalization Edge Cases**:
   - All-zero features
   - Features already in [0,1] range
   - Features with extreme outliers
   - Constant features (min = max)

3. **Configuration Edge Cases**:
   - Conflicting flags (e.g., use_jepa=False but jepa_weight > 0)
   - Invalid mode strings
   - Negative loss weights
   - jepa_pretrain_epochs > total epochs

### Performance Benchmarks

**Metrics to Track:**

1. **Training Speed**:
   - Baseline (no JEPA): X seconds/epoch
   - JEPA + EMA: Y seconds/epoch (expect ~1.5-2x baseline)
   - JEPA + SimSiam: Z seconds/epoch (expect ~1.2-1.5x baseline)

2. **Memory Usage**:
   - Baseline: A GB GPU memory
   - JEPA + EMA: B GB (expect ~2x for target model)
   - JEPA + SimSiam: C GB (expect ~1.2x for projector/predictor)

3. **Convergence Speed**:
   - Epochs to reach validation KKT loss < threshold
   - With vs without JEPA pre-training

---

## Success Criteria

### Feature 1: JEPA Self-Supervised Pre-training

#### Functional Requirements

- [x] ✓ Users can enable JEPA with `--use_jepa` flag
- [ ] Pre-training runs for configured epochs before joint training begins
- [ ] Joint training correctly combines JEPA and KKT losses with configurable weighting
- [ ] Both EMA and SimSiam modes produce valid training runs without crashes
- [ ] Checkpoints correctly save and restore all model states (online + target when applicable)
- [ ] Training logs show separate loss terms for JEPA and KKT
- [ ] Feature works for both MLP and GNN architectures with appropriate masking

#### Performance Requirements

- [ ] Models trained with JEPA pre-training show measurable improvement in **at least one** of:
  - Faster KKT loss convergence (fewer epochs to reach threshold)
  - Lower final validation KKT loss
  - Better primal/dual feasibility metrics
  - More stable training (lower loss variance)

- [ ] Training overhead is acceptable:
  - EMA mode: < 2x baseline training time per epoch
  - SimSiam mode: < 1.5x baseline training time per epoch

#### Code Quality Requirements

- [ ] All JEPA code follows existing conventions in CODING_CONVENTIONS.md
- [ ] Docstrings for all new functions
- [ ] No pylint/flake8 errors
- [ ] Backward compatibility maintained (training without JEPA unchanged)

### Feature 2: Optional Feature Normalization

#### Functional Requirements

- [ ] Users can disable normalization by setting `--normalize_features false`
- [ ] Generated instances with normalization disabled contain raw, unnormalized features
- [ ] Generated instances with normalization enabled behave identically to current behavior
- [ ] Default behavior (normalization enabled) ensures backward compatibility
- [ ] Models train successfully on both normalized and unnormalized data
- [ ] No errors or warnings when loading/processing unnormalized data

#### Validation Requirements

- [ ] Features with normalization OFF: values outside [0, 1] range (verify by inspection)
- [ ] Features with normalization ON: values in [0, 1] range (verify by inspection)
- [ ] Training on unnormalized data: no NaN losses, stable gradients

### Overall Success

- [ ] All integration tests pass (Tasks 4.1-4.9)
- [ ] Documentation updated with usage examples
- [ ] Example configuration files created
- [ ] No regressions in baseline performance

---

## Notes and Considerations

### Important Implementation Notes

1. **Training Mode Selection**:
   The system supports 4 training modes controlled by `jepa_pretrain_epochs` and `jepa_weight`:
   - **Mode 1 (Baseline)**: No JEPA flags → Standard KKT training (control group)
   - **Mode 2 (SSL Fine-tune)**: `jepa_pretrain_epochs > 0, jepa_weight = 0` → Traditional SSL: pre-train representations, then fine-tune with KKT only
   - **Mode 3 (Pre-train + Joint)**: `jepa_pretrain_epochs > 0, jepa_weight > 0` → Pre-train then maintain representations with auxiliary JEPA loss
   - **Mode 4 (Joint from Start)**: `jepa_pretrain_epochs = 0, jepa_weight > 0` → Combined training from epoch 0

   **When to use each mode:**
   - Mode 2 (SSL fine-tune) follows traditional self-supervised learning (like BERT, SimCLR)
   - Mode 3 (pre-train + joint) prevents representation collapse during task-specific training
   - Mode 4 (joint from start) is faster but may learn less general representations
   - Mode 1 (baseline) is essential for experimental comparison

2. **JEPA Loss Weighting**:
   - Start with `jepa_weight=0.2` (recommended) for Modes 3 & 4
   - Set `jepa_weight=0` for Mode 2 (traditional SSL fine-tuning)
   - Too high: May overwhelm KKT loss, model ignores KKT conditions
   - Too low: JEPA has minimal effect on representations
   - Tuning may be needed per problem type

3. **LP-Aware Masking Strategy (MLP)**:
   The masking respects LP structure by tying masks to semantic units:
   - **Row masking (constraints)**: When masking row i, also mask b[i]. This hides entire constraint: A[i,:]x ≤ b[i]
   - **Column masking (variables)**: When masking column j, also mask c[j]. This hides entire variable: x[j] and its objective coefficient
   - **Entry masking**: Individual A[i,j] coefficients masked independently
   - **Composition**: M_A = M_row ∨ M_col ∨ M_entry maintains structural coherence

   **Asymmetric view ratios:**
   - Online (context/heavy): 40% entries + 20% rows + 20% cols → ~60-70% of A masked
   - Target (clean/light): 10% entries + 5% rows + 5% cols → ~15-20% of A masked (or 0 for clean)
   - **Safety rules**: Always keep ≥1 unmasked row AND ≥1 unmasked column
   - **Padding respect**: Only mask within real problem size (honor mask_m, mask_n)

   **Why asymmetric?** MAE-style: model predicts clean/lightly-masked target from heavily-masked context. Forces learning of problem structure, not memorization.

   **Optional augmentations:**
   - Noisy masking: ε ~ N(0, 0.01 × median|coefficients|) instead of zeros
   - Row scaling: Multiply constraints by s_i ~ LogUniform(0.5, 2.0) → teaches invariance to equivalent formulations (Ax≤b ⟺ SAx≤Sb)

   **GNN**: Simple node-level masking (30% of nodes) since graph structure already encodes relationships

4. **Architecture: Encoder + Projector + Predictor**:
   JEPA uses a three-component architecture (BYOL/I-JEPA style):
   - **Encoder**: Task-specific trunk (shared with KKT heads). Outputs hidden representation.
   - **Projector**: Maps hidden → embedding space. Applied to both online and target. L2-normalized output.
   - **Predictor**: Online-only. Maps online embedding → prediction. L2-normalized output. **Key asymmetry!**

   **Forward passes:**
   ```
   Online:  x_on → encoder → proj → L2-norm → z_on → pred → L2-norm → p_on
   Target:  x_tg → encoder* (EMA) → proj* (EMA) → L2-norm → z_tg [stop-grad]
   Loss:    2 - 2·cos(p_on, z_tg)
   ```

   **Why predictor?** Prevents collapse. Without it, model could learn trivial constant embeddings. Predictor creates asymmetry between online and target paths.

5. **EMA Momentum**:
   - 0.996 is standard for vision tasks (from BYOL, I-JEPA)
   - Higher (e.g., 0.999): Slower target updates, more stable but less responsive
   - Lower (e.g., 0.99): Faster updates, potentially less stable

6. **Pre-training Duration**:
   - 3 epochs is a reasonable default for exploration
   - Longer pre-training (10-50 epochs) may be beneficial if:
     - Training data is very limited
     - Problem structure is complex
     - You have computational budget
   - Can also use 0 epochs (joint training from start) for quick experiments

7. **Normalization Trade-offs**:
   - **Keep normalization ON** (default) when:
     - Features have vastly different scales
     - Using standard neural network architectures
     - Starting new research with unknown properties
   - **Turn normalization OFF** when:
     - Reproducing papers that don't normalize
     - Features already on similar scales
     - Studying impact of scale information on learning
     - Problem domain has meaningful absolute scales

### Potential Challenges

1. **JEPA May Not Always Help**:
   - If KKT training data is abundant and diverse, JEPA may provide minimal benefit
   - On very small problems (e.g., CA with size=5), structure may be too simple for JEPA to learn useful patterns
   - Success depends on problem complexity and data availability

2. **Hyperparameter Sensitivity**:
   - JEPA introduces several hyperparameters (weight, momentum, mask ratio, pretrain epochs)
   - May need tuning per problem type (CA, IS, SC, CFL, RND)
   - Recommend starting with defaults, then systematic grid search if needed

3. **Memory Constraints with EMA**:
   - Target model doubles memory footprint
   - For large GNN models, may need to reduce batch size or use SimSiam mode
   - Monitor GPU memory usage during development

4. **Checkpoint Size**:
   - Saving target model increases checkpoint size by ~2x
   - May need to clean up old checkpoints more aggressively
   - Consider saving target_model only in "best.pt", not "last.pt"

5. **Unnormalized Data Training**:
   - May require different learning rates (features on different scales affect gradient magnitudes)
   - May need gradient clipping to prevent instability
   - Recommend starting with normalized data for baseline, then experiment

### Future Enhancements

**Possible Extensions (Out of Scope for Sprint 1):**

1. **Advanced Masking Strategies**:
   - Block masking (mask contiguous blocks of constraints/variables)
   - Learned masking (model predicts what to mask)
   - Multi-view masking (multiple masked views per sample)

2. **Adaptive Loss Weighting**:
   - Uncertainty-based weighting (learn λ for JEPA/KKT dynamically)
   - Curriculum learning (gradually increase KKT weight over training)

3. **Alternative SSL Objectives**:
   - Barlow Twins (redundancy reduction)
   - VICReg (variance-invariance-covariance regularization)
   - Contrastive learning with negative pairs

4. **Normalization Options**:
   - Standardization (z-score) in addition to min-max
   - Per-feature normalization statistics
   - Layer normalization within model (instead of data preprocessing)

5. **Multi-GPU Training**:
   - JEPA is well-suited for data parallelism
   - Target model can be on separate GPU
   - Potential for significant speedup on large problems

---

## References

### Academic Papers

1. **I-JEPA**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (Assran et al., CVPR 2023)
   - Official implementation: https://github.com/facebookresearch/ijepa
   - Key concepts: Block masking, latent prediction, EMA teacher

2. **BYOL**: "Bootstrap Your Own Latent" (Grill et al., NeurIPS 2020)
   - EMA momentum encoder concept
   - Self-supervised without negative pairs

3. **SimSiam**: "Exploring Simple Siamese Representation Learning" (Chen & He, CVPR 2021)
   - Stop-gradient alternative to EMA
   - Simpler, lighter architecture

4. **V-JEPA**: "V-JEPA: The next step toward advanced machine intelligence" (Meta AI, 2024)
   - Video extension of JEPA
   - Insights on masking strategies and efficiency

### Technical Documentation

1. **Project-Specific**:
   - `/home/joachim-verschelde/Repos/KKT_MPNN/PRPs/context/JEPA_integration_analysis.md` - Detailed implementation specification
   - `/home/joachim-verschelde/Repos/KKT_MPNN/claude/CODING_CONVENTIONS.md` - Coding standards

2. **External**:
   - PyTorch documentation: https://pytorch.org/docs/stable/index.html
   - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
   - WandB Python SDK: https://docs.wandb.ai/

### Codebase Reference Files

**Critical Files for Reference:**

1. `src/train.py:21-60` - TrainingState class pattern
2. `src/train.py:84-103` - Argument group registration
3. `src/train.py:251-255` - Model selection based on flag
4. `src/train.py:308-321` - Checkpoint save/load
5. `src/models/models.py:9-40` - KKTNetMLP architecture
6. `src/models/models.py:89-280` - GNNPolicy architecture
7. `src/models/losses.py:4-100` - Modular loss function pattern
8. `src/data/common.py:19-20` - Settings dataclass
9. `src/data/generators.py:108-114` - Normalization function
10. `src/data/generators.py:732-745` - Conditional normalization application

---

## Appendix: Implementation Checklist

### Pre-Implementation

- [x] Read and understand requirements documents
- [x] Review codebase patterns and conventions
- [x] Research JEPA best practices
- [x] Create implementation plan

### Phase 1: JEPA Foundation

- [ ] Task 1.1: Create jepa_utils.py with core functions
- [ ] Task 1.2: Extend KKTNetMLP with JEPA components
- [ ] Task 1.3: Extend GNNPolicy with JEPA components
- [ ] Task 1.4: Add JEPA configuration arguments

### Phase 2: JEPA Training Integration

- [ ] Task 2.1: Create optional EMA target model
- [ ] Task 2.2: Update TrainingState for multi-loss tracking
- [ ] Task 2.3: Extend checkpoint save/load
- [ ] Task 2.4: Update train_epoch signature
- [ ] Task 2.5: Integrate JEPA loss computation
- [ ] Task 2.6: Add EMA update after optimizer step

### Phase 3: Optional Normalization

- [ ] Task 3.1: Add normalize_features to Settings
- [ ] Task 3.2: Add CLI argument for normalization
- [ ] Task 3.3: Update get_bipartite_graph signature
- [ ] Task 3.4: Implement conditional normalization logic
- [ ] Task 3.5: Propagate normalize_features through call chain

### Phase 4: Testing & Validation

- [ ] Task 4.1: Test JEPA with MLP + EMA
- [ ] Task 4.2: Test JEPA with MLP + SimSiam
- [ ] Task 4.3: Test JEPA with GNN + EMA
- [ ] Task 4.4: Test JEPA with GNN + SimSiam
- [ ] Task 4.5: Test baseline (no JEPA) still works
- [ ] Task 4.6: Test normalization disabled
- [ ] Task 4.7: Test normalization enabled (default)
- [ ] Task 4.8: Train model on unnormalized data
- [ ] Task 4.9: Compare JEPA vs baseline performance

### Phase 5: Documentation

- [ ] Task 5.1: Update README with JEPA usage
- [ ] Task 5.2: Update README with normalization options
- [ ] Task 5.3: Add inline code comments
- [ ] Task 5.4: Create example configuration files

### Final Review

- [ ] All tests passing
- [ ] No regressions in baseline
- [ ] Documentation complete
- [ ] Code reviewed for conventions
- [ ] Ready for production use

---

**This plan is ready for execution with `/execute-plan PRPs/sprint-1-implementation-plan.md`**

---

*Plan created: 2025-11-12*
*Plan updated: 2025-11-13 (added LP-aware masking strategy)*
*Estimated total effort: 30-35 hours (increased due to sophisticated LP-aware masking implementation)*
*Recommended implementation order: Sequential (Phase 1 → 2 → 3 → 4 → 5)*
