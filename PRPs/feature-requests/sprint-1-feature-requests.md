# Sprint 1 Feature Requests

## Feature: Self-Supervised JEPA Pre-training for Improved KKT Prediction

**Description:**
Enable self-supervised learning through Joint-Embedding Predictive Architecture (JEPA) to improve the quality of learned representations before or during KKT-based training. This feature allows the model to learn robust structural patterns from optimization problems without relying solely on KKT condition violations, potentially improving convergence speed and solution quality, especially on limited training data.

JEPA works by training the model to predict representations of complete problem instances from partially masked/corrupted versions, forcing it to learn meaningful structural features of linear programming problems.

**Key Requirements:**

- **Flexible Training Modes**: Support both pure JEPA pre-training (representation learning only) and joint training (JEPA + KKT loss combined)

- **Multiple Architecture Support**: Work seamlessly with both MLP baseline models (instance-level masking) and GNN models (node-level masking)

- **Configurable Training Strategy**: Allow users to choose between:

  - EMA (Exponential Moving Average) mode: Uses a slowly-updated target encoder for stable learning (BYOL/I-JEPA style)
  - SimSiam mode: Lighter alternative without target encoder, using stop-gradient instead

- **Masking Strategies**:

  - For MLP: Mask random entries in the flattened problem representation (constraint matrix, RHS, objective)
  - For GNN: Mask random constraint and variable nodes while preserving graph topology

- **Adjustable Training Schedule**: Configure how many epochs to use pure JEPA pre-training before switching to joint training, and control the relative weighting between JEPA and KKT losses

- **Checkpoint Compatibility**: Save and restore both online and target models (when using EMA mode) to support training resumption

**Integration Points:**

- Training pipeline (command-line arguments and configuration)
- Model architectures (both KKTNetMLP and GNNPolicy)
- Loss computation and backpropagation
- Checkpoint saving/loading system
- Experiment tracking and logging (WandB)

**Success Criteria:**

- Users can enable JEPA training with a single `--use_jepa` flag
- Pre-training runs successfully for configured number of epochs before joint training begins
- Joint training correctly combines both JEPA and KKT losses with configurable weighting
- Both EMA and SimSiam modes produce valid training runs without crashes
- Checkpoints correctly save and restore all model states (online + target when applicable)
- Training logs show separate loss terms for JEPA and KKT for monitoring
- Models trained with JEPA pre-training show measurable improvement in KKT loss convergence or final solution quality compared to baseline (on validation set)
- Feature works for both MLP and GNN model architectures with appropriate masking strategies

**Recommended Configuration Defaults:**

- JEPA weight: 0.2 (relative to KKT loss)
- EMA momentum: 0.996 (when using EMA mode)
- MLP mask ratio: 0.5 (fraction of entries masked)
- GNN mask ratio: 0.3 (fraction of nodes masked)
- Pre-training epochs: 3 (before joint training)

**Expected User Workflows:**

1. **MLP with JEPA pre-training then joint training:**
   Enable JEPA with EMA mode, run 3 epochs of pure representation learning, then continue with combined JEPA+KKT loss

2. **GNN with immediate joint training:**
   Enable JEPA in SimSiam mode (no EMA overhead), train with combined loss from epoch 0

3. **Baseline comparison:**
   Run with and without JEPA to measure representation quality improvements on validation metrics

---

**Technical Context:**

For detailed implementation guidance, architecture decisions, and code-level specifications, see:

- [JEPA Integration Analysis](../context/JEPA_integration_analysis.md)

---

## Feature: Optional Feature Normalization for Problem Instances

**Description:**
Allow users to control whether problem instance features are normalized during data generation. Currently, all generated problem instances undergo min-max normalization automatically, which may not be desirable for all experimental setups or research questions. This feature gives researchers the flexibility to work with raw, unnormalized features or apply normalization based on their specific needs.

**Why This is Needed:**
- Different research experiments may require different data preprocessing strategies
- Some models may perform better with unnormalized features that preserve the original problem structure and scale
- Researchers need to compare model performance with and without normalization to understand its impact
- Reproducibility of results from papers that use unnormalized features

**Key Requirements:**

- **Configuration Flag**: Provide a simple command-line flag or configuration option to enable/disable feature normalization during data generation

- **Default Behavior**: Maintain backward compatibility by keeping normalization enabled by default (current behavior)

- **Applied Consistently**: When disabled, normalization should be skipped for all relevant feature types (constraint features, variable features, edge features, objective coefficients, etc.)

- **Documentation**: Clear indication in generated data metadata whether normalization was applied

**Integration Points:**

- Data generation pipeline (problem instance generators)
- Configuration system (config files and command-line arguments)
- Dataset loading utilities

**Success Criteria:**

- Users can disable normalization by setting a flag (e.g., `--normalize_features false` or similar)
- Generated instances with normalization disabled contain raw, unnormalized feature values
- Generated instances with normalization enabled behave identically to current behavior
- Default behavior (normalization enabled) ensures backward compatibility with existing experiments
- Models can be trained successfully on both normalized and unnormalized data
- No errors or warnings occur when loading/processing unnormalized data
- Experimental results can be reproduced with and without normalization to measure its impact
