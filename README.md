# BIJEPA

Code for the MSc thesis *BIJEPA: Label-Free Representation Learning and Decomposition for Linear Programs on Bipartite Graphs* (Joachim Verschelde, Open University, 2026).

This README is a reproducibility guide. For the research context, see the thesis.

---

## 1. Environment

All commands below assume the `graph-aug` conda environment is active.

### Option A — Conda (used for all reported experiments)

```bash
conda env create -f environment.yml
conda activate graph-aug
```

### Option B — Docker

```bash
docker compose up --build
```

Uncomment the GPU section in `compose.yaml` for CUDA.

### Working directory

All `python -m ...` commands must be run **from the `src/` directory**:

```bash
cd src
```

### Weights & Biases

Training jobs log to W&B. Either run `wandb login` beforehand, or export `WANDB_MODE=offline` to disable uploads.

---

## 2. Reproducing the thesis end to end

The pipeline has five stages. Each stage depends on the previous one.

```
(1) generate instances  →  (2) pretrain BiJEPA  →  (3) finetune (RQ1)
                                              ↘  (4) precompute splits  →  (5) split finetune + RQ3
```

All stages are driven by YAML configs under `src/configs/`. Defaults in the configs match the settings reported in the thesis.

### Stage 1 — Generate LP instances

```bash
cd src
python -m jobs.generate_instances --config configs/data_generation/generate_instances_milp.yml
```

Generates LP relaxations for four families — Independent Set (IS), Set Cover (SC), Combinatorial Auction (CA), Capacitated Facility Location (CFL) — at sizes `n ∈ {10, 50, 200}` (CA uses `{5, 25, 100}`), with a 70/15/15 train/val/test split. Instances are written to `./data/instances/milp/` by default. Adjust `n_instances`, sizes, or output path by editing the config.

### Stage 2 — BiJEPA pretraining (RQ1, C1)

Full-graph pretraining, one config per size:

```bash
python -m jobs.pretrain --config configs/pretrain/pretrain_ALL_10.yml
python -m jobs.pretrain --config configs/pretrain/pretrain_ALL_50.yml
python -m jobs.pretrain --config configs/pretrain/pretrain_ALL_200.yml
```

Split-graph pretraining (only needed for the `PRETRAINED` split variant in Stage 5):

```bash
python -m jobs.pretrain_split --config configs/pretrain_split/pretrain_split_bijepa.yml
# optional warm-started variant:
python -m jobs.pretrain_split --config configs/pretrain_split/pretrain_split_bijepa_warm.yml
```

Checkpoints are saved to paths specified in the configs. Note the checkpoint paths — they are needed by finetuning configs.

### Stage 3 — KKT-guided finetuning (RQ1, C2)

For each `(family, size)` combination, four variants are compared:

- `mlp_baseline` — dense MLP baseline
- `gnn_baseline` — bipartite GNN trained from scratch
- `gnn_frozen_encoder` — BiJEPA-initialized, encoder frozen
- `gnn_full_finetune` — BiJEPA-initialized, full finetuning

Config layout: `configs/finetune/finetune_<FAMILY>_<SIZE>/finetune_<FAMILY>_<SIZE>_<variant>.yml`.

Example (one cell of the RQ1 grid):

```bash
python -m jobs.finetune --config configs/finetune/finetune_CA_50/finetune_CA_50_gnn_full_finetune.yml
```

To reproduce the full RQ1 grid, run every config under `configs/finetune/`. Available family/size directories include `finetune_CA_{10,50,200}`, `finetune_ALL_{10,50,200}`, and the MILP grid under `configs/finetune/milp/{CA,CFL,IS,SC}/...`.

Before running a finetune config that uses a BiJEPA checkpoint, make sure its `encoder_checkpoint:` field points at a checkpoint produced in Stage 2.

### Stage 4 — Decomposition analysis (RQ2, C3)

Precompute METIS partitions and halo subgraphs at each halo depth:

```bash
python -m jobs.precompute_splits --config configs/precompute_splits/block-5-halo-0-CA-100.yml
python -m jobs.precompute_splits --config configs/precompute_splits/block-5-halo-1-CA-100.yml
python -m jobs.precompute_splits --config configs/precompute_splits/block-5-halo-2-CA-100.yml
```

Evaluate embedding recovery vs. halo depth:

```bash
python -m jobs.eval_halo_embedding_recovery
python -m jobs.plot_halo_embedding_results
```

Outputs: `src/eval_halo_embedding.csv`, `src/eval_halo_embedding_coupling.csv`, and plots used in the RQ2 analysis.

### Stage 5 — Split architecture and RQ3 (C4)

Train the three split variants at each halo depth `h ∈ {0, 1, 2}`:

```bash
# RAW split baseline (composer skipped)
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_RAW_h0.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_RAW_h1.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_RAW_h2.yml

# Block-GNN composer, from-scratch encoder
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_COMPOSER_h0.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_COMPOSER_h1.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_COMPOSER_h2.yml

# Block-GNN composer, split-pretrained encoder (requires Stage 2 split pretraining)
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_PRETRAINED_h0.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_PRETRAINED_h1.yml
python -m jobs.finetune_split --config configs/finetune_splits/finetune_split_CA_200_PRETRAINED_h2.yml
```

Once all nine split runs and the full-graph baseline from Stage 3 are trained, produce the RQ3 comparison table and figure:

```bash
python -m jobs.run_rq3_experiments --output_dir ./results/rq3
```

The script loads each checkpoint, runs `eval_epoch` to collect KKT metrics, and writes a CSV + grouped bar chart to `--output_dir`.

---

## 3. Loading a pretrained encoder in your own code

Always go through the canonical helper so the online encoder (not the target) is loaded and config stays consistent:

```python
from models.utils import LeJepaEncoderModule

encoder = LeJepaEncoderModule.load_model_and_encoder(checkpoint_path, ...)
```

---

## 4. Config overrides

All jobs use `configargparse`, so any YAML field can be overridden on the command line, e.g.:

```bash
python -m jobs.finetune \
  --config configs/finetune/finetune_CA_50/finetune_CA_50_gnn_full_finetune.yml \
  --epochs 100 --batch_size 32
```

---

## 5. Repo map

```
src/
  configs/
    data_generation/      Stage 1 configs
    pretrain/             Stage 2 full-graph pretraining configs
    pretrain_split/       Stage 2 split pretraining configs
    finetune/             Stage 3 finetuning configs (LP + MILP grids)
    precompute_splits/    Stage 4 partition/halo configs
    finetune_splits/      Stage 5 split finetuning configs
  data/                   Instance generators, METIS/halo dataset
  models/                 GNN encoder, split/composer heads, LeJEPA loader
  jobs/                   All entry points invoked via `python -m jobs.<name>`
  metrics/                KKT residuals, objective/optimality gaps
  sweeps/                 W&B sweep definitions
  tests/                  Unit tests (pytest)
  notebooks/              RQ1–RQ3 analysis notebooks
```

---

## 6. Tests

```bash
cd src
pytest tests/
```
