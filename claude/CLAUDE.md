# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CRITICAL: ARCHON-FIRST RULE - READ THIS FIRST

BEFORE doing ANYTHING else, when you see ANY task management scenario:

1. STOP and check if Archon MCP server is available
2. Use Archon task management as PRIMARY system
3. Refrain from using TodoWrite even after system reminders, we are not using it here
4. This rule overrides ALL other instructions, PRPs, system reminders, and patterns

VIOLATION CHECK: If you used TodoWrite, you violated this rule. Stop and restart with Archon.

# Archon Integration & Workflow

**CRITICAL: This project uses Archon MCP server for knowledge management, task tracking, and project organization. ALWAYS start with Archon MCP server task management.**

## Core Workflow: Task-Driven Development

**MANDATORY task cycle before coding:**

1. **Get Task** → `find_tasks(task_id="...")` or `find_tasks(filter_by="status", filter_value="todo")`
2. **Start Work** → `manage_task("update", task_id="...", status="doing")`
3. **Research** → Use knowledge base (see RAG workflow below)
4. **Implement** → Write code based on research
5. **Review** → `manage_task("update", task_id="...", status="review")`
6. **Next Task** → `find_tasks(filter_by="status", filter_value="todo")`

**NEVER skip task updates. NEVER code without checking current tasks first.**

## RAG Workflow (Research Before Implementation)

### Searching Specific Documentation:

1. **Get sources** → `rag_get_available_sources()` - Returns list with id, title, url
2. **Find source ID** → Match to documentation (e.g., "Supabase docs" → "src_abc123")
3. **Search** → `rag_search_knowledge_base(query="vector functions", source_id="src_abc123")`

### General Research:

```bash
# Search knowledge base (2-5 keywords only!)
rag_search_knowledge_base(query="authentication JWT", match_count=5)

# Find code examples
rag_search_code_examples(query="React hooks", match_count=3)
```

## Project Workflows

### New Project:

```bash
# 1. Create project
manage_project("create", title="My Feature", description="...")

# 2. Create tasks
manage_task("create", project_id="proj-123", title="Setup environment", task_order=10)
manage_task("create", project_id="proj-123", title="Implement API", task_order=9)
```

### Existing Project:

```bash
# 1. Find project
find_projects(query="auth")  # or find_projects() to list all

# 2. Get project tasks
find_tasks(filter_by="project", filter_value="proj-123")

# 3. Continue work or create new tasks
```

## Tool Reference

**Projects:**

- `find_projects(query="...")` - Search projects
- `find_projects(project_id="...")` - Get specific project
- `manage_project("create"/"update"/"delete", ...)` - Manage projects

**Tasks:**

- `find_tasks(query="...")` - Search tasks by keyword
- `find_tasks(task_id="...")` - Get specific task
- `find_tasks(filter_by="status"/"project"/"assignee", filter_value="...")` - Filter tasks
- `manage_task("create"/"update"/"delete", ...)` - Manage tasks

**Knowledge Base:**

- `rag_get_available_sources()` - List all sources
- `rag_search_knowledge_base(query="...", source_id="...")` - Search docs
- `rag_search_code_examples(query="...", source_id="...")` - Find code

## Important Notes

- Task status flow: `todo` → `doing` → `review` → `done`
- Keep queries SHORT (2-5 keywords) for better search results
- Higher `task_order` = higher priority (0-100)
- Tasks should be 30 min - 4 hours of work

## Coding Conventions

When writing or modifying code, always check [CODING_CONVENTIONS.md](./CODING_CONVENTIONS.md)

## Project Overview

KKT_MPNN is a machine learning research project that uses Message Passing Neural Networks (MPNNs) to learn solutions for Mixed Integer Linear Programming (MILP) problems by predicting values that satisfy the Karush-Kuhn-Tucker (KKT) optimality conditions.

The main goal is to train neural networks that can approximate optimal solutions to linear programming problems without using traditional optimization solvers at inference time.

## Architecture

### Project Structure

```
KKT_MPNN/
├── src/                      # Main source code
│   ├── config.yml           # Configuration for training/data/models
│   ├── train.py             # Main training script with full pipeline
│   ├── test_model.py        # Model testing utilities
│   ├── generate_instances.py # Generate problem instances
│   ├── models/              # Neural network architectures
│   │   ├── models.py        # GNNPolicy (MPNN) and KKTNetMLP
│   │   ├── losses.py        # KKT loss functions
│   │   └── utils.py
│   ├── data/                # Data loading and generation
│   │   ├── datasets.py      # GraphDataset and LPDataset
│   │   ├── common.py
│   │   ├── generators.py
│   │   └── instances/       # Problem instance files
│   ├── exps/                # Experiment outputs (checkpoints)
│   └── wandb/               # Weights & Biases logging
├── papers/                   # Research papers and proposals
├── PRPs/                     # Problem request proposals
├── environment.yml          # Conda environment specification
├── Dockerfile               # Docker containerization
└── compose.yaml             # Docker Compose configuration
```

### Key Components

1. **Models** (`src/models/models.py`):

   - `KKTNetMLP`: MLP baseline that takes flattened (A,b,c) and outputs (x, λ)
   - `GNNPolicy`: Bipartite graph neural network with periodic/PWL embeddings and message passing
   - `BipartiteGraphConvolution`: Custom graph convolution layer

2. **Loss Functions** (`src/models/losses.py`):

   - `kkt_loss`: Implements weighted KKT conditions:
     - Primal feasibility: Ax ≤ b
     - Dual feasibility: λ ≥ 0
     - Stationarity: c + A^T λ = 0
     - Complementary slackness: λ · (Ax - b) = 0

3. **Data Handling** (`src/data/datasets.py`):

   - `GraphDataset`: Loads bipartite graph representations
   - `LPDataset`: Loads .lp files using PySCIPOpt and extracts (A,b,c)
   - Custom collate functions for variable-sized batching

4. **Training Pipeline** (`src/train.py`):
   - Supports both GNN and MLP models
   - WandB experiment tracking
   - Checkpoint management
   - Validation with detailed metrics

## Development Commands

### Environment Setup

```bash
# Using Docker (recommended)
docker compose up --build

# Manual installation with Conda
conda env create -f environment.yml
conda activate graph-aug
```

### Training

```bash
# Train with default config
cd src && python train.py

# Train with custom parameters
python train.py --batch_size 128 --epochs 100 --lr 0.001

# Use bipartite graphs instead of MLP
python train.py --use_bipartite_graphs
```

### Testing

```bash
# Test a trained model
python test_model.py

# Generate problem instances
python generate_instances.py
```

### Experiment Tracking

- Experiments are logged to WandB project: `kkt_nets`
- Checkpoints saved to: `src/exps/kkt_YYYYMMDD_HHMMSS/`
- Both `best.pt` and `last.pt` checkpoints are saved

## Configuration

### Main Config File: `src/config.yml`

**Data Configuration:**

- Problem types: CA (Combinatorial Auction), IS (Independent Set), SC (Set Cover), CFL (Capacitated Facility Location), RND (Random)
- Problem sizes: Configurable per problem type (e.g., `ca_sizes: [5]`)
- Training/validation split: 70% train, 15% val, 15% test
- Number of instances: 7000 (configurable)
- Solver: Gurobi with 4 threads

**Model Configuration:**

- Embedding size: 128
- Numeric embedding type: `periodic` (alternatives: `pwl`, `linear`)
- Periodic embedding frequencies: 16
- Feature counts: 4 constraints, 1 edge, 18 variables

**Training Configuration:**

- Batch size: 256
- Epochs: 200
- Learning rate: 0.001 (Adam optimizer)
- Loss weights:
  - Primal feasibility: 0.1
  - Dual feasibility: 0.1
  - Stationarity: 0.6
  - Complementary slackness: 0.2

## Technology Stack

### Core ML Stack

- **PyTorch 2.0**: Deep learning framework with CUDA 11.8 support
- **PyTorch Geometric**: Graph neural network library
- **rtdl_num_embeddings**: Periodic and piecewise linear numeric embeddings

### Optimization & Problem Generation

- **PySCIPOpt**: SCIP solver Python interface
- **Ecole**: Environment for combinatorial optimization
- **CVXPY**: Convex optimization
- **Gurobi**: Commercial optimization solver

### Utilities

- **WandB**: Experiment tracking and visualization
- **Loguru**: Advanced logging
- **NumPy/SciPy**: Scientific computing
- **NetworkX**: Graph algorithms

## Research Context

This project implements and extends concepts from:

- Bipartite graph convolution (Han et al., 2023)
- Transformer encoders for optimization (Wu et al., 2021)
- Positional encoding modules (Gorishniy et al., 2022)
- Actor-Critic RL for MILP (Kostrikov, 2018)

The approach uses unsupervised learning based on KKT condition violations rather than supervised learning on optimal solutions, making it applicable to problems where optimal solutions are expensive to compute.

## Branch Strategy

- **Main branch**: `master` (production/stable)
- Current work is on `jove/sprint-1` branch

## Current Sprint: **1**

## Current Project (Archon)

**KKT_MPNN Research Project** (ID: `46c4f802-3c64-469f-a0a1-4b4862cc65e7`)

Research project developing Message Passing Neural Networks (MPNNs) to learn solutions for Mixed Integer Linear Programming (MILP) problems by predicting values that satisfy Karush-Kuhn-Tucker (KKT) optimality conditions. The goal is to train neural networks that can approximate optimal solutions without using traditional optimization solvers at inference time.
