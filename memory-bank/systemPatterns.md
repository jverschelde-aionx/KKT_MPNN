# System Patterns: KKT_MPNN

## Architecture Overview

### Two-Network Pipeline for MILPs
```
Input (A,b,c) → LP_Network → Lower Bound (O_d) → MILP_Network → Integer Solution
                    ↓                              ↑
                 KKT Loss                    Upper Bound Update
```

**LP Network**: Solves continuous relaxation using KKT-based loss
**MILP Network**: Finds integer-feasible solutions within computed bounds

### Bipartite Graph Representation
- **Constraint Nodes**: Represent each row of constraint matrix A
- **Variable Nodes**: Represent each decision variable
- **Edges**: Connect constraints to variables where A[i,j] ≠ 0
- **Node Features**: 
  - Variable nodes: objective coefficients c_j, bounds
  - Constraint nodes: right-hand sides b_i, constraint types

## Key Design Patterns

### 1. KKT-Based Loss Functions
**Pattern**: Convert optimization constraints into differentiable penalties

**LP Loss Function**:
```
L(x,u) = ||A^T u + c||² + ||ReLU(Ax-b)||² + ||ReLU(-u)||² + ||u⊙(Ax-b)||²
```

**MILP Loss Function**:
```
L_MILP(x) = ||ReLU(Ax-b)||² + ReLU(-c^T x + O_d) + ReLU(c^T x - O_p) + ||x⊙(1-x)||²
```

**Key Properties**:
- All terms ≥ 0
- Global minimum = 0 iff solution satisfies all conditions
- No local minima (proven mathematically)

### 2. Graph Neural Network Architecture
**Message Passing Pattern**:
1. **Initialization**: Embed node features into hidden representations
2. **Variable → Constraint**: Variables send information about current values
3. **Constraint → Variable**: Constraints send violation/satisfaction signals
4. **Update**: Both node types update representations based on received messages
5. **Repeat**: Multiple rounds of message passing (typically 2-4 iterations)

### 3. Iterative Bound Tightening
**Algorithm Pattern**:
```python
O_d = solve_LP_relaxation(A, b, c)  # Lower bound
O_p = δ * O_d                       # Initial upper bound
while not converged:
    x_new = MILP_network(A, b, c, O_d, O_p)
    if c^T x_new < O_p:
        O_p = c^T x_new             # Tighten upper bound
    else:
        break
```

## Component Relationships

### Training Pipeline Architecture
```
DataLoader → Graph Construction → GNN Forward → Loss Computation → Backprop → Parameter Update
     ↑                                                  ↓
Benchmark Instances                              Gradient Clipping & Scheduling
```

### Integration Points
- **PyTorch Geometric**: Graph data structures and message passing
- **SCIP**: Baseline comparison and potentially ground truth generation
- **WandB**: Experiment tracking and hyperparameter optimization
- **OGB**: Graph property prediction datasets (for benchmarking)

## Critical Implementation Paths

### 1. Graph Construction Pipeline
```python
def build_bipartite_graph(A, b, c):
    # Create constraint and variable nodes
    # Add edges for non-zero A[i,j]
    # Set node features (c, b, bounds)
    return PyG.Data object
```

### 2. Loss Function Implementation
```python
def kkt_loss(x, u, A, b, c):
    dual_feasibility = torch.norm(A.T @ u + c)**2
    primal_feasibility = torch.norm(F.relu(A @ x - b))**2
    nonnegativity = torch.norm(F.relu(-u))**2
    complementarity = torch.norm(u * (A @ x - b))**2
    return dual_feasibility + primal_feasibility + nonnegativity + complementarity
```

### 3. Two-Network Coordination
- **Sequential Training**: Train LP network first, then MILP network
- **Shared Features**: Both networks use same graph representation
- **Bound Propagation**: LP solution provides constraints for MILP network

## Design Principles

### 1. Permutation Invariance
GNN architecture ensures solution is independent of:
- Variable ordering
- Constraint ordering
- Graph node labeling

### 2. Scale Generalization
Graph representation allows training on small instances and testing on larger ones (within hardware limits).

### 3. Mathematical Grounding
Every component connects to optimization theory:
- Loss functions ↔ KKT conditions
- Graph structure ↔ constraint-variable relationships
- Iterative process ↔ branch-and-bound inspiration

### 4. Unsupervised Learning
No dependence on pre-solved instances:
- KKT violations provide training signal
- Infeasible solutions contribute to learning
- Self-improving through experience
