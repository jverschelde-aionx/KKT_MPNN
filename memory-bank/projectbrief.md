# Project Brief: KKT_MPNN

## Core Purpose
Develop an end-to-end neural-based framework for solving linear programs (LPs) and mixed-integer linear programs (MILPs) using Karush-Kuhn-Tucker (KKT) conditions integrated with Graph Neural Networks (GNNs).

## Primary Research Questions

### RQ1: KKT-Conditions via GNN Architecture
Can we model the KKT-conditions of a linear program (LP) as a minimization problem using a graph neural network architecture?

**Approach**: Encode KKT conditions as a differentiable loss function that vanishes only at optimal solutions, using ReLU activations to handle inequalities.

### RQ2: Extension to MILPs
Can we reuse the architecture from RQ1 for mixed-integer linear programs? What modifications are required?

**Approach**: Two-network pipeline - first network solves LP relaxation for lower bound, second network finds feasible integer solutions with iterative bound tightening.

### RQ3: Learning from Infeasible Solutions
Can infeasible solutions be reused to guide the search towards feasible solutions?

**Approach**: Prioritized replay buffer mechanism to learn from constraint violations and improve convergence.

## Key Innovation
Unlike traditional ML-augmented solvers that assist existing algorithms, this creates a pure neural replacement that:
- Requires no ground-truth solution labels (unsupervised learning via KKT violations)
- Handles variable-sized problems through graph representation
- Respects mathematical optimization theory through KKT conditions

## Success Criteria
- Competitive accuracy and runtime vs classical solvers (SCIP)
- Generalization to unseen problem sizes
- Feasible solutions that respect all constraints
- Demonstrable improvement from infeasible solution reuse

## Timeline
6-month research project (April - September) with incremental milestones from LP-only to full MILP implementation.
