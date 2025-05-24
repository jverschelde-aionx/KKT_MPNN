# Product Context: KKT_MPNN

## Why This Project Exists

### Problem Statement
Traditional optimization solvers treat each problem instance independently, solving from scratch without leveraging patterns from previous solutions. This is inefficient when:
- Similar problem instances recur regularly (e.g., daily logistics planning)
- Real-time decisions are required
- Problem structure contains learnable patterns

### Current Limitations
**ML-Augmented Solvers**: Enhance existing algorithms but still require conventional solvers for final solutions.
**End-to-End Methods**: Often struggle with:
- Feasibility guarantees (solutions may violate constraints)
- Need for extensive labeled training data (expensive optimal solutions)
- Fixed input sizes (can't generalize to different problem dimensions)

## What This Solves

### Core Innovation
Transform optimization from an algorithmic process to a learning process by:
1. **Unsupervised Learning**: Use KKT condition violations as loss signal - no need for pre-solved examples
2. **Graph Representation**: Handle variable-sized problems through bipartite graphs (constraints ↔ variables)
3. **Theoretical Foundation**: Ground learning in mathematical optimization theory rather than ad-hoc penalties

### Target Users
- **Researchers**: Investigating neural approaches to combinatorial optimization
- **Industry**: Companies with repetitive optimization workflows (logistics, scheduling, resource allocation)
- **ML Community**: Demonstrating how domain knowledge (KKT conditions) can guide neural architectures

## How It Should Work

### User Experience Flow
1. **Input**: Provide LP/MILP instance (constraint matrix A, bounds b, objective c)
2. **Processing**: Neural network predicts optimal solution using learned patterns
3. **Output**: Feasible solution competitive with traditional solvers
4. **Learning**: Network improves from each instance without requiring ground-truth labels

### Expected Benefits
- **Speed**: Faster than iterative algorithms for similar problem classes
- **Integration**: Easy incorporation into ML pipelines (just another neural network)
- **Adaptability**: Continuous improvement on recurring problem types
- **Feasibility**: Mathematical guarantees through KKT-based training

## Success Metrics
- **Accuracy**: <5% optimality gap vs SCIP on benchmark instances
- **Feasibility**: >95% of solutions respect all constraints
- **Speed**: 2-10x faster inference than traditional solvers
- **Generalization**: Effective on problem sizes not seen during training
