# Active Context: KKT_MPNN

## Current Work Focus

### Memory Bank Initialization (Current)
Just completed initializing the memory bank structure to establish project knowledge base for future development sessions. This ensures continuity and context preservation across work sessions.

### Project Status
- **Research Phase**: Complete (detailed proposal in papers/proposal.latex)
- **Implementation Phase**: Beginning - existing GNN infrastructure present but KKT-specific components need development
- **Current Priority**: RQ1 - Implementing KKT-based loss functions for LP solving

## Recent Changes & Observations

### Existing Infrastructure Analysis
- Found comprehensive main.py with graph neural network training pipeline
- Uses PyTorch Geometric with sophisticated data loading (OGB datasets)
- Includes Weights & Biases integration for experiment tracking
- Docker containerization already in place
- Missing: KKT-specific loss functions and bipartite graph construction for optimization problems

### Key Code Components Identified
```python
# In src/main.py - existing GNN training loop structure
def train(model, device, train_loader, optimizer, args, calc_loss, scheduler):
    # Could be adapted for KKT-based training
    
def eval(model, device, loader, evaluator):
    # Evaluation framework exists, needs optimization-specific metrics
```

## Next Steps & Immediate Actions

### Phase 1: LP Implementation (Weeks 1-6)
1. **Create KKT Loss Module** (Next)
   - Implement `kkt_loss(x, u, A, b, c)` function
   - Add mathematical verification tests
   - Ensure no local minima property holds

2. **Bipartite Graph Construction**
   - Build `optimization_to_graph()` converter
   - Handle sparse constraint matrices efficiently
   - Create node features for variables and constraints

3. **Adapt Training Pipeline**
   - Modify existing main.py to use KKT loss
   - Update data loaders for LP instances
   - Integrate NETLIB benchmark loading

### Phase 2: MILP Extension (Weeks 7-12)
1. **Second Network Architecture**
   - Design integer-constraint handling network
   - Implement bound tightening algorithm
   - Create iterative training procedure

2. **MILP Loss Implementation**
   - Add integrality penalties: `x_i^2(1-x_i)^2`
   - Include bound constraints in loss
   - Test convergence properties

## Active Decisions & Considerations

### Architecture Choices
- **Reusing Existing GNN Layers**: Leverage current message passing infrastructure
- **Loss Function Design**: Stick to mathematical KKT formulation vs. adding heuristic terms
- **Training Strategy**: End-to-end vs. progressive (LP first, then MILP)

### Performance Trade-offs
- **Memory Usage**: Large constraint matrices may require sparse operations
- **Convergence Speed**: Balance between mathematical rigor and practical training time
- **Generalization**: Training set diversity vs. computational resources

### Implementation Priorities
1. **Mathematical Correctness**: Ensure KKT conditions are properly encoded
2. **Proof of Concept**: Get working LP solver on small instances first
3. **Scalability**: Design for larger problems from the start
4. **Benchmarking**: Establish comparison methodology with SCIP

## Current Insights & Patterns

### Key Learning: Existing Infrastructure Alignment
The current main.py shows sophisticated GNN training patterns that align well with our needs:
- Model factory pattern with configurable architectures
- Comprehensive evaluation framework
- Hardware acceleration support
- Experiment tracking integration

### Technical Insights
- **Graph Construction**: Need to create bipartite graphs from optimization matrices
- **Loss Integration**: Can adapt existing loss calculation patterns for KKT objectives
- **Evaluation Metrics**: Must define optimization-specific success criteria (feasibility, optimality gap)

### Research Insights from Proposal Analysis
- **Two-Network Approach**: Clear separation between LP relaxation and integer feasibility
- **Unsupervised Learning**: KKT violations provide natural training signal
- **Theoretical Foundation**: Mathematical guarantees distinguish this from ad-hoc neural approaches

## Immediate Development Environment

### Files to Modify First
1. `src/losses/kkt_loss.py` (new) - Core KKT loss implementation
2. `src/data/optimization_dataset.py` (new) - LP/MILP data loading
3. `src/models/bipartite_gnn.py` (new) - Optimization-specific GNN architecture
4. Adapt `src/main.py` for optimization training

### Testing Priorities
1. KKT loss mathematical properties (no local minima)
2. Small LP instances (2-3 variables, 2-3 constraints)
3. Graph construction correctness
4. Training convergence on toy problems

## Questions & Investigations Needed

### Technical Questions
- How to handle numerical precision in KKT condition checking?
- What message passing rounds are optimal for constraint-variable interaction?
- How to balance different terms in the KKT loss function?

### Experimental Questions
- Which NETLIB instances are best for initial testing?
- How does training time scale with problem size?
- What hyperparameters transfer from standard GNN training?

### Integration Questions
- How to incorporate SCIP for baseline comparison?
- What data preprocessing is needed for real-world instances?
- How to handle degenerate or infeasible problems during training?
