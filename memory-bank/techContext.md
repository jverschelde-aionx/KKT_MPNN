# Technical Context: KKT_MPNN

## Technology Stack

### Core Framework
- **PyTorch 1.7.1**: Deep learning framework with CUDA 10.2 support
- **PyTorch Geometric 1.6.3**: Graph neural network library
- **Python 3.8.8**: Base language with conda environment management

### Graph Processing
- **torch-geometric**: Bipartite graph construction and message passing
- **torch-scatter 2.0.6**: Efficient scatter operations for graph aggregation
- **torch-sparse 0.6.9**: Sparse tensor operations for large constraint matrices
- **networkx 2.5**: Graph analysis and visualization

### Optimization & Scientific Computing
- **SCIP**: Baseline solver for comparison (via external integration)
- **NumPy 1.19.2**: Numerical computations
- **SciPy 1.6.1**: Scientific computing utilities
- **scikit-learn 0.24.1**: Additional ML utilities

### Experiment Management
- **Weights & Biases (wandb) 0.10.22**: Experiment tracking and hyperparameter optimization
- **configargparse 1.3**: Configuration management
- **loguru 0.5.3**: Structured logging

### Benchmarking & Data
- **Open Graph Benchmark (OGB) 1.2.6**: Graph property prediction benchmarks
- **pandas 1.2.3**: Data manipulation for results analysis
- **h5py 3.2.1**: Efficient data storage for large instances

## Development Environment

### Containerization
- **Docker**: Environment isolation and reproducibility
- **docker-compose**: Multi-service orchestration
- **CUDA Toolkit 10.2**: GPU acceleration support

### Hardware Requirements
- **GPU**: CUDA-compatible GPU for training acceleration
- **Memory**: Minimum 16GB RAM for medium-scale instances
- **Storage**: SSD recommended for fast data loading

### Environment Setup
```bash
# Conda environment creation
conda env create -f requirements.yml
conda activate graph-aug

# Additional PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
```

## Technical Constraints

### Memory Limitations
- **Graph Size**: Limited by GPU memory (typically 8-16GB)
- **Batch Processing**: May require gradient accumulation for large instances
- **Sparse Matrices**: Use PyTorch sparse tensors for efficiency

### Computational Complexity
- **Message Passing**: O(|E| × hidden_dim) per layer
- **Loss Computation**: O(m×n) for constraint matrix operations
- **Training Time**: Scales with problem size and number of message passing rounds

### Precision Considerations
- **Numerical Stability**: ReLU activations can cause gradient issues
- **Constraint Satisfaction**: Floating-point precision affects feasibility
- **KKT Tolerance**: Define epsilon thresholds for practical convergence

## Data Pipeline

### Input Format
```python
# Expected data structure
{
    'A': torch.sparse.FloatTensor,  # Constraint matrix (m×n)
    'b': torch.FloatTensor,         # RHS vector (m,)
    'c': torch.FloatTensor,         # Objective coefficients (n,)
    'bounds': torch.FloatTensor,    # Variable bounds (n×2)
    'integer_mask': torch.BoolTensor # Which variables are integer (n,)
}
```

### Benchmark Integration
- **NETLIB**: Standard LP benchmark library
- **MIPLIB**: Mixed-integer programming benchmark
- **Custom Generators**: Synthetic problem generation for scalability testing

### Data Loading Strategy
```python
# Efficient batch processing
class OptimizationDataset(torch.utils.data.Dataset):
    def __init__(self, problem_files):
        self.problems = self.load_and_preprocess(problem_files)
    
    def __getitem__(self, idx):
        return self.to_bipartite_graph(self.problems[idx])
```

## Performance Monitoring

### Key Metrics
- **Training Metrics**: Loss convergence, gradient norms, parameter updates
- **Solution Quality**: Optimality gap, feasibility rate, constraint violations
- **Runtime Performance**: Forward pass time, memory usage, convergence speed

### Debugging Tools
- **TensorBoard**: Training visualization and debugging
- **PyTorch Profiler**: Performance bottleneck identification
- **CUDA Memory Profiler**: GPU memory usage analysis

## Integration Patterns

### Model Checkpointing
```python
# State preservation
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_loss': best_loss,
    'hyperparameters': config
}
```

### Configuration Management
```yaml
# config.yaml structure
model:
  hidden_dim: 128
  num_layers: 4
  num_message_passing: 2

training:
  learning_rate: 1e-4
  batch_size: 32
  max_epochs: 100
  
loss:
  kkt_weight: 1.0
  feasibility_weight: 10.0
  integrality_weight: 5.0
```

### External Tool Integration
- **SCIP Python Interface**: For baseline comparison
- **Custom C++ Extensions**: If performance bottlenecks arise
- **Distributed Training**: Multi-GPU scaling for large experiments

## Development Workflow

### Testing Strategy
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline verification
- **Benchmark Tests**: Performance regression detection

### Code Organization
```
src/
├── models/          # GNN architectures
├── losses/          # KKT-based loss functions
├── data/           # Data loading and preprocessing
├── training/       # Training loops and optimization
├── evaluation/     # Metrics and comparison tools
└── utils/          # Helper functions and utilities
```

### Version Control
- **Git**: Source code management
- **Git LFS**: Large file storage for datasets
- **Branch Strategy**: Feature branches with PR reviews
