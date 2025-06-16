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
