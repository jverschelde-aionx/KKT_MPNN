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

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
https://github.com/yandex-research/rtdl-num-embeddings
https://github.com/ucbrise/graphtrans
https://github.com/sribdcn/Predict-and-Search_MILP_method
–python 3.8.13

–pytorch 1.10.2

–cudatoolkit 11.3

–pyscipopt 4.2

–gurobipy 9.5.2

–pyg 2.0.4