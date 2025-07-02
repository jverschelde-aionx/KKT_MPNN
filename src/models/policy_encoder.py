import pickle

import numpy as np
import torch
import torch_geometric


class GNNPolicy(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        BipartiteGraphConvolution.add_args(parser)
        group = parser.add_argument_group("GNNPolicy - MPNN Config")
        group.add_argument(
            "--embedding_size", default=64, type=int, help="Size of the node embeddings"
        )
        group.add_argument(
            "--cons_nfeats",
            default=4,
            type=int,
            help="Number of features for constraints",
        )
        group.add_argument(
            "--edge_nfeats", default=1, type=int, help="Number of features for edges"
        )
        group.add_argument(
            "--var_nfeats", default=6, type=int, help="Number of features for variables"
        )

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += f"-embedding_size={args.embedding_size}"
        name += f"-cons_nfeats={args.cons_nfeats}"
        name += f"-edge_nfeats={args.edge_nfeats}"
        name += f"-var_nfeats={args.var_nfeats}"
        return name

    def __init__(self, args):
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(args.cons_nfeats),
            torch.nn.Linear(args.cons_nfeats, args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(args.edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(args.var_nfeats),
            torch.nn.Linear(args.var_nfeats, args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    @staticmethod
    def add_args(parser):
        BipartiteGraphConvolution.add_args(parser)
        group = parser.add_argument_group("BipartiteGraphConvolution - MPNN Config")
        # The embedding size is the size of the node embeddings
        group.add_argument(
            "--embedding_size", default=64, type=int, help="Size of the node embeddings"
        )

    @staticmethod
    def name(args):
        name = "bipartite_graph_convolution"
        name += f"-embedding_size={args.embedding_size}"
        return name

    def __init__(self, args):
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, args.embedding_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(args.embedding_size, args.embedding_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(args.embedding_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * args.embedding_size, args.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embedding_size, args.embedding_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        b = torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a = self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData["var_names"]

        sols = solData["sols"][:50]  # [0:300]
        objs = solData["objs"][:50]  # [0:300]

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars = BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        constraint_features[np.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map)

        graph.varInds = [[varname_map], [b_vars]]

        return graph


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


# custom class that wraps the GNNPolicy to be used as an encoder that can be used in a GraphTrans model.
class PolicyEncoder(torch.nn.Module):
    """
    Wrapper that turns GNNPolicy into a pure encoder.
    `out_dim` is used later by the transformer.
    """

    def __init__(self, args):
        super().__init__()
        self.base = GNNPolicy(args)
        self.out_dim = args.embedding_size  # 64 by default

    def forward(self, data, perturb=None):
        c = data.constraint_features  # [n_c, f_c]
        v = data.variable_features  # [n_v, f_v]
        e_idx = data.edge_index  # shape [2, |E|]
        e_attr = data.edge_attr  # [|E|, f_e]

        # identical to first part of original GNNPolicy.forward
        rev_idx = torch.stack([e_idx[1], e_idx[0]], dim=0)
        c = self.base.cons_embedding(c)
        e = self.base.edge_embedding(e_attr)
        v = self.base.var_embedding(v)

        c = self.base.conv_v_to_c(v, rev_idx, e, c)
        v = self.base.conv_c_to_v(c, e_idx, e, v)
        c = self.base.conv_v_to_c2(v, rev_idx, e, c)
        v = self.base.conv_c_to_v2(c, e_idx, e, v)

        h = torch.cat([c, v], dim=0)  # [n_c+n_v, 64]
        if perturb is not None:
            h = h + perturb
        return h


# This collate function is used to create batches of BipartiteNodeData objects.
def collate(batch):
    """
    A hand‑written collate_fn that turns a list of
    BipartiteNodeData objects into one batch *and*
    returns A, b, c with fixed sizes (m, n).
    """
    graphs, A_list, b_list, c_list = [], [], [], []

    for data in batch:
        graphs.append(data)  # still a pyg Data object

        # Suppose `data.edge_index` holds (row, col) of A
        #         `data.edge_attr`  holds the coefficient value
        m, n = data.varInds[1][0].size(0), data.ntvars
        A = torch.zeros(m, n, device=data.edge_index.device)

        rows, cols = data.edge_index
        A[rows, cols - m] = data.edge_attr.squeeze(-1)  # convert to dense
        A_list.append(A)

        b_list.append(data.constraint_features[:, 0])  # adapt to your layout
        c_list.append(data.variable_features[:, 0])  # adapt to your layout

    batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
    A_batch = torch.stack(A_list)  # (B, m, n)
    b_batch = torch.stack(b_list)  # (B, m)
    c_batch = torch.stack(c_list)  # (B, n)

    return batch_graph, A_batch, b_batch, c_batch
