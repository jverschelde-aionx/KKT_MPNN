import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch_geometric
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings

_CONS_PAD: int = 4  # global max constraint feature width
_VAR_PAD: int = 26  # global max variable   feature width


class GNNPolicy(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        BipartiteGraphConvolution.add_args(parser)
        group = parser.add_argument_group("GNNPolicy - MPNN Config")
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
        group.add_argument(
            "--num_emb_type", choices=["periodic", "pwl", "linear"], default="periodic"
        )
        group.add_argument("--num_emb_bins", type=int, default=32)  # for PWL
        group.add_argument("--num_emb_freqs", type=int, default=16)  # for periodic

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += f"-embedding_size={args.embedding_size}"
        name += f"-cons_nfeats={args.cons_nfeats}"
        name += f"-edge_nfeats={args.edge_nfeats}"
        name += f"-var_nfeats={args.var_nfeats}"
        name += f"-num_emb_type={args.num_emb_type}"
        if args.num_emb_type == "pwl":
            name += f"-num_emb_bins={args.num_emb_bins}"
        elif args.num_emb_type == "periodic":
            name += f"-num_emb_freqs={args.num_emb_freqs}"
        return name

    def __init__(self, args):
        super().__init__()
        d_out = args.embedding_size
        # Constraint embedding
        self.cons_num_emb, cons_out_dim = self._build_numeric_block(
            in_feats=args.cons_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.cons_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_out_dim),
            torch.nn.Linear(cons_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Variable embedding
        self.var_num_emb, var_out_dim = self._build_numeric_block(
            in_feats=args.var_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.var_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(var_out_dim),
            torch.nn.Linear(var_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_embedding = torch.nn.LayerNorm(args.edge_nfeats)

        # Message‑passing layers
        self.conv_v_to_c = BipartiteGraphConvolution(args)
        self.conv_c_to_v = BipartiteGraphConvolution(args)
        self.conv_v_to_c2 = BipartiteGraphConvolution(args)
        self.conv_c_to_v2 = BipartiteGraphConvolution(args)

        # Output module
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, 1, bias=False),
        )

    def forward(
        self,
        constraint_features,  # (n_c, cons_nfeats)
        edge_indices,  # (2, |E|)
        edge_features,  # (|E|, edge_nfeats)
        variable_features,
    ):  # (n_v, var_nfeats)
        rev_idx = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # numeric embeddings + small MLP
        c = self.cons_proj(self.cons_num_emb(constraint_features))

        v = self.var_proj(self.var_num_emb(variable_features))
        e = self.edge_embedding(edge_features)

        # two rounds of bipartite convolution
        c = self.conv_v_to_c(v, rev_idx, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)
        c = self.conv_v_to_c2(v, rev_idx, e, c)
        v = self.conv_c_to_v2(c, edge_indices, e, v)

        # optional scalar head
        return self.output_module(v).squeeze(-1)  # (n_v,)

    @staticmethod
    def _build_numeric_block(
        in_feats: int,
        emb_type: str,
        n_freq: int,
        n_bins: int,
    ) -> Tuple[torch.nn.Module, int]:
        # Build numeric embedding and return (module, output_dim).

        if emb_type == "periodic":
            emb_layer = PeriodicEmbeddings(
                n_features=in_feats,
                n_frequencies=n_freq,
                lite=True,  # keep your current setting
            )
        elif emb_type == "pwl":
            emb_layer = PiecewiseLinearEmbeddings(
                n_features=in_feats,
                num_bins=n_bins,
            )
        else:  # "linear"
            return torch.nn.Identity(), in_feats

        # Determine the true output width by a forward pass on a dummy tensor.
        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(1, in_feats)
            out_dim: int = emb_layer(dummy).numel()

        return torch.nn.Sequential(emb_layer, torch.nn.Flatten(start_dim=1)), out_dim


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    @staticmethod
    def add_args(parser):
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

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
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
        # Handle single BG file path (no solution file needed for KKT training)
        BGFilepath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)

        BG = bgData
        A, v_map, v_nodes, c_nodes, b_vars = BG

        # Extract variable names from the BG data structure
        varNames = list(v_map.keys())

        n_vars = len(v_nodes)
        sols = np.zeros((1, n_vars))  # Single dummy solution
        objs = np.array([0.0])  # Single dummy objective

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

        constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)
        variable_features = torch.as_tensor(variable_features, dtype=torch.float32)
        edge_features = torch.as_tensor(edge_features, dtype=torch.float32)

        # Replace NaN / ±Inf with finite values
        constraint_features = torch.nan_to_num(
            constraint_features, nan=0.0, posinf=1e6, neginf=-1e6
        )
        variable_features = torch.nan_to_num(
            variable_features, nan=0.0, posinf=1e6, neginf=-1e6
        )
        edge_features = torch.nan_to_num(
            edge_features, nan=0.0, posinf=1e6, neginf=-1e6
        )

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
        graph.sample_idx = index
        graph.sample_path = self.sample_files[index]

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
    Node‑bipartite graph for MILP problems.
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    Must support zero‑arg construction so that PyG's Batch can create
    a template object internally.
    """

    def __init__(
        self,
        constraint_features: Optional[torch.Tensor] = None,
        edge_indices: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        variable_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # allow empty init (template object)
        if constraint_features is None:
            return

        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

        m = constraint_features.size(0)
        n = variable_features.size(0)

        self.is_var_node = torch.cat(
            [torch.zeros(m, dtype=torch.bool), torch.ones(n, dtype=torch.bool)]
        )

        self.is_constr_node = ~self.is_var_node

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
        c = self.base.cons_proj(self.base.cons_num_emb(c))
        e = self.base.edge_embedding(e_attr)
        v = self.base.var_proj(self.base.var_num_emb(v))

        c = self.base.conv_v_to_c(v, rev_idx, e, c)
        v = self.base.conv_c_to_v(c, e_idx, e, v)
        c = self.base.conv_v_to_c2(v, rev_idx, e, c)
        v = self.base.conv_c_to_v2(c, e_idx, e, v)

        h = torch.cat([c, v], dim=0)  # [n_c+n_v, 64]
        if perturb is not None:
            h = h + perturb
        return h


def _right_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    # Pads the last dim of `x` with zeros on the right up to `target`.
    if x.size(1) == target:
        return x
    pad_width: int = target - x.size(1)
    pad: torch.Tensor = torch.zeros(
        x.size(0), pad_width, dtype=x.dtype, device=x.device
    )
    return torch.cat([x, pad], dim=1)


def collate(
    batch: List["BipartiteNodeData"],
) -> Tuple[
    torch_geometric.data.Batch,  # batch_graph
    List[torch.Tensor],  # A_list (sparse)
    torch.Tensor,
    torch.Tensor,  # b_pad , c_pad
    torch.Tensor,
    torch.Tensor,  # b_mask, c_mask
    List[int],
    List[int],  # m_sizes, n_sizes
]:
    graphs: List["BipartiteNodeData"] = []
    A_list: List[torch.Tensor] = []
    b_list: List[torch.Tensor] = []
    c_list: List[torch.Tensor] = []
    m_sizes: List[int] = []
    n_sizes: List[int] = []

    for data in batch:
        assert torch.isfinite(data.variable_features).all(), (
            "NaN/Inf in variable_features"
        )
        assert torch.isfinite(data.edge_attr).all(), "NaN/Inf in edge values"

        # ----- enforce fixed feature width ---------------------------
        data.constraint_features = _right_pad(data.constraint_features, _CONS_PAD)
        data.variable_features = _right_pad(data.variable_features, _VAR_PAD)

        graphs.append(data)

        m: int = data.constraint_features.size(0)
        n: int = data.variable_features.size(0)
        m_sizes.append(m)
        n_sizes.append(n)

        rows, cols = data.edge_index
        A_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=data.edge_attr.squeeze(-1),
            size=(m, n),
        ).coalesce()
        A_list.append(A_sparse)

        b_list.append(data.constraint_features[:, 0])  # any column is fine
        c_list.append(data.variable_features[:, 0])

    B: int = len(batch)
    max_m: int = max(m_sizes)
    max_n: int = max(n_sizes)

    b_pad: torch.Tensor = torch.zeros(B, max_m)
    b_mask: torch.Tensor = torch.zeros(B, max_m, dtype=torch.bool)
    for i, (b, m) in enumerate(zip(b_list, m_sizes)):
        b_pad[i, :m] = b
        b_mask[i, :m] = True

    c_pad: torch.Tensor = torch.zeros(B, max_n)
    c_mask: torch.Tensor = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (c, n) in enumerate(zip(c_list, n_sizes)):
        c_pad[i, :n] = c
        c_mask[i, :n] = True

    batch_graph: torch_geometric.data.Batch = torch_geometric.data.Batch.from_data_list(
        graphs
    )
    sources = [data.sample_path for data in batch]

    return (
        batch_graph,
        A_list,
        b_pad,
        c_pad,
        b_mask,
        c_mask,
        m_sizes,
        n_sizes,
        sources,
    )
