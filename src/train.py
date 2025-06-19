from datetime import datetime

import configargparse
from torch_geometric.loader import DataLoader

from models.GCN import GNNPolicy, GraphDataset
from models.gnn_transformer import GNNTransformer

now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")


def main():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="KKT transformer baseline on ecole-IS data with Pytorch Geometrics",
    )
    parser.add_argument("--configs", required=False, is_config_file=True)

    # parser.add_argument('--aug', type=str, default='baseline',
    #                     help='augment method to use [baseline|flag|augment]')

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="maximum sequence length to predict (default: None)",
    )

    group = parser.add_argument_group("model")
    group.add_argument(
        "--model_type", type=str, default="gnn", help="gnn|pna|gnn-transformer"
    )
    group.add_argument("--graph_pooling", type=str, default="mean")
    group = parser.add_argument_group("gnn")
    group.add_argument("--gnn_type", type=str, default="gcn")
    group.add_argument("--gnn_virtual_node", action="store_true")
    group.add_argument("--gnn_dropout", type=float, default=0)
    group.add_argument(
        "--gnn_num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    group.add_argument(
        "--gnn_emb_dim",
        type=int,
        default=300,
        help="dimensionality of hidden units in GNNs (default: 300)",
    )
    group.add_argument("--gnn_JK", type=str, default="last")
    group.add_argument("--gnn_residual", action="store_true", default=False)

    # group = parser.add_argument_group('training')
    # group.add_argument('--devices', type=str, default="0",
    #                     help='which gpu to use if any (default: 0)')
    # group.add_argument('--batch_size', type=int, default=128,
    #                     help='input batch size for training (default: 128)')
    # group.add_argument('--eval_batch_size', type=int, default=None,
    #                     help='input batch size for training (default: train batch size)')
    # group.add_argument('--epochs', type=int, default=30,
    #                     help='number of epochs to train (default: 30)')
    # group.add_argument('--num_workers', type=int, default=0,
    #                     help='number of workers (default: 0)')
    # group.add_argument('--scheduler', type=str, default=None)
    # group.add_argument('--pct_start', type=float, default=0.3)
    # group.add_argument('--weight_decay', type=float, default=0.0)
    # group.add_argument('--grad_clip', type=float, default=None)
    # group.add_argument('--lr', type=float, default=0.001)
    # group.add_argument('--max_lr', type=float, default=0.001)
    # group.add_argument('--runs', type=int, default=10)
    # group.add_argument('--test-freq', type=int, default=1)
    # group.add_argument('--start-eval', type=int, default=15)
    # group.add_argument('--resume', type=str, default=None)
    # group.add_argument('--seed', type=int, default=None)

    args, _ = parser.parse_known_args()

    train_data = GraphDataset(train_files)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_data = GraphDataset(valid_files)
    valid_loader = DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    node_encoder = GNNPolicy(args)

    model = GNNTransformer(
        num_tasks=num_tasks,
        args=args,
        node_encoder=node_encoder,
        edge_encoder_cls=edge_encoder_cls,
    ).to(device)


if __name__ == "__main__":
    main()
