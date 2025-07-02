import os
from datetime import datetime

import configargparse
import torch
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from models.gnn_transformer import GNNTransformer
from models.policy_encoder import GraphDataset, PolicyEncoder, collate

wandb.init(project="kkt_transformer")
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
    group.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    group.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="input batch size for training (default: train batch size)",
    )
    # group.add_argument('--epochs', type=int, default=30,
    #                     help='number of epochs to train (default: 30)')
    group.add_argument(
        "--num_workers", type=int, default=0, help="number of workers (default: 0)"
    )
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

    parser.add_argument(
        "--dataset",
        type=str,
        default="IS",
        help="dataset to use",
    )

    args, _ = parser.parse_known_args()

    DIR_BG = f"../data/dataset/{args.dataset}/BG"

    train_files = os.listdir(f"{DIR_BG}/train")
    valid_files = os.listdir(f"{DIR_BG}/valid")

    train_data = GraphDataset(train_files)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    valid_data = GraphDataset(valid_files)
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and args.devices
        else torch.device("cpu")
    )
    node_encoder = PolicyEncoder(args)

    model = GNNTransformer(
        num_tasks=train_data.len(),
        args=args,
        gnn_node=node_encoder,
    ).to(device)

    print("Model Parameters: ", model.count_parameters())

    wandb.watch(model)


if __name__ == "__main__":
    main()
