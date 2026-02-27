import math
from typing import Dict, List

import torch
from configargparse import Namespace
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    OneCycleLR,
    _LRScheduler,
)


def make_scheduler(optimizer: Optimizer, args, steps_per_epoch: int) -> _LRScheduler:
    total_steps = int(args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_pct * total_steps)
    min_ratio = float(args.min_lr_ratio)
    if args.scheduler == "none":
        return None
    elif args.scheduler == "onecycle":
        # OneCycle wants a max_lr; use base lr as max_lr for simplicity
        return OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=args.warmup_pct,
            anneal_strategy="cos",
        )
    elif args.scheduler in ["cosine", "cosine_warmup"]:

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_param_groups(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, object]]:
    """Apply weight decay to weights only; exclude biases and norm parameters."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    norm_types = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.GroupNorm,
    )
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name.endswith("bias") or isinstance(module, norm_types):
                no_decay.append(param)
            else:
                decay.append(param)
    groups: List[Dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def make_optimizer(model: torch.nn.Module, args: Namespace) -> Optimizer:
    wd = float(args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    elif args.optimizer == "adamw":
        pg = build_param_groups(model, weight_decay=wd)
        return torch.optim.AdamW(pg, lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
