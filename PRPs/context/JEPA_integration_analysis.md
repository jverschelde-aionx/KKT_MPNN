# JEPA Integration Spec for KKT Nets (Markdown for Coding Agent)

This document contains **precise, file-by-file instructions** to add a **JEPA (Joint Embedding Predictive Architecture) objective in embedding space** to your codebase, while **keeping your existing KKT loss unchanged**. It also adds **one toggleable feature**: switch between an **EMA teacher** (BYOL/I‑JEPA‑style) and a **no‑EMA SimSiam‑style** branch at runtime.

The changes apply to both paths you already support:
- **MLP path**: `KKTNetMLP` with flat input `[vec(A), b, c]`
- **GNN path**: `GNNPolicy` with bipartite graphs

---

## TL;DR (What you’ll get)

- ✅ JEPA pretraining / regularization **in embedding space** (no input-space reconstruction).
- ✅ Works **without labels**; can be used for pretraining or jointly with your existing **KKT residual loss**.
- ✅ **Toggleable** teacher strategy via CLI: `--jepa_mode {ema, simsiam}` (EMA momentum teacher vs. stop‑grad w/o EMA).
- ✅ Keep your current KKT heads and loss **unchanged** (feasibility, stationarity, complementary slackness).

---

## New CLI Flags (edit `train.py` – training arg group)

Add the following args under the **training** group:

```python
t.add_argument("--use_jepa", action="store_true", help="Enable JEPA training")
t.add_argument("--jepa_mode", choices=["ema", "simsiam"], default="ema",
              help="EMA teacher (BYOL/I-JEPA) or SimSiam (no EMA)")
t.add_argument("--jepa_weight", type=float, default=0.2, help="Weight λ for JEPA loss")
t.add_argument("--jepa_pretrain_epochs", type=int, default=0,
              help="JEPA-only epochs before joint KKT+JEPA")
t.add_argument("--jepa_mask_ratio_entries", type=float, default=0.5,
              help="MLP: fraction of entries in [vec(A), b, c] masked in the context view")
t.add_argument("--jepa_mask_ratio_nodes", type=float, default=0.3,
              help="GNN: fraction of variable/constraint nodes masked in the context view")
t.add_argument("--ema_momentum", type=float, default=0.996,
              help="Momentum for EMA teacher (used only when --jepa_mode=ema)")
```

> 🔀 **Toggleable feature**: `--jepa_mode {ema, simsiam}` switches between an **EMA teacher** and a **no‑EMA SimSiam** target. Also add `--use_jepa` to turn JEPA **on/off** entirely.

---

## Model changes (edit `models/models.py`)

### 1) `KKTNetMLP`: add projector/predictor + embedding API

**Inside `__init__`**, right after `self.net` (before heads), insert:

```python
# --- JEPA heads (shared trunk -> projector/predictor) ---
proj_dim = hidden  # or 256
self.jepa_proj = nn.Sequential(
    nn.Linear(hidden, proj_dim), nn.SELU(),
    nn.Linear(proj_dim, proj_dim)
)
self.jepa_pred = nn.Sequential(
    nn.Linear(proj_dim, proj_dim), nn.SELU(),
    nn.Linear(proj_dim, proj_dim)
)
```

**Add helper methods** inside the class:

```python
def encode_trunk(self, flat_input: torch.Tensor) -> torch.Tensor:
    return self.net(flat_input)  # [B, hidden]

def jepa_embed(self, flat_input: torch.Tensor) -> torch.Tensor:
    z = self.encode_trunk(flat_input)
    z = self.jepa_proj(z)
    return torch.nn.functional.normalize(z, dim=-1)  # [B, D]
```

> **Note**: Existing heads (`head_x`, `head_lam`) and forward() are unchanged.

---

### 2) `GNNPolicy`: per‑node projectors/predictors + embedding API

**Inside `__init__`**, near the heads, add:

```python
d = args.embedding_size
# Projectors
self.jepa_proj_v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
self.jepa_proj_c = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
# Predictors (online only)
self.jepa_pred_v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
self.jepa_pred_c = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
```

**Add an embedding method** inside the class:

```python
def jepa_embed_nodes(
    self,
    constraint_features: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_features: torch.Tensor,
    variable_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns L2-normalized node embeddings (constraints, variables) after projectors
    c, v = self.encode(constraint_features, edge_indices, edge_features, variable_features)
    zc = torch.nn.functional.normalize(self.jepa_proj_c(c), dim=-1)
    zv = torch.nn.functional.normalize(self.jepa_proj_v(v), dim=-1)
    return zc, zv
```

> **Note**: `encode()` and heads remain unchanged.

---

## New utilities (place in `train.py` near the top or in a new `models/jepa_utils.py` and import)

```python
from copy import deepcopy

@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, m: float):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

def cosine_pred_loss(p: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    p = torch.nn.functional.normalize(p, dim=-1)
    return (2.0 - 2.0 * (p * z_t).sum(dim=-1)).mean()

def jepa_loss_mlp(online, target_or_online, x_ctx, x_tgt, mode: str):
    # mode: "ema" uses target_or_online as EMA teacher; "simsiam": same net with stop-grad
    z_o = online.jepa_embed(x_ctx)  # [B, D]
    with torch.no_grad() if mode == "ema" else torch.enable_grad():
        z_t = target_or_online.jepa_embed(x_tgt).detach()
    p = online.jepa_pred(z_o)
    return cosine_pred_loss(p, z_t)

def jepa_loss_gnn(online, target_or_online, ctx, tgt, mask_cons, mask_vars, mode: str):
    # ctx/tgt each: (constraint_features, edge_index, edge_attr, variable_features)
    zc_o, zv_o = online.jepa_embed_nodes(*ctx)
    with torch.no_grad() if mode == "ema" else torch.enable_grad():
        zc_t, zv_t = target_or_online.jepa_embed_nodes(*tgt)
        zc_t, zv_t = zc_t.detach(), zv_t.detach()

    pc = online.jepa_pred_c(zc_o[mask_cons])
    pv = online.jepa_pred_v(zv_o[mask_vars])

    loss_c = cosine_pred_loss(pc, zc_t[mask_cons])
    loss_v = cosine_pred_loss(pv, zv_t[mask_vars])
    return 0.5 * (loss_c + loss_v)
```

### View / masking builders

**MLP (flat input)**

```python
def make_mlp_views(flat_input: torch.Tensor, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, M, N = A.shape
    D_in = M * N + M + N
    device = flat_input.device

    mask = torch.zeros((B, D_in), dtype=torch.bool, device=device)
    k = int(mask_ratio * D_in)
    if k < 1:  # safeguard
        return flat_input, flat_input

    for i in range(B):
        idx = torch.randperm(D_in, device=device)[:k]
        mask[i, idx] = True

    noise = torch.zeros_like(flat_input).normal_(0, 0.01)
    x_tgt = flat_input
    x_ctx = flat_input.clone()
    x_ctx[mask] = 0.0
    x_ctx[mask] += noise[mask]  # small corruption on masked positions

    return x_ctx, x_tgt
```

**GNN (node masking)**

```python
def make_gnn_views(batch_graph, mask_ratio_nodes: float):
    n_c = batch_graph.constraint_features.size(0)
    n_v = batch_graph.variable_features.size(0)
    device = batch_graph.constraint_features.device

    m_cons = torch.zeros(n_c, dtype=torch.bool, device=device)
    m_vars = torch.zeros(n_v, dtype=torch.bool, device=device)

    num_mc = max(1, int(mask_ratio_nodes * n_c))
    num_mv = max(1, int(mask_ratio_nodes * n_v))
    m_cons[torch.randperm(n_c, device=device)[:num_mc]] = True
    m_vars[torch.randperm(n_v, device=device)[:num_mv]] = True

    # context = masked copy; target = original
    ctx_c = batch_graph.constraint_features.clone()
    ctx_v = batch_graph.variable_features.clone()
    ctx_e = batch_graph.edge_attr

    ctx_c[m_cons] = 0.0
    ctx_v[m_vars] = 0.0

    context_tuple = (ctx_c, batch_graph.edge_index, ctx_e, ctx_v)
    target_tuple  = (batch_graph.constraint_features, batch_graph.edge_index,
                     batch_graph.edge_attr, batch_graph.variable_features)
    return context_tuple, target_tuple, m_cons, m_vars
```

---

## Training loop changes (edit `train.py`)

### 1) Build an optional EMA target after the model is created

Right after the model is constructed:

```python
target_model = None
if args.use_jepa and args.jepa_mode == "ema":
    target_model = deepcopy(model)
    for p in target_model.parameters():
        p.requires_grad_(False)
```

### 2) **Checkpointing**: save and restore EMA target if present

When creating `ckpt` before saving:

```python
ckpt = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "args": vars(args),
}
if target_model is not None:
    ckpt["target_model"] = target_model.state_dict()
```

When resuming (if you support resuming):

```python
if args.use_jepa and args.jepa_mode == "ema" and "target_model" in ckpt:
    target_model.load_state_dict(ckpt["target_model"])
```

### 3) Update `train_epoch` signature and call site

**Change the function definition** to accept `args` and `target_model`:

```python
def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    training_state: TrainingState,
    primal_weight: float,
    dual_weight: float,
    stationarity_weight: float,
    complementary_slackness_weight: float,
    args=None,
    target_model=None,
) -> float:
    ...
```

**Update the call** inside `train(...)`:

```python
train_loss = train_epoch(
    model=model,
    loader=train_loader,
    optimizer=optimizer,
    device=device,
    training_state=training_state,
    primal_weight=args.primal_weight,
    dual_weight=args.dual_weight,
    stationarity_weight=args.stationarity_weight,
    complementary_slackness_weight=args.complementary_slackness_weight,
    args=args,
    target_model=target_model,
)
```

### 4) Inject JEPA computation inside `train_epoch(...)`

Inside the training loop, **after** building `y_pred` but **before** backward:

```python
# existing KKT loss
loss_kkt, _ = kkt_loss(
    y_pred=y_pred,
    A=A, b=b, c=c,
    mask_m=mask_m, mask_n=mask_n,
    primal_weight=primal_weight,
    dual_weight=dual_weight,
    stationarity_weight=stationarity_weight,
    complementary_slackness_weight=complementary_slackness_weight,
)

loss = loss_kkt

if args and args.use_jepa:
    if isinstance(batch[0], torch_geometric.data.Batch):
        # GNN path
        batch_graph, A, b, c, mask_m, mask_n, m_sizes, n_sizes = batch
        ctx, tgt, m_cons, m_vars = make_gnn_views(batch_graph, args.jepa_mask_ratio_nodes)
        online_ref  = model
        target_ref  = target_model if args.jepa_mode == "ema" else model
        loss_jepa = jepa_loss_gnn(online_ref, target_ref, ctx, tgt, m_cons, m_vars, args.jepa_mode)
    else:
        # MLP path
        model_input, A, b, c, mask_m, mask_n = batch
        x_ctx, x_tgt = make_mlp_views(model_input, A, b, c, args.jepa_mask_ratio_entries)
        online_ref  = model
        target_ref  = target_model if args.jepa_mode == "ema" else model
        loss_jepa = jepa_loss_mlp(online_ref, target_ref, x_ctx, x_tgt, args.jepa_mode)

    # Optional JEPA-only warm-up
    jepa_only = training_state.get_epoch() < (args.jepa_pretrain_epochs or 0)
    loss = (loss_jepa if jepa_only else (loss_kkt + args.jepa_weight * loss_jepa))

    if (training_state.get_step() % training_state.log_every) == 0:
        wandb.log({"train/loss_jepa": float(loss_jepa)}, step=training_state.get_step())
```

**After `optimizer.step()`**, update EMA if enabled:

```python
if args and args.use_jepa and args.jepa_mode == "ema" and (target_model is not None):
    ema_update(target_model, model, m=args.ema_momentum)
```

> **Eval loop**: leave as-is (KKT metrics only). You may also log JEPA loss on val for debugging if desired.

---

## Nothing else changes

- **`kkt_loss(...)`** remains unchanged.
- Heads/forward of both models remain unchanged for the task outputs.
- Data loaders and padding remain unchanged.

---

## Example runs

**MLP + JEPA (EMA), joint training:**
```bash
python train.py --devices 0 --batch_size 8 --epochs 30 \
  --use_jepa --jepa_mode ema --jepa_pretrain_epochs 0 \
  --jepa_weight 0.2 --jepa_mask_ratio_entries 0.5 --ema_momentum 0.996
```

**GNN + 3‑epoch JEPA pretrain (no EMA, SimSiam):**
```bash
python train.py --devices 0 --batch_size 4 --epochs 30 --use_bipartite_graphs \
  --use_jepa --jepa_mode simsiam --jepa_pretrain_epochs 3 \
  --jepa_weight 0.2 --jepa_mask_ratio_nodes 0.3
```

---

## Acceptance checklist (for CI / the coding agent)

- [ ] Code compiles with both `--use_bipartite_graphs` on/off and `--use_jepa` on/off.
- [ ] Running a single train step with `--use_jepa --jepa_mode simsiam` does **not** create `target_model`.
- [ ] Running with `--use_jepa --jepa_mode ema` **does** create and update `target_model` (EMA) each step.
- [ ] JEPA-only warm‑up activates when `epoch < --jepa_pretrain_epochs` (loss = JEPA only).
- [ ] `wandb` logs include `train/loss_jepa` at the specified cadence.
- [ ] Checkpoint contains `target_model` state when `--jepa_mode=ema`, and load restores it.
- [ ] Training without JEPA reproduces previous behavior and metrics (loss = KKT only).

---

## Recommended defaults and notes

- Start with `--jepa_weight 0.2`, `--ema_momentum 0.996`, mask ratios `0.5` (MLP) / `0.3` (GNN).
- L2‑normalize embeddings before cosine; predictor only on the online branch.
- Simpler is often better: start with zero‑masking (optionally add small Gaussian noise).

---

## Appendix: minimal diff hints (search/insert anchors)

- In **`models/models.py`**, search for `class KKTNetMLP` → after `self.net = ...` insert **JEPA heads**; add `encode_trunk` and `jepa_embed` methods.
- In **`models/models.py`**, search for `class GNNPolicy` → near heads add **JEPA projectors/predictors**; add `jepa_embed_nodes` method.
- In **`train.py`**, add the new **CLI flags** and **imports** (`deepcopy`).
- In **`train.py`**, define **EMA**, **JEPA losses**, and **view builders** near the top.
- In **`train.py`**, after model creation, create `target_model` when required.
- In **`train.py`**, update `train_epoch(...)` signature and call; integrate JEPA loss + optional warmup; update EMA after optimizer step.
- In **`train.py`**, extend checkpointing to include `target_model` when using EMA.

---

**End of file.**
