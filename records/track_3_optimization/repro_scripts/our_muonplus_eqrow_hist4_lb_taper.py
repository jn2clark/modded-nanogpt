"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import math
import uuid
import time
from pathlib import Path
import argparse

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
SEED = args.seed


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def _apply_muon_eq(G: Tensor, axis: str) -> Tensor:
    if axis == "row":
        target = G.float().norm() / (G.size(-2) ** 0.5)
        row_norm = G.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return G * (target / row_norm).to(G.dtype)
    if axis == "col":
        target = G.float().norm() / (G.size(-1) ** 0.5)
        col_norm = G.float().norm(dim=-2, keepdim=True).clamp_min(1e-6)
        return G * (target / col_norm).to(G.dtype)
    return G

def _bounded_logit(value: float, max_value: float) -> float:
    frac = min(1.0 - 1e-6, max(1e-6, float(value) / max(float(max_value), 1e-12)))
    return math.log(frac / (1.0 - frac))

def _scheduled_history_blend(base_blend: float, step: int, group: dict) -> float:
    blend = base_blend
    blend_end = float(group.get("muon_history_blend_end", -1.0))
    ramp_start = int(group.get("muon_history_blend_ramp_start_step", -1))
    ramp_end = int(group.get("muon_history_blend_ramp_end_step", -1))
    if blend_end >= 0.0 and ramp_start >= 0 and ramp_end > ramp_start:
        progress = min(1.0, max(0.0, (step - ramp_start) / (ramp_end - ramp_start)))
        blend = base_blend * (1.0 - progress) + blend_end * progress
    decay_start = int(group.get("muon_history_decay_start_step", -1))
    decay_end = int(group.get("muon_history_decay_end_step", -1))
    final_mult = float(group.get("muon_history_final_blend_mult", 1.0))
    if decay_start >= 0 and decay_end > decay_start and step >= decay_start:
        progress = min(1.0, max(0.0, (step - decay_start) / (decay_end - decay_start)))
        blend *= (1.0 - progress) + progress * final_mult
    return blend

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True, muon_plus=False, muon_eq_axis="none"):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = _apply_muon_eq(update, muon_eq_axis)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    if muon_plus:
        target_norm = min(update.size(-2), update.size(-1)) ** 0.5
        update = update * (target_norm / update.float().norm().clamp_min(1e-6)).to(update.dtype)
    return update

def apply_muon_history(update: Tensor, state: dict, step: int, group: dict) -> Tensor:
    history_steps = int(group.get("muon_history_steps", 0) or 0)
    if history_steps <= 0:
        return update
    base_blend = float(group.get("muon_history_blend", 0.0) or 0.0)
    learn_blend = bool(group.get("muon_history_learn_blend", False))
    if base_blend <= 0.0 and not learn_blend:
        return update

    if "muon_history_buffer" not in state:
        state["muon_history_buffer"] = torch.zeros((history_steps, *update.shape), dtype=update.dtype, device=update.device)
    history = state["muon_history_buffer"]
    ready = step > history_steps
    scheduled_blend = _scheduled_history_blend(base_blend, step, group)
    current_g = update

    if ready:
        cur_norm = current_g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        history_blend = torch.tensor(scheduled_blend, dtype=torch.float32, device=current_g.device)

        if learn_blend:
            blend_max = float(group.get("muon_history_blend_max", 0.40))
            blend_init = float(group.get("muon_history_blend_init", base_blend if base_blend > 0.0 else 0.25))
            blend_prior = float(group.get("muon_history_blend_prior", 0.20))
            if "muon_history_blend_logit" not in state:
                state["muon_history_blend_logit"] = torch.tensor(
                    [_bounded_logit(blend_init, blend_max)], dtype=torch.float32, device=current_g.device
                )
                state["muon_history_blend_prior_logit"] = torch.tensor(
                    [_bounded_logit(blend_prior, blend_max)], dtype=torch.float32, device=current_g.device
                )
                state["muon_history_probe_plus"] = torch.zeros_like(current_g)
                state["muon_history_probe_minus"] = torch.zeros_like(current_g)

            z_current = current_g.float() / cur_norm
            meta_warmup = int(group.get("muon_history_meta_warmup", 64))
            if step > history_steps + meta_warmup:
                prev_plus_cos = (state["muon_history_probe_plus"].float() * z_current).sum(dim=(-2, -1), keepdim=True)
                prev_minus_cos = (state["muon_history_probe_minus"].float() * z_current).sum(dim=(-2, -1), keepdim=True)
                advantage = prev_plus_cos - prev_minus_cos
                meta_lr = float(group.get("muon_history_meta_lr", 0.03))
                meta_scale = float(group.get("muon_history_meta_scale", 0.01))
                meta_step = meta_lr * torch.tanh(advantage.mean() / max(meta_scale, 1e-12))
                state["muon_history_blend_logit"].add_(meta_step.to(state["muon_history_blend_logit"].dtype))
                prior_decay = float(group.get("muon_history_meta_prior_decay", 0.001))
                state["muon_history_blend_logit"].add_(
                    prior_decay * (state["muon_history_blend_prior_logit"] - state["muon_history_blend_logit"])
                )
                state["muon_history_blend_logit"].clamp_(min=-8.0, max=8.0)

            learned_blend = blend_max * torch.sigmoid(state["muon_history_blend_logit"].float())
            taper_mult = scheduled_blend / max(base_blend, 1e-12)
            history_blend = learned_blend * max(0.0, taper_mult)
        else:
            z_current = current_g.float() / cur_norm

        num_points = history_steps + 1
        x_mean = history_steps / 2.0
        x_pred = history_steps + float(group.get("muon_history_lookahead", 1.0))
        denom = sum((i - x_mean) ** 2 for i in range(num_points))
        current_weight = 1.0 / num_points + (x_pred - x_mean) * (history_steps - x_mean) / denom
        pred = current_g * current_weight
        for i in range(history_steps):
            weight = 1.0 / num_points + (x_pred - x_mean) * (i - x_mean) / denom
            pred = pred + history[i].to(current_g.dtype) * weight

        delta = pred - current_g
        delta_norm = delta.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        max_delta = float(group.get("muon_history_max_delta", 0.25)) * cur_norm
        clip_scale = torch.minimum(torch.ones_like(delta_norm), max_delta / delta_norm)
        delta = delta * clip_scale.to(delta.dtype)
        candidate = current_g + history_blend.to(delta.dtype) * delta
        cand_norm = candidate.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        cos = (candidate.float() * current_g.float()).sum(dim=(-2, -1), keepdim=True) / (cand_norm * cur_norm)
        aligned = cos >= float(group.get("muon_history_min_cos", 0.25))
        update = torch.where(aligned, candidate, current_g)
        update = update * (cur_norm / update.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)).to(update.dtype)

        if learn_blend:
            probe_gate = aligned.float()
            blend_max = float(group.get("muon_history_blend_max", 0.40))
            probe_eps = float(group.get("muon_history_blend_probe_eps", 0.05))
            b_plus = torch.clamp(history_blend + probe_eps, min=0.0, max=blend_max)
            b_minus = torch.clamp(history_blend - probe_eps, min=0.0, max=blend_max)
            probe_plus = current_g + (b_plus * probe_gate).to(delta.dtype) * delta
            probe_minus = current_g + (b_minus * probe_gate).to(delta.dtype) * delta
            probe_plus = probe_plus.float() / probe_plus.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
            probe_minus = probe_minus.float() / probe_minus.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
            state["muon_history_probe_plus"].copy_(probe_plus.to(state["muon_history_probe_plus"].dtype))
            state["muon_history_probe_minus"].copy_(probe_minus.to(state["muon_history_probe_minus"].dtype))
    elif learn_blend:
        cur_norm = current_g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        base_unit = current_g.float() / cur_norm
        if "muon_history_probe_plus" not in state:
            blend_max = float(group.get("muon_history_blend_max", 0.40))
            blend_init = float(group.get("muon_history_blend_init", base_blend if base_blend > 0.0 else 0.25))
            blend_prior = float(group.get("muon_history_blend_prior", 0.20))
            state["muon_history_blend_logit"] = torch.tensor(
                [_bounded_logit(blend_init, blend_max)], dtype=torch.float32, device=current_g.device
            )
            state["muon_history_blend_prior_logit"] = torch.tensor(
                [_bounded_logit(blend_prior, blend_max)], dtype=torch.float32, device=current_g.device
            )
            state["muon_history_probe_plus"] = torch.zeros_like(current_g)
            state["muon_history_probe_minus"] = torch.zeros_like(current_g)
        state["muon_history_probe_plus"].copy_(base_unit.to(state["muon_history_probe_plus"].dtype))
        state["muon_history_probe_minus"].copy_(base_unit.to(state["muon_history_probe_minus"].dtype))

    if history_steps > 1:
        history[:-1].copy_(history[1:].clone())
    history[history_steps - 1].copy_(current_g)
    return update

def muon_update_with_history(grad, momentum, state, step, group):
    momentum.lerp_(grad, 1 - group["mu"])
    update = grad.lerp(momentum, group["mu"])
    update = apply_muon_history(update, state, step, group)
    update = _apply_muon_eq(update, group["muon_eq_axis"])
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    if group["muon_plus"]:
        target_norm = min(update.size(-2), update.size(-1)) ** 0.5
        update = update * (target_norm / update.float().norm().clamp_min(1e-6)).to(update.dtype)
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, muon_plus=False, muon_eq_axis="none", muon_history=None):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, muon_plus=muon_plus, muon_eq_axis=muon_eq_axis)
        if muon_history:
            defaults.update(muon_history)
        self.step_count = 0
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if int(group.get("muon_history_steps", 0) or 0) > 0:
                        update = muon_update_with_history(p.grad, state["momentum"], state, self.step_count, group)
                    else:
                        update = muon_update(p.grad, state["momentum"], mu=group["mu"], muon_plus=group["muon_plus"], muon_eq_axis=group["muon_eq_axis"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
torch.manual_seed(SEED)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0(f"Using seed={SEED}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = int(os.environ.get("TRAIN_STEPS", "3350"))

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    opt_variant = os.environ.get("OPT_VARIANT", "muonplus_eqrow_hist4_lb_taper")
    muon_plus = opt_variant in {
        "muonplus",
        "muonplus_eqrow",
        "muonplus_eqrow_qklr",
        "muonplus_eqrow_hist4_fixed",
        "muonplus_eqrow_hist4_taper",
        "muonplus_eqrow_hist4_lb_taper",
    }
    muon_eq_axis = "row" if opt_variant in {
        "eqrow",
        "muonplus_eqrow",
        "muonplus_eqrow_qklr",
        "muonplus_eqrow_hist4_fixed",
        "muonplus_eqrow_hist4_taper",
        "muonplus_eqrow_hist4_lb_taper",
    } else "none"
    muon_history = None
    if opt_variant in {"muonplus_eqrow_hist4_fixed", "muonplus_eqrow_hist4_taper", "muonplus_eqrow_hist4_lb_taper"}:
        muon_history = dict(
            muon_history_steps=4,
            muon_history_blend=0.25,
            muon_history_lookahead=1.0,
            muon_history_min_cos=0.25,
            muon_history_max_delta=0.25,
        )
        if opt_variant in {"muonplus_eqrow_hist4_taper", "muonplus_eqrow_hist4_lb_taper"}:
            muon_history.update(
                muon_history_decay_start_step=int(0.39 * train_steps),
                muon_history_decay_end_step=train_steps,
                muon_history_final_blend_mult=0.0,
            )
        if opt_variant == "muonplus_eqrow_hist4_lb_taper":
            muon_history.update(
                muon_history_learn_blend=True,
                muon_history_blend_max=0.40,
                muon_history_blend_init=0.25,
                muon_history_blend_prior=0.20,
                muon_history_meta_lr=0.03,
                muon_history_meta_scale=0.01,
                muon_history_meta_prior_decay=0.001,
                muon_history_meta_warmup=64,
                muon_history_blend_probe_eps=0.05,
            )
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    if opt_variant == "muonplus_eqrow_qklr":
        qk_params = [p for name, p in model.blocks.named_parameters() if name.endswith("attn.q.weight") or name.endswith("attn.k.weight")]
        other_params = [p for name, p in model.blocks.named_parameters() if p.ndim >= 2 and not (name.endswith("attn.q.weight") or name.endswith("attn.k.weight"))]
        optimizer2 = Muon(qk_params, lr=0.035 * float(os.environ.get("QK_LR_MULT", "2.5")), weight_decay=0.025, muon_plus=muon_plus, muon_eq_axis=muon_eq_axis, muon_history=muon_history)
        optimizer3 = Muon(other_params, lr=0.035, weight_decay=0.025, muon_plus=muon_plus, muon_eq_axis=muon_eq_axis, muon_history=muon_history)
        optimizers = [optimizer1, optimizer2, optimizer3]
    else:
        optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                          lr=0.035, weight_decay=0.025, muon_plus=muon_plus, muon_eq_axis=muon_eq_axis, muon_history=muon_history)
        optimizers = [optimizer1, optimizer2]
    print0(f"opt_variant:{opt_variant} train_steps:{train_steps} muon_plus:{muon_plus} muon_eq_axis:{muon_eq_axis} muon_history:{muon_history}", console=True)
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
