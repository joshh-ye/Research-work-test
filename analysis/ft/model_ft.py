"""Fine-tunable Borzoi (replicate 0, borzoi-pytorch 0.4.4) + CRC head.  hg38, 524,288 bp input, 183 targets at 32 bp,
central 4,096 bins (offset (6144-4096)/2 = 1024 in the 6144-bin trunk output, as borzoi_code/model.py).

The trunk output used is the INPUT of `human_head` (= output of `final_joined_convs`: ConvBlock 1536->1920 + Dropout + GELU),
exactly the tensor tapped by the pre-hook in borzoi_code/model.py.  The 7,611-track human head is deleted (never computed).
The forward is re-implemented stage by stage (identical ops to `Borzoi.get_embs_after_crop` + `final_joined_convs`; tested for
exact equality in fp32) so that (a) stages before the first trainable parameter run under no_grad and (b) stages can be wrapped
in activation checkpointing.

Trunk is ALWAYS in eval mode (BatchNorm uses pretrained running statistics - batch = 1 tile makes batch statistics unusable;
dropout off unless cfg trunk_dropout=True, which switches only nn.Dropout modules to train).

Stages (parameter prefix -> stage id):
  0  conv_dna, res_tower                       (stem + 5 conv blocks, 4 max-pools; output 1280 x 16384)
  1  unet1, horizontal_conv0, horizontal_conv1 (-> 1536 x 4096 tokens + the two skip tensors)
  2..9 transformer.0 .. transformer.7          (attention + MLP, 1536-d, 4096 tokens)
  10 upsampling_unet1, separable1, upsampling_unet0, separable0, final_joined_convs  (U-Net decoder, crop 6144, 1x1 -> 1920)
Unfreeze spec (partial mode): comma list of  blocks:K (last K transformer blocks), up (decoder: upsampling_unet*, separable*),
final (final_joined_convs), horiz (horizontal_conv0/1), unet1, res (stem + res_tower).
LoRA (hand-written; PEFT not installed): targets attn (to_q,to_k,to_v,to_out Linear), mlp (the two Linear of each block FFN),
conv1x1 (1x1 Conv1d: horizontal_conv0/1, upsampling_unet0/1 first conv, final_joined_convs conv).  Not LoRA-compatible here:
kernel-5 convs of res_tower/unet1, depthwise-separable convs, relative-position projection `to_rel_k` (could be, not targeted),
BatchNorm/LayerNorm (frozen).  B is zero-initialised so the initial model equals the frozen model.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

os.environ.setdefault("HF_HOME", "/home/yejz1/experimental_research/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

EMB_DIM = 1920
CENTER_BINS = 4096
N_STAGES = 11


def stage_of(name: str) -> int:
    """Parameter name (relative to trunk) -> stage id."""
    if name.startswith(("conv_dna", "res_tower")):
        return 0
    if name.startswith(("unet1", "horizontal_conv")):
        return 1
    m = re.match(r"transformer\.(\d+)\.", name)
    if m:
        return 2 + int(m.group(1))
    return 10


# ----------------------------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale


class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Conv1d, r: int, alpha: float):
        super().__init__()
        assert base.kernel_size == (1,) and base.groups == 1
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_channels))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):  # x: (B, C, L)
        h = torch.einsum("rc,bcl->brl", self.lora_A.to(x.dtype), x)
        return self.base(x) + torch.einsum("or,brl->bol", self.lora_B.to(x.dtype), h) * self.scale


def _set(parent_root: nn.Module, dotted: str, new: nn.Module):
    parts = dotted.split(".")
    p = parent_root
    for q in parts[:-1]:
        p = p[int(q)] if q.isdigit() else getattr(p, q)
    if parts[-1].isdigit():
        p[int(parts[-1])] = new
    else:
        setattr(p, parts[-1], new)


def lora_candidates(trunk: nn.Module, targets: set, n_last_blocks: int, depth: int = 8):
    """-> list of (dotted_name, module, kind)."""
    out = []
    first_blk = depth - n_last_blocks
    for name, m in trunk.named_modules():
        mm = re.match(r"transformer\.(\d+)\.", name)
        if mm and int(mm.group(1)) >= first_blk:
            if "attn" in targets and isinstance(m, nn.Linear) and re.search(r"\.(to_q|to_k|to_v|to_out)$", name):
                out.append((name, m, "linear"))
            if "mlp" in targets and isinstance(m, nn.Linear) and re.match(r"transformer\.\d+\.1\.fn\.(1|4)$", name):
                out.append((name, m, "linear"))
        if "conv1x1" in targets and isinstance(m, nn.Conv1d) and name.endswith("conv_layer") and m.kernel_size == (1,) \
                and m.groups == 1 and not name.startswith("transformer"):
            out.append((name, m, "conv"))
    return out


def apply_lora(trunk: nn.Module, targets: set, r: int, alpha: float, n_last_blocks: int):
    cands = lora_candidates(trunk, targets, n_last_blocks)
    for name, m, kind in cands:
        _set(trunk, name, (LoRALinear if kind == "linear" else LoRAConv1x1)(m, r, alpha))
    return [c[0] for c in cands]


# ----------------------------------------------------------------------------------------------------------------- model
def load_trunk(repo_id="johahi/borzoi-replicate-0"):
    from borzoi_pytorch import Borzoi
    trunk = Borzoi.from_pretrained(repo_id)
    if hasattr(trunk, "human_head"):
        del trunk.human_head          # 7,611-track head never needed
    return trunk


def frozen_manifest(repo_id="johahi/borzoi-replicate-0"):
    snap = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    rev = sorted(snap.iterdir())[-1]
    f = (rev / "model.safetensors").resolve()
    return dict(hf_repo_id=repo_id, revision=rev.name, safetensors_blob=f.name, safetensors_bytes=f.stat().st_size,
                note="frozen/initial trunk weights are re-loaded from the HF cache on resume; checkpoints hold trainable params only")


class FTModel(nn.Module):
    def __init__(self, cfg: dict, n_out: int = 183, repo_id="johahi/borzoi-replicate-0"):
        super().__init__()
        self.cfg = cfg
        self.trunk = load_trunk(repo_id)
        self.trunk_arch = dict(self.trunk.config.to_dict())
        self.ckpt_mode = cfg.get("grad_ckpt", "transformer")   # none | transformer | all
        self.trunk_dropout = bool(cfg.get("trunk_dropout", False))
        mode = cfg["mode"]
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.lora_modules = []
        if mode == "partial":
            self._unfreeze(cfg.get("unfreeze", "blocks:2"))
        elif mode == "lora":
            tg = set(cfg.get("lora_targets", "attn").split(","))
            self.lora_modules = apply_lora(self.trunk, tg, cfg.get("lora_r", 8), cfg.get("lora_alpha", 16.0), cfg.get("lora_blocks", 8))
        elif mode == "full":
            for p in self.trunk.parameters():
                p.requires_grad = True
        elif mode != "frozen":
            raise ValueError(mode)
        h = cfg.get("head", "mlp")
        if h == "mlp":
            self.head = nn.Sequential(nn.Linear(EMB_DIM, cfg.get("head_hidden", 1024)), nn.GELU(), nn.Dropout(cfg.get("head_dropout", 0.2)),
                                      nn.Linear(cfg.get("head_hidden", 1024), n_out), nn.Softplus())
        elif h == "linear":
            self.head = nn.Sequential(nn.Linear(EMB_DIM, n_out), nn.Softplus())
        else:
            raise ValueError(h)
        # trainable stage bookkeeping
        stages = [stage_of(n) for n, p in self.trunk.named_parameters() if p.requires_grad]
        self.first_train_stage = min(stages) if stages else N_STAGES   # N_STAGES => whole trunk under no_grad
        self.train_stage_ids = sorted(set(stages))
        self._setup_dropout()

    def _unfreeze(self, spec: str):
        prefixes = []
        for tok in spec.split(","):
            tok = tok.strip()
            if tok.startswith("blocks:"):
                k = int(tok.split(":")[1])
                prefixes += [f"transformer.{i}." for i in range(8 - k, 8)]
            elif tok == "up":
                prefixes += ["upsampling_unet1", "separable1", "upsampling_unet0", "separable0"]
            elif tok == "final":
                prefixes += ["final_joined_convs"]
            elif tok == "horiz":
                prefixes += ["horizontal_conv0", "horizontal_conv1"]
            elif tok == "unet1":
                prefixes += ["unet1"]
            elif tok == "res":
                prefixes += ["conv_dna", "res_tower"]
            elif tok:
                raise ValueError(f"bad unfreeze token {tok}")
        n = 0
        for name, p in self.trunk.named_parameters():
            if any(name.startswith(pf) for pf in prefixes):
                p.requires_grad = True
                n += 1
        assert n > 0, f"unfreeze spec {spec!r} matched nothing"

    def _setup_dropout(self):
        self.trunk.eval()
        if self.trunk_dropout:
            for m in self.trunk.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self._setup_dropout()          # trunk stays eval (BN) regardless
        return self

    # ---- staged trunk forward
    def _run(self, stage: int, fn, *args):
        if not torch.is_grad_enabled():
            return fn(*args)
        if stage < self.first_train_stage:
            with torch.no_grad():
                return fn(*args)
        use_ck = self.ckpt_mode == "all" or (self.ckpt_mode == "transformer" and 2 <= stage <= 9)
        if use_ck:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def trunk_embedding(self, x):
        """x: (B, 4, 524288) float -> (B, 1920, 6144) trunk embedding (human_head input)."""
        t = self.trunk

        def s0(x):
            return t.res_tower(t.conv_dna(x))

        def s1(x0):
            xu1 = t.unet1(x0)
            xt = t._max_pool(xu1)
            return xt.permute(0, 2, 1), t.horizontal_conv1(xu1), t.horizontal_conv0(x0)

        def s10(xt, h1, h0):
            x = xt.permute(0, 2, 1)
            x = t.upsampling_unet1(x) + h1
            x = t.separable1(x)
            x = t.upsampling_unet0(x) + h0
            x = t.separable0(x)
            x = t.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
            return t.final_joined_convs(x)

        x0 = self._run(0, s0, x)
        xt, h1, h0 = self._run(1, s1, x0)
        for i, blk in enumerate(t.transformer):
            xt = self._run(2 + i, blk, xt)
        return self._run(10, s10, xt, h1, h0)

    def forward(self, codes: torch.Tensor, autocast: bool = True):
        """codes: (B, L) uint8 in {0:A,1:C,2:G,3:T,4:N}. Returns (B, 4096, n_out) float32 non-negative."""
        x = F.one_hot(codes.long(), 5)[..., :4].permute(0, 2, 1).float().contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            e = self.trunk_embedding(x)
        n = e.shape[-1]
        off = (n - CENTER_BINS) // 2
        e = e[:, :, off:off + CENTER_BINS].permute(0, 2, 1).float()
        with torch.autocast("cuda", enabled=False):
            return self.head(e)

    # ---- param groups
    def param_groups(self, lr_head, lr_trunk, lr_lora, wd, trunk_wd):
        head, trunk, lora = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head if n.startswith("head.") else lora if "lora_" in n else trunk).append(p)
        g = []
        if head:
            g.append(dict(params=head, lr=lr_head, weight_decay=wd, name="head"))
        if trunk:
            g.append(dict(params=trunk, lr=lr_trunk, weight_decay=trunk_wd, name="trunk"))
        if lora:
            g.append(dict(params=lora, lr=lr_lora, weight_decay=0.0, name="lora"))
        return g

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def module_tree_summary(trunk: nn.Module) -> str:
    lines = [f"{'module':44s} {'params':>12s}  stage"]
    tot = 0
    for n, m in trunk.named_children():
        c = sum(p.numel() for p in m.parameters())
        tot += c
        if n == "transformer":
            for i, b in enumerate(m):
                a = sum(p.numel() for p in b[0].parameters())
                f = sum(p.numel() for p in b[1].parameters())
                lines.append(f"transformer.{i} (attn {a/1e6:.2f}M + ffn {f/1e6:.2f}M)".ljust(44) + f" {a + f:12,d}  {2 + i}")
        else:
            lines.append(f"{n:44s} {c:12,d}  {stage_of(n + '.x')}")
    lines.append(f"{'TOTAL trunk (no human head)':44s} {tot:12,d}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--tree" in sys.argv:
        t = load_trunk()
        print(module_tree_summary(t))
        print(t)
