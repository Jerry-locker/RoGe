# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import xformers.ops as xops


def _get_flash_attention_ops():
    """Pick flash-attention ops that are actually built for this GPU/xformers.

    Prefer Flash Attention 3 on Hopper+ when available; otherwise FA2.
    Returns None to let xFormers auto-dispatch if neither is usable.
    """
    if not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()
    candidates = []
    if major >= 9:
        candidates.append(("Flash Attention 3", "flash3"))
    candidates.append(("Flash Attention 2", "flash"))

    for label, mod_name in candidates:
        mod = getattr(xops.fmha, mod_name, None)
        if mod is None:
            continue
        fw_op, bw_op = mod.FwOp, mod.BwOp
        try:
            available = fw_op.is_available()
        except Exception:
            available = False
        if available:
            return (fw_op, bw_op)
        print(f"{label} not available in this xFormers build; trying fallback")

    print("No explicit flash-attn op available; using xFormers auto-dispatch")
    return None


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias=False,
        fc_bias=False,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        self.attn_dropout = attn_dropout

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Get appropriate flash attention ops based on GPU
        self.flash_attn_ops = _get_flash_attention_ops()

    def forward(self, q: torch.Tensor, kv=None) -> torch.Tensor:
        # attention block that supports non-query keys and values
        if kv is None:
            kv = q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q, k, v = (
            einops.rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        x = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=self.attn_dropout if self.training else 0.0,
            op=self.flash_attn_ops,
        )

        x = einops.rearrange(x, "b n h d -> b n (h d)")

        x = self.attn_fc_dropout(self.proj(x))
        return x
