"""
Attention and its kernel-optimized variants.
"""

from typing import Optional, Tuple

import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F


xformers_installed = True
try:
    from xformers.ops import memory_efficient_attention
    from xformers.components.attention import ScaledDotProduct
except ImportError:
    xformers_installed = False


flash_installed = True
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    flash_installed = False


def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        qkv_bias=False,
        dropout=0.0,
        attention_mode="standard",
    ):
        super().__init__()
        assert attention_mode in [
            "standard",
            "xformers_scaled_dot_product",
            "xformers_memory_efficient",
            "flash",
        ]
        self.attention_mode = attention_mode
        self.use_flash = flash_installed and attention_mode == "flash"

        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if attention_mode == "xformers_scaled_dot_product":
            self.xformers_scaled_dot_product_fn = ScaledDotProduct()

    def xformers_scaled_dot_product_attention(self, x, mask=None):
        if not xformers_installed:
            raise ImportError("xformers is not installed, cannot use xformer attention")

        b, l, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        # xformers scaled dot product attention fn applies the scaling by dim_head ** -0.5 here:
        # https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/core.py#L207

        if mask.ndim == 2:
            # mask = repeat(mask, "b l -> b h l l_prime", h=h, l_prime=l)
            mask = repeat(mask, "b l -> b l l_prime", l_prime=l)

        out = self.xformers_scaled_dot_product_fn(q, k, v, att_mask=mask)
        # out = rearrange(out, "b h l d -> b l (h d)", h=h)
        return self.to_out(out)

    def xformers_memory_efficient_attention(self, x, mask=None):
        if not xformers_installed:
            raise ImportError("xformers is not installed, cannot use xformer attention")

        dtype, device = x.dtype, x.device
        b, l, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # xformers memory efficient attention implementation automatically applies the scaling by dim_heads ** -0.5
        # https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py#L219

        # expects query/key/value tensors to have shape [B, L, H, D]
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b l h d", h=h), (q, k, v))

        # Create attn_bias from padding mask
        if mask is not None:
            attn_bias = torch.zeros_like(mask, dtype=dtype)
            attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
            attn_bias = rearrange(
                attn_bias, "b l -> b () () l"
            )  # Shape: (batch_size, 1, 1, seq_len)
            attn_bias = attn_bias.repeat(1, h, l, 1)
            attn_bias = attn_bias.to(device)
        else:
            attn_bias = None

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b l h d -> b l (h d)")
        return self.to_out(out)

    def standard_attention(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, "b j -> b () () j")
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b l (h d)", h=h)
        return self.to_out(out)

    def flash_attention_padded(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Flash Attention 2 does not implement padding and attention mask in the kernel operation,
        but does offer utilities to make use of `flash_attn_varlen_qkvpacked_func` from (B, L) padding mask.

        Inspired by https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
        """

        h, d = self.heads, self.dim_head
        b, l, _ = x.size()
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        # Transform the data into the format required by flash attention
        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.transpose(1, 3)  # shape: [b, l, 3, num_heads, head_dim]
        key_padding_mask = mask  # shape: [b, l]

        if key_padding_mask is None:
            qkv = qkv.reshape(-1, 3, h, d)
            cu_q_lens = torch.arange(
                0, (b + 1) * l, step=l, dtype=torch.int32, device=qkv.device
            )
            max_s = l
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=self.scale, causal=False
            )
            output = output.view(b, l, -1)
        else:
            # hidden_states: (batch, seqlen, ...)
            # attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
            # qkv = qkv.reshape(b, l, -1)
            qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)

            # If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
            # (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            qkv = qkv.view(-1, 3, h, d)
            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=self.scale, causal=False
            )
            output_unpad = output_unpad.reshape(-1, h, d)
            output = pad_input(output_unpad, indices, b, l)  # shape: [b, l, h, d]
            output = rearrange(output, "b l h d -> b l (h d)", h=h)

        return output

    def forward(self, x, mask=None):
        if self.attention_mode == "xformers_scaled_dot_product":
            return self.xformers_scaled_dot_product_attention(x, mask)
        elif self.attention_mode == "xformers_memory_efficient":
            return self.xformers_memory_efficient_attention(x, mask)
        elif self.attention_mode == "flash":
            return self.flash_attention_padded(x, mask)
        else:
            return self.standard_attention(x, mask)
