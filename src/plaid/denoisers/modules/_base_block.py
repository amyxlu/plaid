"""
For backwards compatibility with U-Evoformer based denoiser experiments.
"""

import typing as T
import torch
import torch.nn as nn
import abc
import einops


class BaseBlock(nn.Module):
    """
    Provide simple conditioning utilities for denoiser blocks.
    """

    def __init__(self, conditioning_strategy, hid_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conditioning_mlp = None
        self.extras = 0
        if not conditioning_strategy is None:
            if conditioning_strategy == "hidden_concat":
                self.conditioning_mlp = self.make_projection_mlp(hid_dim * 2, hid_dim)
            if conditioning_strategy == "length_concat":
                self.extras = 1

    def make_projection_mlp(self, in_dim, hid_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim, bias=True),
        )

    def pointwise_add(self, x, c):
        # x: (B, L, C)
        # c: (B, C)
        c = einops.repeat(c, "b c -> b l c", l=x.shape[1])
        assert x.shape == c.shape
        x = x + c
        return x

    def length_concat(self, x, c):
        # x: (B, L, C)
        # c: (B, C)
        c = einops.rearrange(c, "b c -> b 1 c")
        x = torch.concat((x, c), dim=1)
        return x

    def hidden_concat(self, x, c):
        # x: (B, L, C)
        # c: (B, C)
        c = einops.repeat(c, "b c -> b l c", l=x.shape[1])
        x = torch.concat((x, c), dim=-1)
        return x

    def conditioning(
        self,
        x,
        c,
        strategy: T.Optional[str] = None,
        proj_fn: T.Optional[T.Callable] = None,
    ):
        if strategy is None:
            return x

        assert strategy in ["length_concat", "add", "hidden_concat"]

        if strategy == "length_concat":
            x = self.length_concat(x, c)  # (N, L + 1, C)
        elif strategy == "add":
            x = self.pointwise_add(x, c)  # (N, L, C)
        elif strategy == "hidden_concat":
            assert not proj_fn is None
            x = self.hidden_concat(x, c)  # (N, L, C * 2)
        else:
            raise NotImplementedError

        if not proj_fn is None:
            return proj_fn(x)
        else:
            return x

    @abc.abstractmethod
    def forward(self, x, c, mask=None, z=None, *args, **kwargs):
        raise NotImplementedError
