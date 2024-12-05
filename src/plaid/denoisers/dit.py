import torch
from torch import nn

from .modules import (
    BaseDenoiser,
    Mlp,
    Attention,
)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    """
    Block using Flash Attention components, optionally with:
    - ada-LN conditioning (https://arxiv.org/abs/2212.09748)
    - skip connections (https://arxiv.org/abs/2209.12152)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_skip_connect=False,
        attention_mode="xformers_scaled_dot_product",
        **block_kwargs
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        assert attention_mode in [
            "standard",
            "xformers_scaled_dot_product",
            "xformers_memory_efficient",
            "flash",
        ]
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=False,
            attention_mode=attention_mode,
            dropout=0.0,
            **block_kwargs
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.skip_linear = (
            Mlp(
                in_features=hidden_size * 2,
                hidden_features=hidden_size,
                out_features=hidden_size,
                norm_layer=nn.LayerNorm,
                act_layer=approx_gelu,
            )
            if use_skip_connect
            else None
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask, skip=None):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        if skip is not None and self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FunctionOrganismUDiT(BaseDenoiser):
    def __init__(
        self,
        input_dim=32,
        hidden_size=1024,
        max_seq_len=256,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_conditioning=True,
        timestep_embedding_strategy="sinusoidal",
        use_skip_connect=False,
        attention_mode="xformers_scaled_dot_product",
    ):
        self.attention_mode = attention_mode
        self.use_skip_connect = use_skip_connect

        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_self_conditioning=use_self_conditioning,
            timestep_embedding_strategy=timestep_embedding_strategy,
            max_seq_len=max_seq_len,
        )

        self.initialize_weights()
        self.initialize_adaln_weights()

    def make_output_projection(self):
        return FinalLayer(self.hidden_size, self.input_dim)

    def _make_denoising_blocks(self):
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_skip_connect=False,
                    attention_mode=self.attention_mode,
                )
                for _ in range(self.depth)
            ]
        )

    def _make_denoising_blocks_with_skip(self):
        in_blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_skip_connect=self.use_skip_connect,
                attention_mode=self.attention_mode,
            )
            for _ in range(self.depth // 2)
        ]
        mid_block = DiTBlock(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            use_skip_connect=self.use_skip_connect,
            attention_mode=self.attention_mode,
        )
        out_blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_skip_connect=self.use_skip_connect,
                attention_mode=self.attention_mode,
            )
            for _ in range(self.depth // 2)
        ]

        # these will be used in the forward pass & weight initialization
        self.in_blocks = nn.ModuleList(in_blocks)
        self.mid_block = mid_block
        self.out_blocks = nn.ModuleList(out_blocks)

    def make_denoising_blocks(self, *args, **kwargs):
        if self.use_skip_connect:
            self._make_denoising_blocks_with_skip(*args, **kwargs)
        else:
            self._make_denoising_blocks(*args, **kwargs)

    def blocks_forward_pass(self, x, c, mask, *args, **kwargs):
        if self.use_skip_connect:
            assert (
                hasattr(self, "in_blocks")
                and hasattr(self, "mid_block")
                and hasattr(self, "out_blocks")
            )
            skips = []

            for block in self.in_blocks:
                x = block(x, c, mask, skip=None)
                skips.append(x)

            x = self.mid_block(x, c, mask, skip=None)

            for block in self.out_blocks:
                x = block(x, c, mask, skip=skips.pop())

        else:
            for block in self.blocks:
                x = block(x, c, mask)

        return x

    def initialize_adaln_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        if self.use_skip_connect:
            for block in self.in_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
