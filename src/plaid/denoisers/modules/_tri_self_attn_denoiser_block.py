"""
Adapted from the original ESMFold triangular self-attention
in order to add conditioning pieces and skip connections for diffusion.

This is not used in the final model, but is used in our
ICML ML4LMS Workshop paper.
"""

import torch.nn as nn
import torch
from openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

from ...esmfold.misc import (
    Attention,
    Dropout,
    PairToSequence,
    ResidueMLP,
    SequenceToPair,
)

from ._base_block import BaseBlock


class TriangularSelfAttentionBlock(BaseBlock):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.

    # Note: this should be entirely compatible with the original ESMFold weights
    # in order to finetune for diffusion.

    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        conditioning_strategy=None,
        dropout=0,
        skip=False,
        **__kwargs,
    ):
        super().__init__(conditioning_strategy, sequence_state_dim)
        assert sequence_state_dim % sequence_head_width == 0
        assert pairwise_state_dim % pairwise_head_width == 0
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width
        assert sequence_state_dim == sequence_num_heads * sequence_head_width
        assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        assert pairwise_state_dim % 2 == 0

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = SequenceToPair(
            sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim
        )
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

        self.seq_attention = Attention(
            sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        ###################################################################################
        ###################################################################################
        # Modified from original to add projection for concatenated skip connection
        # Note: these weights are not in the original model
        self.skip_seq_state_proj = (
            nn.Linear(2 * sequence_state_dim, sequence_state_dim) if skip else None
        )
        self.skip_pairwise_state_proj = (
            nn.Linear(2 * pairwise_state_dim, pairwise_state_dim) if skip else None
        )
        ###################################################################################
        ###################################################################################

        self.mlp_seq = ResidueMLP(
            sequence_state_dim, 4 * sequence_state_dim, dropout=dropout
        )
        self.mlp_pair = ResidueMLP(
            pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout
        )

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(
        self,
        x,
        z,
        c=None,  # for backwards compatibility only
        mask=None,
        chunk_size=None,
        skip_seq_state=None,
        skip_pairwise_state=None,
        conditioning_strategy=None,
        **__kwargs,
    ):
        """
        Inputs:
          x: sequence_state: B x L x sequence_state_dim
          z: pairwise_state: B x L x L x pairwise_state_dim
          c: conditioning: B x C
          mask: B x L boolean tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        # check shapes
        sequence_state, pairwise_state = x, z
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]

        ###################################################################################
        ###################################################################################
        # Modified to add conditioning
        sequence_state = self.conditioning(
            sequence_state,
            c,
            strategy=conditioning_strategy,
            proj_fn=self.conditioning_mlp,
        )

        # Modified to add skip connection concatenation
        if not skip_pairwise_state is None:
            assert not self.skip_pairwise_state_proj is None
            pairwise_state = self.skip_pairwise_state_proj(
                torch.cat([pairwise_state, skip_pairwise_state], dim=-1)
            )
        if not skip_seq_state is None:
            assert not self.skip_seq_state_proj is None
            sequence_state = self.skip_seq_state_proj(
                torch.cat([sequence_state, skip_seq_state], dim=-1)
            )

        ###################################################################################
        ###################################################################################

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state
