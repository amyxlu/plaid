import time
from typing import List, Tuple

import torch
import random
import einops

from .esmfold import esmfold_v1


def mask_from_seq_lens(x: torch.Tensor, seqlen: torch.Tensor):
    mask = torch.arange(x.shape[1], device=x.device)
    mask = einops.repeat(mask[None, :], "1 L -> N L", N=x.shape[0]) < seqlen[:, None]
    return mask.long()


def get_random_sequence_crop(s, length):
    if len(s) > length:
        start = random.randint(0, len(s) - length)
        return s[start : start + length]
    else:
        return s


def get_random_sequence_crop_batch(sequence_batch, max_len, min_len=None):
    if not min_len is None:
        sequence_batch = list(filter(lambda s: len(s) >= min_len, sequence_batch))
    return [get_random_sequence_crop(seq, max_len) for seq in sequence_batch]


# def trim_or_pad(tensor: torch.Tensor, pad_to: int, length_dim=0, pad_idx=0):
#     """Trim or pad a tensor with shape (..., L, ...) to a given length."""
#     L = tensor.shape[length_dim]
#     if L >= pad_to:
#         tensor = tensor.index_select(length_dim, torch.arange(length_dim))

#     elif L < pad_to:
#         padding = torch.full(
#             size=(*tensor.shape[:length_dim], pad_to - L, *tensor.shape[length_dim + 1:]),
#             fill_value=pad_idx,
#             dtype=tensor.dtype,
#             device=tensor.device,
#         )
#         tensor = torch.concat((tensor, padding), dim=length_dim)
#     return tensor


# def trim_or_pad(tensor: torch.Tensor, pad_to: int, length_dim=0, pad_idx=0):
#     """Trim or pad a tensor with shape (..., L, ...) to a given length."""
#     > might have a bug in it
#
#     L = tensor.shape[length_dim]
#     if L >= pad_to:
#         tensor = tensor.index_select(length_dim, torch.arange(length_dim))

#     elif L < pad_to:
#         padding = torch.full(
#             size=(*tensor.shape[:length_dim], pad_to - L, *tensor.shape[length_dim + 1:]),
#             fill_value=pad_idx,
#             dtype=tensor.dtype,
#             device=tensor.device,
#         )
#         tensor = torch.concat((tensor, padding), dim=length_dim)
#     return tensor


def trim_or_pad_length_first(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (L, ...) to a given length."""
    L = tensor.shape[0]
    if L >= pad_to:
        # trim, assuming first dimension is the dim to trim
        tensor = tensor[:pad_to]
    elif L < pad_to:
        padding = torch.full(
            size=(pad_to - tensor.shape[0], *tensor.shape[1:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=0)
    return tensor


def trim_or_pad_batch_first(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (B, L, ...) to a given length."""
    N, L = tensor.shape[0], tensor.shape[1]
    if L >= pad_to:
        tensor = tensor[:, :pad_to, ...]
    elif L < pad_to:
        padding = torch.full(
            size=(N, pad_to - L, *tensor.shape[2:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=1)
    return tensor


class ESMFoldEmbed:
    def __init__(
        self,
        esmfold=None,
        shorten_len_to=None,
        hourglass_model=None,
        latent_scaler=None,
        unscale_and_compress=False,
    ):
        if esmfold is None:
            esmfold = esmfold_v1()
        self.esmfold = esmfold
        self.esmfold = self.esmfold.eval().requires_grad_(False)
        if shorten_len_to is None:
            self.transform = lambda x: x
        else:
            self.transform = lambda batch: get_random_sequence_crop_batch(
                batch, max_len=shorten_len_to
            )

        # maybe also scale and compress
        self.hourglass_model = hourglass_model
        self.scaler = latent_scaler
        self.unscale_and_compress = unscale_and_compress

    def embed_fn(self, sequence: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.esmfold.infer_embedding(sequence)
        embedding = output["s"]
        mask = output["mask"]

        if self.unscale_and_compress:
            assert not self.hourglass_model is None
            assert not self.scaler is None
            embedding = self.scaler.scale(embedding)
            compressed_embedding, downsampled_mask = self.hourglass_model.compress(
                embedding
            )

        return compressed_embedding, downsampled_mask

    def __call__(self, sequence, device=None):
        sequence = self.transform(sequence)
        if not device is None:
            self.esmfold = self.esmfold.to(device)
        return self.embed_fn(sequence)


class ESMEmbed:
    def __init__(self, lm_embedder_type, device=None):
        self.lm_embedder_type = lm_embedder_type
        self.embedder, alphabet = self.make_embedder(lm_embedder_type)
        self.batch_converter = self.alphabet.get_batch_converter()
        if not device is None:
            self.embedder = self.embedder.to(device)
            self.device = device

    def to(self, device):
        self.embedder = self.embedder.to(device)
        return self

    def make_embedder(lm_embedder_type):
        start = time.time()
        print("loading LM from torch hub")
        embedder, alphabet = torch.hub.load(
            "facebookresearch/esm:main", lm_embedder_type
        )
        embedder = embedder.eval().to("cuda")
        for param in embedder.parameters():
            param.requires_grad = False

        end = time.time()
        print(f"done loading model in {end - start:.2f} seconds.")
        return embedder, alphabet

    def embed_batch_esm(self, sequences, max_len=512):
        sequences = get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=0
        )
        batch = [("", seq) for seq in sequences]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.embedder(
                tokens, repr_layers=[self.repr_layer], return_contacts=False
            )
            feats = results["representations"][self.repr_layer]

        seq_lens = [len(seq) for seq in sequences]
        seq_lens = torch.tensor(seq_lens, device=self.device, dtype=torch.int16)
        return feats, tokens, seq_lens, sequences
