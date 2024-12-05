from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import linalg
import math
import torch
import einops

from ..typed import DeviceLike, PathLike, ArrayLike
from ..datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES


"""
Clean FID implementations
https://arxiv.org/abs/2104.11222
"""


def default(x, val):
    return x if x is not None else val


def npy(x: ArrayLike):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


def is_fn_conditional(fn_idx):
    ret = True
    if fn_idx is None:
        ret = False
    if fn_idx == NUM_FUNCTION_CLASSES:
        ret = False
    return ret


def is_org_conditional(org_idx):
    ret = True
    if org_idx is None:
        ret = False
    if org_idx == NUM_ORGANISM_CLASSES:
        ret = False
    return ret


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


"""
Compute the KID score given the sets of features
"""


def parmar_kid(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


def parmar_fid(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    try:
        return frechet_distance(mu1, sig1, mu2, sig2)
    except ValueError as e:
        print(e)
        return np.nan


def polynomial_kernel(x, y):
    d = x.shape[-1]
    dot = x @ y.transpose(-2, -1)
    return (dot / d + 1) ** 3


def squared_mmd(x, y, kernel=polynomial_kernel):
    m = x.shape[-2]
    n = y.shape[-2]
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(dim1=-1, dim2=-2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(dim1=-1, dim2=-2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    term_1 = kxx_sum / m / (m - 1)
    term_2 = kyy_sum / n / (n - 1)
    term_3 = kxy_sum * 2 / m / n
    return term_1 + term_2 - term_3


def calc_kid_fn(x, y, max_size=5000):
    x_size, y_size = x.shape[0], y.shape[0]
    n_partitions = math.ceil(max(x_size / max_size, y_size / max_size))
    total_mmd = x.new_zeros([])
    for i in range(n_partitions):
        cur_x = x[
            round(i * x_size / n_partitions) : round((i + 1) * x_size / n_partitions)
        ]
        cur_y = y[
            round(i * y_size / n_partitions) : round((i + 1) * y_size / n_partitions)
        ]
        total_mmd = total_mmd + squared_mmd(cur_x, cur_y)
    return total_mmd / n_partitions


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise RuntimeError("tensor of matrices must have at least 2 dimensions")
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError("tensor must be batches of square matrices")
    return _MatrixSquareRootEig.apply(a)


def calc_fid_fn(x, y, eps=1e-8):
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    mean_term = (x_mean - y_mean).pow(2).sum()
    x_cov = torch.cov(x.T)
    y_cov = torch.cov(y.T)
    eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
    x_cov = x_cov + eps_eye
    y_cov = y_cov + eps_eye
    x_cov_sqrt = sqrtm_eig(x_cov)
    cov_term = torch.trace(
        x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt)
    )
    return mean_term + cov_term


class ConditionalDistributionDistance:
    """
    Given a sampled latent from a function and organism, compute the distribution distance.

    Currently supports: sinkhorn (multi-dimensional Wasserstein) and FID.

    This will look in the cache to see if we've already computed the CHEAP latent embedding
    for this function + organism combination. If not, it will load the CHEAP pipeline,
    look for examples in the validation data with this combination, run the pipeline,
    and save the results to cache, if desired.
    """

    def __init__(
        self,
        function_idx: Optional[int] = None,
        organism_idx: Optional[int] = None,
        val_parquet_fpath: Optional[PathLike] = "/data/lux70/data/pfam/val.parquet",
        cached_pt_dir: Optional[PathLike] = "/data/lux70/data/pfam/features",
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_eval_samples: int = 512,
        device: DeviceLike = "cuda",
        cheap_pipeline: Optional[torch.nn.Module] = None,
        write_to_cache: bool = True,
        min_samples: int = 512,
        metric: str = "fid",
    ):
        self.function_idx = default(function_idx, NUM_FUNCTION_CLASSES)
        self.organism_idx = default(organism_idx, NUM_ORGANISM_CLASSES)
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples
        self.write_to_cache = write_to_cache
        self.min_samples = min_samples
        self.metric = metric

        self.val_parquet_fpath = Path(val_parquet_fpath)
        self.cached_pt_dir = Path(cached_pt_dir)

        self.cheap_pipeline = cheap_pipeline
        self.val_parquet = None

        # trigger pipeline that creates the reference embedding
        self.real = self.make_reference_embedding()

    def _make_cheap_pipeline(self):
        from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32

        self.cheap_pipeline = CHEAP_pfam_shorten_2_dim_32()
        self.cheap_pipeline.to(self.device)

    def _load_val_parquet(self):
        self.val_parquet = pd.read_parquet(self.val_parquet_fpath)

    def _features_from_sequence_strings(self, sequences):
        from ..transforms import get_random_sequence_crop_batch

        if self.cheap_pipeline is None:
            self._make_cheap_pipeline()

        sequences = [s[: self.max_seq_len] for s in sequences]
        feats_all = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            with torch.no_grad():
                feats, mask = self.cheap_pipeline(batch)

            # save GPU memory
            feats = feats.detach().cpu()  # (N, L, 32)
            masks = mask.detach().cpu()  # (N, L)

            # calculate masked average
            masks = einops.repeat(masks, "N L -> N L C", C=32)  # (N, L, 32)
            feats = feats * masks  # (N, L, 32)
            feats = feats.sum(dim=1) / masks.sum(dim=1)  # (N, 32)

            # append batch
            feats_all.append(feats)

        return torch.cat(feats_all, dim=0)

    def _load_cached_embedding(self, filecode):
        from safetensors import safe_open

        with safe_open(self.cached_pt_dir / f"{filecode}.pt", "pt") as f:
            x = f.get_tensor("features").numpy()

        assert (
            x.shape[0] >= self.min_samples
        ), f"Need at least {self.min_samples} samples, as configured."

        if len(x) > self.max_eval_samples:
            idxs = np.random.choice(len(x), self.max_eval_samples, replace=False)
            x = x[idxs]
        return x

    def _save_embedding_to_cache(self, feats):
        from safetensors.torch import save_file

        filecode = f"f{self.function_idx}_o{self.organism_idx}"
        save_file({"features": feats}, self.cached_pt_dir / f"{filecode}.pt")

    def _run_pipeline_for_condition(self, save=True):
        if self.val_parquet is None:
            self._load_val_parquet()

        df = self.val_parquet

        # filter by condition
        if is_fn_conditional(self.function_idx):
            df = df[df.GO_idx == self.function_idx]

        if is_org_conditional(self.organism_idx):
            df = df[df.organism_index == self.organism_idx]

        print(f"Found {len(df)} samples for this condition.")

        sequences = df.sequence.values

        if len(sequences) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, as configured."
            )

        np.random.shuffle(sequences)
        sequences = sequences[: self.max_eval_samples]

        x = self._features_from_sequence_strings(sequences)
        if save:
            self._save_embedding_to_cache(x)
        return x

    def make_reference_embedding(self):
        if (self.organism_idx == NUM_ORGANISM_CLASSES) and (
            self.function_idx == NUM_FUNCTION_CLASSES
        ):
            # unconditional
            x = self._load_cached_embedding(filecode="all")
        else:
            filecode = f"f{self.function_idx}_o{self.organism_idx}"
            cached_pt_fpath = self.cached_pt_dir / f"f{filecode}.pt"
            if cached_pt_fpath.exists():
                x = self._load_cached_embedding(filecode)
            else:
                x = self._run_pipeline_for_condition(save=self.write_to_cache)

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        return x

    def run(self, sampled):
        if isinstance(sampled, torch.Tensor):
            sampled = sampled.cpu().numpy()

        N = min(sampled.shape[0], self.max_eval_samples)
        sampled = sampled[:N]
        real = self.real[:N]

        if self.metric == "fid":
            return parmar_fid(sampled, real)
        elif self.metric == "sinkhorn":
            return sinkhorn(sampled, real)
        else:
            raise ValueError(f"Unsupported metric {self.metric}")


def sinkhorn(x, y):
    x, y = npy(x), npy(y)
    import ot

    M = ot.dist(x, y)
    wasserstein_dist = ot.emd2([], [], M)
    return wasserstein_dist
