import typing as T
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

from tqdm import trange
import einops
import torch
import numpy as np
import pandas as pd

import wandb

from plaid.diffusion import FunctionOrganismDiffusion
from plaid.diffusion.beta_schedulers import make_beta_scheduler
from plaid.diffusion.dpm_samplers import (
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
    sample_dpmpp_2s_ancestral,
    sample_dpmpp_sde,
    get_sigmas_karras,
    get_sigmas_exponential,
    get_sigmas_polyexponential,
    get_sigmas_vp,
    ModelWrapper,
    DiscreteSchedule,
)
from plaid.datasets import NUM_ORGANISM_CLASSES, NUM_FUNCTION_CLASSES
from plaid.utils import get_pfam_length, round_to_multiple
from plaid.pretrained import create_denoiser_and_cfg_from_id


def check_function_is_uncond(idx):
    return (idx is None) or (idx == NUM_FUNCTION_CLASSES)


def check_organism_is_uncond(idx):
    return (idx is None) or (idx == NUM_ORGANISM_CLASSES)


def default(x, val):
    return x if x is not None else val


PLAID_MODELS = ["PLAID-100M", "PLAID-2B"]


AVAILABLE_SAMPLERS = [
    "ddim",
    "ddpm",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
]


PFAM_HMM_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "../../../assets", "Pfam-A.hmm.dat"
)


PFAM_GO_INDEX_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "../../../assets", "go_index.csv"
)


class SampleLatent:
    def __init__(
        self,
        # model setup
        model_id: str = "PLAID-2B",  # "PLAID-100M", "PLAID-2B"
        use_compile: bool = False,
        # sampling setup
        organism_idx: int = NUM_ORGANISM_CLASSES,
        function_idx: int = NUM_FUNCTION_CLASSES,
        cond_scale: float = 3,
        num_samples: int = 4,
        beta_scheduler_name: T.Optional[str] = None,
        beta_scheduler_start: T.Optional[int] = None,
        beta_scheduler_end: T.Optional[int] = None,
        beta_scheduler_tau: T.Optional[int] = None,
        sampling_timesteps: int = 1000,
        batch_size: int = -1,
        length: T.Optional[
            int
        ] = 32,  # the final length, after decoding back to structure/sequence, is twice this value
        return_all_timesteps: bool = False,
        full_pfam_hmm_file_path: str = PFAM_HMM_FILE_PATH,
        go_to_representative_pfam_path: str = PFAM_GO_INDEX_FILE_PATH,
        # output setup
        output_root_dir: str = "./samples/",
        use_condition_output_suffix: bool = False,
        use_uid_output_suffix: bool = False,
        # scheduler
        sample_scheduler: str = "ddim",  # ["ddim", ""ddpm"]
        # dpm parameters
        sigma_min: float = 1e-2,
        sigma_max: float = 160,
        # motif
        motif_seq: T.Optional[str] = None,
        motif_start_pos: T.Optional[int] = None,
        cheap_encoder: T.Optional[torch.nn.Module] = None,
    ):
        ############################################################
        # Set up
        ############################################################

        assert (
            sample_scheduler in AVAILABLE_SAMPLERS
        ), f"Invalid sample scheduler: {sample_scheduler}. Must be one of {AVAILABLE_SAMPLERS}."
        
        assert model_id in PLAID_MODELS, f"Invalid model ID: {model_id}. Must be one of {PLAID_MODELS}."

        self.model_id = str(model_id)
        self.organism_idx = int(organism_idx)
        self.function_idx = int(function_idx)
        self.cond_scale = float(cond_scale)
        self.num_samples = int(num_samples)
        self.return_all_timesteps = bool(return_all_timesteps)
        self.output_root_dir = Path(output_root_dir)
        self.sample_scheduler = str(sample_scheduler)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.use_compile = bool(use_compile)
        self.uid = wandb.util.generate_id()

        # default to cuda
        self.device = torch.device("cuda")

        # if no batch size is provided, sample all at once
        self.batch_size = batch_size if batch_size > 0 else num_samples

        ############################################################
        # Load pretrained denoiser
        ############################################################
        self.denoiser, self.cfg = create_denoiser_and_cfg_from_id(model_id)

        ############################################################
        # Override sampling hyperparameters
        ############################################################

        # Override sampling timesteps if provided; otherwise, use what was used during training.
        self.sampling_timesteps = default(
            sampling_timesteps, self.cfg.diffusion.timesteps
        )

        self.beta_scheduler_name = default(
            beta_scheduler_name, self.cfg.diffusion.beta_scheduler_name
        )
        self.beta_scheduler_start = default(
            beta_scheduler_start, self.cfg.diffusion.beta_scheduler_start
        )
        self.beta_scheduler_end = default(
            beta_scheduler_end, self.cfg.diffusion.beta_scheduler_end
        )
        self.beta_scheduler_tau = default(
            beta_scheduler_tau, self.cfg.diffusion.beta_scheduler_tau
        )

        # create the diffusion object
        self.diffusion = self._create_diffusion()
        self.diffusion = self.diffusion.to(self.device)

        ############################################################
        # Set up auto length selection and output paths
        ############################################################
        
        # if length is not specified and we are using conditional generation, automatically choose length
        self.full_pfam_hmm_file = None
        self.go_to_representative_pfam_df = None
        self.full_pfam_hmm_file_path = full_pfam_hmm_file_path
        self.go_to_representative_pfam_path = go_to_representative_pfam_path

        if (length is None) or (length == "None"):
            assert (full_pfam_hmm_file_path is not None) and (
                go_to_representative_pfam_path is not None
            ), "If length is not provided, `full_pfam_hmm_file_path` and `go_to_representative_pfam_path` must be provided."

            assert not check_function_is_uncond(
                function_idx
            ), "If length is not provided, function_idx cannot be unconditional, such that we can sample a pfam ID."

            length = self._auto_choose_length()

        else:
            if length % 4 != 0:
                print(f"WARNING: length {length} is not a multiple of 4.")
                length = round_to_multiple(length, 4)
                print(f"Rounding to the nearest multiple of 4 as {length}.")

        self.length = int(length)

        # set up output paths
        self.cond_code = f"f{self.function_idx}_o{self.organism_idx}_l{int(self.length)}_s{int(self.cond_scale)}"
        self.outdir = self._setup_paths(
            output_root_dir, use_condition_output_suffix, use_uid_output_suffix
        )

        ############################################################
        # Set up motif scaffolding
        ############################################################

        if motif_start_pos is not None:
            self.motif_start_pos = motif_start_pos
            self.motif_seq = motif_seq
            self.cheap_encoder = cheap_encoder
            if self.cheap_encoder is None:
                self._create_cheap_encoder()

            motif_x, _ = self.cheap_encoder(motif_seq)
            implicit_length = motif_x.shape[1]
            motif_x = einops.repeat(motif_x, "1 l c -> n l c", n=self.batch_size)
            self.motif_x = motif_x.to(self.device)
            self.motif_pos = (motif_start_pos, motif_start_pos + implicit_length)

        else:
            self.motif_pos = None
            self.motif_x = None

    def _auto_choose_length(self):
        if self.full_pfam_hmm_file is None:
            with open(self.full_pfam_hmm_file_path, "r") as f:
                self.full_pfam_hmm_file = f.read()

        if self.go_to_representative_pfam_df is None:
            self.go_to_representative_pfam_df = pd.read_csv(
                self.go_to_representative_pfam_path
            )

        def _go_idx_to_representative_pfam():
            df = self.go_to_representative_pfam_df
            return df[df.GO_idx == self.function_idx].pfam_id.values[0]

        pfam_id = _go_idx_to_representative_pfam()
        length = get_pfam_length(pfam_id, self.full_pfam_hmm_file)

        length = round_to_multiple(length / 2, 4)
        print(
            f"Auto-choosing length {length * 2} (implicit length in GPU memory: {length})."
        )
        return length

    def _setup_paths(
        self,
        output_root_dir,
        use_condition_output_suffix=False,
        use_uid_output_suffix=False,
    ):
        # set up paths
        outdir = Path(output_root_dir)
        if use_condition_output_suffix:
            outdir = outdir / self.cond_code

        if use_uid_output_suffix:
            outdir = outdir / self.uid

        if not outdir.exists():
            outdir.mkdir(parents=True)

        return outdir

    def _create_cheap_encoder(self):
        # only really needed if motif is provided.
        from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32

        self.cheap_encoder = CHEAP_pfam_shorten_2_dim_32()
        self.cheap_encoder = self.cheap_encoder.to(self.device)

    def _create_diffusion(self):
        diffusion_kwargs = self.cfg.diffusion
        diffusion_kwargs.pop("_target_")
        diffusion_kwargs["sampling_timesteps"] = self.sampling_timesteps
        diffusion_kwargs["beta_scheduler_name"] = self.beta_scheduler_name
        diffusion_kwargs["beta_scheduler_start"] = self.beta_scheduler_start
        diffusion_kwargs["beta_scheduler_end"] = self.beta_scheduler_end
        diffusion_kwargs["beta_scheduler_tau"] = self.beta_scheduler_tau
        diffusion = FunctionOrganismDiffusion(model=self.denoiser, **diffusion_kwargs)
        diffusion = diffusion.to(self.device)
        return diffusion

    def sample(self):
        N, L, C = self.batch_size, self.length, self.diffusion.model.input_dim
        shape = (N, L, C)

        if self.sample_scheduler == "ddim":
            sample_loop_fn = self.diffusion.ddim_sample_loop
        elif self.sample_scheduler == "ddpm":
            sample_loop_fn = self.diffusion.p_sample_loop
        else:
            self.dpm_setup()
            return self.dpm_sample()

        # assuming no gradient-guided diffusion:
        with torch.no_grad():
            sampled_latent = sample_loop_fn(
                shape=shape,
                organism_idx=self.organism_idx,
                function_idx=self.function_idx,
                return_all_timesteps=self.return_all_timesteps,
                cond_scale=self.cond_scale,
                motif_pos=self.motif_pos,
                motif_x=self.motif_x,
            )
        return sampled_latent

    def dpm_setup(self, **kwargs):
        self.sample_fn = globals()[f"sample_{self.sample_scheduler}"]
        self.sigmas = get_sigmas_karras(
            self.sampling_timesteps,
            self.sigma_min,
            self.sigma_max,
            rho=7.0,
            device=self.device,
        )
        discrete_schedule = DiscreteSchedule(self.sigmas, quantize=True)

        self.extra_args = {
            "function_idx": self.function_idx,
            "organism_idx": self.function_idx,
            "mask": None,
            "cond_scale": self.cond_scale,
            "rescaled_phi": 0.7,
        }

        self.model = ModelWrapper(self.diffusion, discrete_schedule)

    def dpm_sample(self):
        N, L, C = self.batch_size, self.length, self.diffusion.model.input_dim
        shape = (N, L, C)
        x = torch.randn(shape, device=self.device)
        with torch.no_grad():
            return self.sample_fn(
                self.model,
                x,
                self.sigmas,
                extra_args=self.extra_args,
                return_intermediates=self.return_all_timesteps,
            )

    def run(self):
        num_samples = max(self.num_samples, self.batch_size)
        all_sampled = []
        cur_n_sampled = 0

        start = time.time()
        for _ in trange(0, num_samples, self.batch_size, desc="Sampling batches"):
            sampled_latent = self.sample()
            all_sampled.append(sampled_latent)
            cur_n_sampled += self.batch_size
        end = time.time()

        print(f"Sampling took {end-start:.2f} seconds.")

        all_sampled = torch.cat(all_sampled, dim=0)
        all_sampled = all_sampled.cpu().numpy()
        all_sampled = all_sampled.astype(
            np.float16
        )  # this is ok because values are [-1,1]

        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        outpath = self.outdir / "latent.npz"

        np.savez(outpath, samples=all_sampled)
        print(f"Saved .npz file to {outpath} [shape={all_sampled.shape}].")

        with open(outpath.parent / "sample.log", "w") as f:
            self.sampling_time = end - start
            f.write("Sampling time: {:.2f} seconds.\n".format(end - start))

        self.outpath = outpath
        self.x = all_sampled

        return self
