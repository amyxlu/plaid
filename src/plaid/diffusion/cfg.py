"""
Main PyTorch lightning module for our diffusion model.
Roughly follows Dhariwal et al., with min-SNR weighting.

Adapted from:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py
"""

import typing as T
from random import random
from functools import partial
from collections import namedtuple

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import lightning as L

from .beta_schedulers import make_beta_scheduler
from ..utils import get_lr_scheduler
from ..losses import masked_mse_loss
from ..denoisers import DenoiserKwargs
from ..datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES
from ..typed import ArrayLike


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class FunctionOrganismDiffusion(L.LightningModule):
    def __init__(
        self,
        model: T.Optional[torch.nn.Module] = None,  # denoiser
        *,
        # beta scheduler
        beta_scheduler_name="adm_cosine",
        beta_scheduler_start=None,
        beta_scheduler_end=None,
        beta_scheduler_tau=None,
        # additional diffusion parameters
        x_downscale_factor: float = 1.0,
        timesteps=1000,
        objective="pred_v",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        x_clip_val: T.Optional[float] = 1.0,
        # classifier free guidance dropout probabilities
        function_y_cond_drop_prob: float = 0.3,
        organism_y_cond_drop_prob: float = 0.3,
        # sampling
        sampling_timesteps=1000,  # None,
        ddim_sampling_eta=0.0,
        # optimization
        ema_decay: T.Optional[float] = 0.9999,
        lr=1e-4,
        lr_adam_betas=(0.9, 0.999),
        lr_sched_type: str = "schedule_free",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        use_old_ema_module: bool = False,
    ):
        super().__init__()

        self.model = model
        if model is not None:
            self.self_condition = self.model.use_self_conditioning
        self.function_y_cond_drop_prob = function_y_cond_drop_prob
        self.organism_y_cond_drop_prob = organism_y_cond_drop_prob

        self.x_downscale_factor = x_downscale_factor
        self.x_clip_val = x_clip_val
        self.clip_fn = partial(
            torch.clamp, min=self.x_clip_val * -1, max=self.x_clip_val
        )

        self.lr = lr
        self.lr_adam_betas = lr_adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        # if (ema_decay is not None) and use_old_ema_module:
        #     self.ema_model = AveragedModel(
        #         self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
        #     )
        #     self.ema_model.eval()
        #     self.ema_model.requires_grad_(False)
        self.ema_model = None

        ###########################################################
        # Diffusion helper functions
        ###########################################################

        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"
        self.beta_scheduler = make_beta_scheduler(
            beta_scheduler_name,
            beta_scheduler_start,
            beta_scheduler_end,
            beta_scheduler_tau,
        )
        betas = self.beta_scheduler(timesteps)
        betas = torch.tensor(betas, dtype=torch.float64)
        assert (betas >= 0).all() and (betas <= 1).all()
        assert len(betas.shape) == 1

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)
        register_buffer("snr", snr)
        self.save_hyperparameters(ignore=["model"])

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    ###########################################################
    # Reverse diffusion sampling
    ###########################################################

    def model_predictions(
        self,
        denoiser_kwargs: DenoiserKwargs,
        cond_scale: float,
        rescaled_phi: float = 0.7,
    ):
        """Calls the forward_with_cond_scale method of the model to get the model output with conditioning (no label dropout)."""
        model_output = self.model.forward_with_cond_scale(
            denoiser_kwargs, cond_scale, rescaled_phi
        )

        x, t = denoiser_kwargs.x, denoiser_kwargs.t

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = self.clip_fn(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = self.clip_fn(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = self.clip_fn(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        denoiser_kwargs: DenoiserKwargs,
        cond_scale: float,
        rescaled_phi: float = 0.7,
        clip_denoised=True,
    ):
        x, t = denoiser_kwargs.x, denoiser_kwargs.t
        preds = self.model_predictions(denoiser_kwargs, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start = self.clip_fn(x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def sample_step_inputs_to_kwargs(
        self,
        x: torch.Tensor,
        t: int,
        x_self_cond: T.Optional[torch.Tensor] = None,
        function_idx: T.Optional[int] = None,
        organism_idx: T.Optional[int] = None,
        mask: T.Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """
        Given the x, t, function_idx, organism_idx, mask, and x_self_cond at a given
        sample step, create the DenoiserKwargs input for the model forward function.
        this keeps code more readable and avoids repetition.
        """

        # If unspecified, then use unconditional sampling using the dummy index,
        # which is equal to the number of classes since class indices are 0-indexed.
        function_idx = default(function_idx, NUM_FUNCTION_CLASSES)
        organism_idx = default(organism_idx, NUM_ORGANISM_CLASSES)

        # batch integers into tensors
        b, *_, device = *x.shape, x.device

        if isinstance(t, int):
            t = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        if isinstance(function_idx, int):
            function_idx = torch.full(
                (x.shape[0],), function_idx, device=x.device, dtype=torch.long
            )
        if isinstance(organism_idx, int):
            organism_idx = torch.full(
                (x.shape[0],), organism_idx, device=x.device, dtype=torch.long
            )

        return DenoiserKwargs(
            x=x,
            t=t,
            function_y=function_idx,
            organism_y=organism_idx,
            mask=mask,
            x_self_cond=x_self_cond,
        )

    def continuous_time_noise_predictions(
        self,
        x: torch.Tensor,
        t_continuous: int,
        total_T: int,
        x_self_cond: T.Optional[torch.Tensor] = None,
        function_idx: T.Optional[int] = None,
        organism_idx: T.Optional[int] = None,
        mask: T.Optional[torch.Tensor] = None,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
    ):
        """
        Wrapper for model forward for continuous time sampling steps, e.g. for DPM++ solvers.
        """
        discrete_t = (t_continuous - 1.0 / total_T) * 1000.0
        denoiser_kwargs = self.sample_step_inputs_to_kwargs(
            x=x,
            t=discrete_t,
            x_self_cond=x_self_cond,
            function_idx=function_idx,
            organism_idx=organism_idx,
            mask=mask,
        )
        preds = self.model_predictions(denoiser_kwargs, cond_scale, rescaled_phi)
        return preds.pred_noise  # needed for DPM solvers

    @torch.inference_mode()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        function_idx: T.Optional[int] = None,
        organism_idx: T.Optional[int] = None,
        mask: T.Optional[torch.Tensor] = None,
        x_self_cond: T.Optional[torch.Tensor] = None,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_denoised=True,
        motif_x: T.Optional[torch.Tensor] = None,
        motif_pos: T.Optional[T.Tuple[int, int]] = None,
    ):
        if motif_pos is not None:
            assert motif_x is not None
            start, end = motif_pos
            if end > x.shape[1]:
                raise ValueError("Motif position is out of bounds")

            x[:, start:end, :] = motif_x
            if x_self_cond is not None:
                x_self_cond[:, start:end, :] = motif_x

        denoiser_kwargs = self.sample_step_inputs_to_kwargs(
            x, t, function_idx, organism_idx, mask, x_self_cond
        )

        # calculate p_mean_variance and perform reverse diffusion step
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            denoiser_kwargs=denoiser_kwargs,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: ArrayLike,
        function_idx: T.Optional[int] = None,
        organism_idx: T.Optional[int] = None,
        mask: T.Optional[torch.Tensor] = None,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_denoised=True,
        return_all_timesteps=False,
        motif_x: T.Optional[torch.Tensor] = None,
        motif_pos: T.Optional[T.Tuple[int, int]] = None,
    ):
        batch, device = shape[0], self.betas.device

        cur_sample = torch.randn(shape, device=device)
        samples = [cur_sample]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc="sampling loop time step",
            total=self.sampling_timesteps,
        ):
            x_self_cond = x_start if self.self_condition else None
            cur_sample, x_start = self.p_sample(
                x=cur_sample,
                t=t,
                function_idx=function_idx,
                organism_idx=organism_idx,
                mask=mask,
                x_self_cond=x_self_cond,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                clip_denoised=clip_denoised,
                motif_x=motif_x,
                motif_pos=motif_pos,
            )
            samples.append(cur_sample)

        # if return all timesteps, return a tensor of shape (B, T, L, C), otherwise return (B, L, C)
        ret = cur_sample if not return_all_timesteps else torch.stack(samples, dim=1)
        return ret

    @torch.inference_mode()
    def ddim_sample_loop(
        self,
        shape: ArrayLike,
        function_idx: T.Optional[int] = None,
        organism_idx: T.Optional[int] = None,
        mask: T.Optional[torch.Tensor] = None,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        clip_denoised: bool = True,
        return_all_timesteps: bool = False,
        motif_x: T.Optional[torch.Tensor] = None,
        motif_pos: T.Optional[T.Tuple[int, int]] = None,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        cur_sample = torch.randn(shape, device=device)
        samples = [cur_sample]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            batched_function_idx = torch.full(
                (batch,), function_idx, device=device, dtype=torch.long
            )
            batched_organism_idx = torch.full(
                (batch,), organism_idx, device=device, dtype=torch.long
            )
            x_self_cond = x_start if self.self_condition else None

            if motif_pos is not None:
                assert motif_x is not None
                start, end = motif_pos
                if end > cur_sample.shape[1]:
                    raise ValueError("Motif position is out of bounds")

                cur_sample[:, start:end, :] = motif_x
                if x_self_cond is not None:
                    x_self_cond[:, start:end, :] = motif_x

            denoiser_kwargs = DenoiserKwargs(
                x=cur_sample,
                t=time_cond,
                function_y=batched_function_idx,
                organism_y=batched_organism_idx,
                mask=mask,
                x_self_cond=x_self_cond,
            )
            preds = self.model_predictions(denoiser_kwargs, cond_scale, rescaled_phi)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if clip_denoised:
                x_start = self.clip_fn(x_start)

            samples.append(cur_sample)

            if time_next < 0:
                cur_sample = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(cur_sample)

            cur_sample = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # if return all timesteps, return a tensor of shape (B, T, L, C), otherwise return (B, L, C)
        ret = cur_sample if not return_all_timesteps else torch.stack(samples, dim=1)
        return ret

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int,
        max_length: int,
        function_idx: T.Optional[T.Union[int, torch.Tensor]] = None,
        organism_idx: T.Optional[T.Union[int, torch.Tensor]] = None,
        mask: T.Optional[torch.Tensor] = None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        return_all_timesteps=False,
        clip_denoised=True,
        motif_x: T.Optional[torch.Tensor] = None,
        motif_pos: T.Optional[T.Tuple[int, int]] = None,
    ):
        """
        Sample from the diffusion model

        :param n_samples: the number of samples to generate.
        :param max_length: the maximum length of the samples.
        :param function_idx: the class index for the function conditioning; use None for unconditional. Can be either a single index or a tensor of different classes.
        :param organism_idx: the class index for the organism conditioning; use None for unconditional. Can be either a single index or a tensor of different classes.
        :param mask: only necessary if wishing to generate samples with different lengths, in which case the mask should be True for padded regions.
        :param cond_scale: the conditioning strength in classifier free guidance (https://arxiv.org/abs/2207.12598).
        :param rescaled_phi: the rescaled phi value as proposed in (https://arxiv.org/abs/2305.08891)
        :param return_all_timesteps: whether to return all timesteps as tensor of (B, T, L, C) or just the final sample (B, L, C).
        :param clip_denoised: whether to clip the denoised image to the range [-1, 1] (or other specified self.x_clip_val).
        :return: the generated samples.
        """
        for idxs in (function_idx, organism_idx):
            if isinstance(idxs, torch.Tensor):
                assert (
                    len(idxs) == n_samples
                ), f"Received array of class labels with {len(idxs)} samples but n_samples is set to {n_samples}."

        # If unspecified, then use unconditional sampling using the dummy index
        function_idx = default(function_idx, NUM_FUNCTION_CLASSES)
        organism_idx = default(organism_idx, NUM_ORGANISM_CLASSES)

        # If the class indices are not tensors, then broadcast them to the number of samples
        if isinstance(function_idx, int):
            function_idx = torch.full(
                (n_samples,), function_idx, device=self.betas.device, dtype=torch.long
            )

        if isinstance(organism_idx, int):
            organism_idx = torch.full(
                (n_samples,), organism_idx, device=self.betas.device, dtype=torch.long
            )

        shape = (n_samples, max_length, self.model.input_dim)
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_loop
        )
        return sample_fn(
            shape,
            function_idx,
            organism_idx,
            mask,
            cond_scale,
            rescaled_phi,
            clip_denoised,
            return_all_timesteps,
            motif_x,
            motif_pos,
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, classes, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img, _ = self.p_sample(img, i, classes)

        return img

    ###########################################################
    # Forward diffusion training
    ###########################################################

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def run_step(
        self,
        batch: T.Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        noise: T.Optional[torch.Tensor] = None,
    ):
        x_start, mask, go_idx, organism_idx, _, _, _ = batch
        noise = default(noise, lambda: torch.randn_like(x_start))

        # potentially suppress signal-to-noise ratio
        x_start *= self.x_downscale_factor

        # sample timesteps
        B, L, _ = x_start.shape
        t = torch.randint(0, self.num_timesteps, (B,))
        t = t.long().to(self.device)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        denoiser_kwargs = DenoiserKwargs(
            x=x,
            t=t,
            function_y=go_idx,
            organism_y=organism_idx,
            mask=mask,
            x_self_cond=None,
        )

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # for self-conditioning, ignore class indices when making the initial guess

        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    denoiser_kwargs, cond_scale=0.0, rescaled_phi=0.0
                ).pred_x_start
                x_self_cond.detach_()

            denoiser_kwargs = denoiser_kwargs._replace(x_self_cond=x_self_cond)

        # Denoise with random label dropout for classifier free guidance
        model_out = self.model.forward_with_cond_drop(
            denoiser_kwargs,
            function_y_cond_drop_prob=self.function_y_cond_drop_prob,
            organism_y_cond_drop_prob=self.organism_y_cond_drop_prob,
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Used masked MSE loss to get per-batch result
        loss = masked_mse_loss(model_out, target, mask, reduce="batch")

        # apply min-SNR weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss = loss.mean()

        log_dict = {"loss": loss}

        return loss, log_dict

    ###########################################################
    # Lightning boiler plate
    ###########################################################

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.run_step(batch)
        N = batch[0].shape[0]
        self.log_dict(
            {f"train/{k}": v for k, v in log_dict.items()},
            on_step=True,
            on_epoch=False,
            batch_size=N,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.run_step(batch)
        N = batch[0].shape[0]
        self.log_dict(
            {f"val/{k}": v for k, v in log_dict.items()},
            on_step=False,
            on_epoch=True,
            batch_size=N,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        parameters = self.model.parameters()

        if self.lr_sched_type == "schedule_free":
            # https://arxiv.org/abs/2405.15682
            # optimizer = AdamWScheduleFree(
            #     parameters,
            #     lr=self.lr,
            #     warmup_steps=self.lr_num_warmup_steps,
            #     betas=self.lr_adam_betas,
            # )
            # return optimizer
            raise ValueError("Schedule-free AdamW is deprecated.")

        else:
            optimizer = torch.optim.AdamW(
                parameters, lr=self.lr, betas=self.lr_adam_betas
            )
            scheduler = get_lr_scheduler(
                optimizer=optimizer,
                sched_type=self.lr_sched_type,
                num_warmup_steps=self.lr_num_warmup_steps,
                num_training_steps=self.lr_num_training_steps,
                num_cycles=self.lr_num_cycles,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # To handle schedule-free AdamW

    # def on_train_start(self):
    #     optimizer = self.optimizers().optimizer
    #     if isinstance(optimizer, AdamWScheduleFree):
    #         optimizer.train()

    # def on_val_start(self):
    #     optimizer = self.optimizers().optimizer
    #     if isinstance(self.optimizer(), AdamWScheduleFree):
    #         optimizer.eval()

    # def on_test_start(self):
    #     optimizer = self.optimizers().optimizer
    #     if isinstance(self.optimizers(), AdamWScheduleFree):
    #         optimizer.eval()

    # def on_predict_start(self):
    #     optimizer = self.optimizers().optimizer
    #     if isinstance(self.optimizers(), AdamWScheduleFree):
    #         optimizer.eval()
