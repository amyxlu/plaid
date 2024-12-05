import torch
import abc
import math
import numpy as np


import torch
import abc
import math


######
# The following is adapted from pseudocode in Chen et al.,
# https://arxiv.org/abs/2301.10972
######


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.0)


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


def modified_cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


######
# openai guided diffusion codebase
######


def adm_cosine_schedule(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class BetaScheduler(abc.ABC):
    def __init__(self, start=None, end=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, timesteps):
        raise NotImplementedError


class LinearBetaScheduler(BetaScheduler):
    def __init__(self, start=0.0001, end=0.02):
        self.beta_start = start
        self.beta_end = end

    def __call__(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


class SigmoidBetaScheduler(BetaScheduler):
    def __init__(self, start=-3, end=3, tau=1.0):
        super().__init__()
        self.start = start
        self.end = end
        self.tau = tau

    def __call__(self, timesteps):
        return betas_for_alpha_bar(
            timesteps,
            lambda t: sigmoid_schedule(t, start=self.start, end=self.end, tau=self.tau),
        )


class CosineBetaScheduler(BetaScheduler):
    def __init__(self, start=0, end=1, tau=1):
        super().__init__()
        self.start = start
        self.end = end
        self.tau = tau

    def __call__(self, timesteps):
        return betas_for_alpha_bar(
            timesteps,
            lambda t: modified_cosine_schedule(
                t, start=self.start, end=self.end, tau=self.tau
            ),
        )


class ADMCosineBetaScheduler(BetaScheduler):
    def __init__(self):
        super().__init__()

    def __call__(self, timesteps):
        return betas_for_alpha_bar(timesteps, lambda t: adm_cosine_schedule(t))


def make_beta_scheduler(sched_name, start=None, end=None, tau=None):
    if sched_name == "chen_cosine":
        return CosineBetaScheduler(start=start, end=end, tau=tau)
    elif sched_name == "adm_cosine":
        return ADMCosineBetaScheduler()
    elif sched_name == "sigmoid":
        return SigmoidBetaScheduler(start=start, end=end, tau=tau)
    elif sched_name == "linear":
        return LinearBetaScheduler(start=start, end=end)
    else:
        raise
