from typing import Tuple

import torch
from huggingface_hub import hf_hub_download

from .denoisers import FunctionOrganismUDiT
from .diffusion import FunctionOrganismDiffusion
from .constants import CHECKPOINT_DIR_PATH

from omegaconf import OmegaConf


def _load_state_dict(model_id) -> dict:
    """If not already cached, this will download the weights from the given URL and return the state dict."""
    model_path = hf_hub_download(
        repo_id="amyxlu/plaid",
        subfolder=model_id,
        filename="last.ckpt",
        cache_dir=CHECKPOINT_DIR_PATH,
    )
    return torch.load(model_path)


def _load_config(model_id) -> OmegaConf:
    """If not already cached, this will download the config from the given URL and return the config."""
    config_path = hf_hub_download(
        repo_id="amyxlu/plaid",
        subfolder=model_id,
        filename="config.yaml",
        cache_dir=CHECKPOINT_DIR_PATH,
    )
    return OmegaConf.load(config_path)


def create_denoiser_and_cfg_from_id(
    model_id: str = "PLAID-2B", inference_mode: bool = True
) -> Tuple[FunctionOrganismDiffusion, OmegaConf]:
    """
    Load a pretrained PLAID denoiser weights and the config used to train the diffusion model from HuggingFace.

    Args:
        model_id: The ID of the pretrained model to load. Options: "PLAID-2B", "PLAID-100M".
    """

    assert model_id in [
        "PLAID-2B",
        "PLAID-100M",
    ], f"Invalid model ID: {model_id}. Expected one of: ['PLAID-2B', 'PLAID-100M']."
    ckpt = _load_state_dict(model_id)
    cfg = _load_config(model_id)

    mod_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k[:16] == "model._orig_mod.":
            mod_state_dict[k[16:]] = v

    denoiser_kwargs = cfg.denoiser
    _ = denoiser_kwargs.pop("_target_")

    denoiser = FunctionOrganismUDiT(**denoiser_kwargs)
    denoiser.load_state_dict(mod_state_dict)

    if inference_mode:
        denoiser.eval().requires_grad_(False)

    return denoiser, cfg


def PLAID_2B() -> Tuple[FunctionOrganismDiffusion, OmegaConf]:
    return create_denoiser_and_cfg_from_id("PLAID-2B")


def PLAID_100M() -> Tuple[FunctionOrganismDiffusion, OmegaConf]:
    return create_denoiser_and_cfg_from_id("PLAID-100M")
