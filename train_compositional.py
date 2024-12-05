import typing as T
from pathlib import Path

import wandb
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info
from lightning.pytorch.callbacks import LearningRateMonitor

from omegaconf import DictConfig, OmegaConf
import torch
import re

from plaid import constants
from plaid.utils import count_parameters
from plaid.denoisers import FunctionOrganismUDiT, FunctionOrganismDiT
from plaid.diffusion import FunctionOrganismDiffusion


def delete_key(d: dict, key: str = "_target_") -> dict:
    if key in d:
        d.pop(key)
    return d


def maybe_resume_job_from_config(cfg: OmegaConf) -> T.Tuple[dict, str, bool]:
    # maybe use prior job id, else generate new ID
    is_resumed = cfg.resume_from_model_id is not None
    job_id = cfg.resume_from_model_id if is_resumed else wandb.util.generate_id()

    # save config to disk
    config_path = Path(cfg.paths.checkpoint_dir) / job_id / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        rank_zero_info(f"Overriding config from job ID {job_id}!")

    # https://github.com/facebookresearch/xformers/issues/920
    if hasattr(cfg.trainer, "precision"):
        if (
            cfg.trainer.precision == "bf16-mixed"
            and cfg.denoiser._target_ == "plaid.denoisers.FunctionOrganismUDiT"
        ):
            cfg.trainer.update({"precision": "32"})
            print(
                "torch.compile does not yet work for memory-efficient attention.\n"
                "Overriding precision to 32-bit for FunctionOrganismUDiT denoiser."
            )
        else:
            print(
                "Precision is not bf16-mixed or denoiser is not FunctionOrganismUDiT. Skipping override."
            )

    return cfg, job_id, is_resumed


@hydra.main(version_base=None, config_path="configs", config_name="train_compositional")
def train(cfg: DictConfig) -> None:
    import torch

    torch.set_float32_matmul_precision("medium")

    # current run config specifies checkpoint dir, not loaded config.
    ckpt_dir = cfg.paths.checkpoint_dir

    # override all else with what's specified in the config
    cfg, job_id, is_resumed = maybe_resume_job_from_config(cfg)
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    rank_zero_info(OmegaConf.to_yaml(log_cfg))

    ####################################################################################################
    # Data and model setup
    ####################################################################################################

    # dimensions
    input_dim = constants.COMPRESSION_INPUT_DIMENSIONS[cfg.compression_model_id]
    shorten_factor = constants.COMPRESSION_SHORTEN_FACTORS[cfg.compression_model_id]

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if is_resumed:
        denoiser_cls = cfg.denoiser._target_
        denoiser_cfg = delete_key(OmegaConf.to_container(cfg.denoiser), "_target_")
        diffusion_cfg = delete_key(OmegaConf.to_container(cfg.diffusion), "_target_")

        # TODO: make class init based on the _target_ class
        if denoiser_cls == "plaid.denoisers.FunctionOrganismUDiT":
            denoiser = FunctionOrganismUDiT(**denoiser_cfg, input_dim=input_dim)
        elif denoiser_cls == "plaid.denoisers.FunctionOrganismDiT":
            denoiser = FunctionOrganismDiT(**denoiser_cfg, input_dim=input_dim)
        else:
            raise ValueError(f"Unknown denoiser class: {denoiser_cls}")
        denoiser = torch.compile(denoiser)

        # backwards compatibility:
        diffusion = FunctionOrganismDiffusion(**diffusion_cfg, model=denoiser)

    else:
        denoiser = hydra.utils.instantiate(cfg.denoiser, input_dim=input_dim)
        denoiser = torch.compile(denoiser)
        diffusion = hydra.utils.instantiate(cfg.diffusion, model=denoiser)

    # logging details
    trainable_parameters = count_parameters(diffusion, require_grad_only=True)
    total_parameters = count_parameters(diffusion, require_grad_only=False)
    log_cfg["trainable_params_millions"] = trainable_parameters / 1_000_000
    log_cfg["total_params_millions"] = total_parameters / 1_000_000
    log_cfg["shorten_factor"] = shorten_factor
    log_cfg["input_dim"] = input_dim
    logger = hydra.utils.instantiate(cfg.logger, id=job_id)

    ####################################################################################################
    # Callbacks and model saving set-up
    ####################################################################################################

    outdir = Path(ckpt_dir) / job_id
    lr_monitor = LearningRateMonitor()

    # exponential moving average calculations callback
    ema_callback = hydra.utils.instantiate(cfg.callbacks.ema)

    # checkpoint callback (also handles EMA logic, if used)
    checkpoint_callback = hydra.utils.instantiate(
        cfg.callbacks.checkpoint, dirpath=outdir
    )

    callbacks = [lr_monitor, ema_callback, checkpoint_callback]

    # save configs
    config_path = Path(ckpt_dir) / job_id / "config.yaml"
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)
        if rank_zero_only.rank == 0:
            OmegaConf.save(cfg, config_path)

    ####################################################################################################
    # Trainer
    ####################################################################################################

    rank_zero_info("Initializing training...")

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if is_resumed:
        ckpt_fpath = Path(ckpt_dir) / job_id / "last.ckpt"
        assert ckpt_fpath.exists(), f"Checkpoint {ckpt_fpath} not found!"
        rank_zero_info(f"Resuming from checkpoint {ckpt_fpath}")
        trainer.fit(diffusion, datamodule=datamodule, ckpt_path=ckpt_fpath)

    else:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
