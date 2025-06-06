import torch
import os
import os.path
import warnings
import glob

import lightning.pytorch as pl

from pathlib import Path
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_warn, rank_zero_info, rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

from typing import Any, Dict, List, Optional


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps: Apply EMA every n global steps.
        start_step: Start applying EMA from ``start_step`` global step onwards.
        save_ema_weights_in_callback_state: Enable saving EMA weights in callback state.
        evaluate_ema_weights_instead: Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        super().__init__()
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        rank_zero_info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [
                p.detach().clone() for p in pl_module.state_dict().values()
            ]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [
            p.to(pl_module.device) for p in self._ema_model_weights
        ]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        return self.apply_ema(pl_module)

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        for orig_weight, ema_weight in zip(
            list(pl_module.state_dict().values()), self._ema_model_weights
        ):
            if torch.is_floating_point(ema_weight.data) and torch.is_floating_point(
                orig_weight.data
            ):
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return (
            step != self._cur_step
            and step >= self.start_step
            and step % self.apply_ema_every_n_steps == 0
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        # when loading within apps such as NeMo, EMA weights will be loaded by the experiment manager separately
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION

            # if the resuming checkpoint is the EMA weights, use them for the EMA copy as well.
            if (str(trainer.ckpt_path).endswith(f"-EMA{ext}")) or (
                str(trainer.ckpt_path).endswith(f"last.ckpt")
            ):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return

            # otherwise, look to see if there are EMA weights saved in the checkpoint directory
            ema_path = str(trainer.ckpt_path).replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, weights_only=False, map_location=torch.device("cpu"))
                self._ema_model_weights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                rank_zero_info(
                    "EMA weights have been loaded successfully. Continuing training with saved EMA weights."
                )
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        self._weights_buffer = [
            p.detach().clone().to("cpu") for p in pl_module.state_dict().values()
        ]
        new_state_dict = {
            k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)
        }
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


class EMAModelCheckpoint(ModelCheckpoint):
    """
    Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the model as well.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, **kwargs):
        # call the parent class constructor with the provided kwargs
        super().__init__(**kwargs)

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            # save EMA copy of the model as well
            ema_callback.replace_model_weights(trainer.lightning_module)
            filepath = self._ema_format_filepath(filepath)
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
            super()._save_checkpoint(trainer, filepath)
            if rank_zero_only.rank == 0:
                try:
                    self._delete_prev_emas(filepath)
                except:
                    pass
            ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _delete_prev_emas(self, filepath) -> None:
        ckpt_dir = Path(filepath).parent
        ema_paths = glob.glob(str(ckpt_dir / f"*-EMA{self.FILE_EXTENSION}"))
        for path in ema_paths:
            if path != filepath:
                os.remove(path)
