import os
from pathlib import Path
from plaid.pipeline._decode import DecodeLatent
import hydra
from omegaconf import DictConfig, OmegaConf
from plaid.esmfold import esmfold_v1


@hydra.main(config_path="../configs/inference/decode", config_name="default")
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running"""
    npz_path = Path(cfg.npz_path)
    output_root_dir = npz_path.parent
    esmfold = esmfold_v1()
    decode_latent = hydra.utils.instantiate(
        cfg, output_root_dir=output_root_dir, esmfold=esmfold
    )
    decode_latent.run()


if __name__ == "__main__":
    hydra_run()
