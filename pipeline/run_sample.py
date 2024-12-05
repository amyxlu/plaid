import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs/inference/sample", config_name="sample_latent")
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    print(OmegaConf.to_yaml(cfg))

    sample_latent = hydra.utils.instantiate(cfg)
    sample_latent = sample_latent.run()

    with open(sample_latent.outdir / "sample.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    hydra_run()
