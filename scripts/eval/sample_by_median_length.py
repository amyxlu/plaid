"""
Runs pipeline for sampling, but is based on the dataframe and average lengths.
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
import torch
import pandas as pd

from plaid.utils import round_to_multiple, get_pfam_length
from plaid.pipeline import SampleLatent
from plaid.datasets import NUM_ORGANISM_CLASSES, NUM_FUNCTION_CLASSES, HUMAN_IDX
from plaid.esmfold import esmfold_v1
from plaid.typed import PathLike


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Sample latent space.")
    parser.add_argument("--function_idx", type=int, default=None)
    parser.add_argument("--organism_idx", type=int, default=None)
    parser.add_argument(
        "--loop_over",
        required=True,
        choices=["function", "organism"],
        default="function",
    )
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument(
        "--outdir", default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist"
    )
    args = parser.parse_args()
    return args


def check_function_is_uncond(idx):
    return (idx is None) or (idx == NUM_FUNCTION_CLASSES)


def check_organism_is_uncond(idx):
    return (idx is None) or (idx == NUM_ORGANISM_CLASSES)


def run(df, function_idx, organism_idx, n_samples):
    print(f"Original dataset size: {df.shape[0]}")

    if not check_function_is_uncond(function_idx):
        df = df[df.GO_idx == function_idx]

    print(f"After filtering by function: {df.shape[0]}")

    if not check_organism_is_uncond(organism_idx):
        df = df[df.organism_index == organism_idx]

    print(f"After filtering by organism: {df.shape[0]}")

    median_len = int(df.seq_len.median())
    sample_len = round_to_multiple(median_len / 2, multiple=4)

    # =============================================================================
    # Sample
    # =============================================================================

    # override configs
    sample_cfg = OmegaConf.load(
        "/homefs/home/lux70/code/plaid/configs/pipeline/sample/sample_latent.yaml"
    )
    sample_cfg.function_idx = int(function_idx)
    # sample_cfg.organism_idx = int(organism_idx)
    sample_cfg.organism_idx = HUMAN_IDX  # int(organism_idx)
    sample_cfg.length = int(sample_len)
    sample_cfg.num_samples = int(n_samples)
    sample_cfg.output_root_dir = args.outdir

    print(OmegaConf.to_yaml(sample_cfg))

    # instantiate and run
    sample_latent = hydra.utils.instantiate(sample_cfg)
    sample_latent = sample_latent.run()  # returns npz path

    # save the config
    with open(sample_latent.outdir / "sample.yaml", "w") as f:
        OmegaConf.save(sample_cfg, f)

    # =============================================================================
    # Decode
    # =============================================================================

    # set up some values
    outdir = sample_latent.outdir
    x = sample_latent.x

    decode_cfg = OmegaConf.load(
        "/homefs/home/lux70/code/plaid/configs/pipeline/decode/default.yaml"
    )
    decode_cfg.npz_path = str(outdir / "latent.npz")
    decode_cfg.output_root_dir = str(outdir / "generated")
    print(OmegaConf.to_yaml(decode_cfg))

    # load esmfold
    esmfold = esmfold_v1()
    esmfold.eval().requires_grad_(False)
    esmfold.cuda()

    decode_latent = hydra.utils.instantiate(decode_cfg, esmfold=esmfold)
    seq_strs, pdb_paths = decode_latent.run()

    with open(outdir / "generated" / "decode_config.yaml", "w") as f:
        OmegaConf.save(decode_cfg, f)

    assert len(seq_strs) == len(pdb_paths) == x.shape[0]


def main(args):
    # =============================================================================
    # Set up function and organism conditioning
    # =============================================================================

    args = parse_args()
    function_idx = args.function_idx
    organism_idx = args.organism_idx

    if function_idx is None:
        function_idx = NUM_FUNCTION_CLASSES

    if organism_idx is None:
        organism_idx = NUM_ORGANISM_CLASSES
        # organism_idx = HUMAN_IDX

    assert check_function_is_uncond(function_idx) or check_organism_is_uncond(
        organism_idx
    )

    # =============================================================================
    # Load and filter dataframes that we'll use to determine the median length
    # =============================================================================

    # organism_df = pd.read_csv("/data/lux70/data/pfam/organism_hierarchy.csv")
    # go_df = pd.read_csv("/data/lux70/data/pfam/go_index.csv")
    df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")
    df = df.sort_values(by="GO_counts")

    unique_functions = df.GO_idx.unique()
    unique_organisms = df.organism_index.unique()

    if args.start_idx is not None:
        unique_functions = unique_functions[int(args.start_idx) :]
        unique_organisms = unique_organisms[: int(args.end_idx)]

    if args.loop_over == "function":
        for function_idx in unique_functions:
            run(df, function_idx, organism_idx, args.n_samples)

    elif args.loop_over == "organism":
        for organism_idx in unique_organisms:
            run(df, function_idx, organism_idx, args.n_samples)


if __name__ == "__main__":
    args = parse_args()
    main(args)
