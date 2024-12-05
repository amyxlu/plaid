from plaid.pipeline._inverse_fold import InverseFoldPipeline
import hydra
from omegaconf import DictConfig
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir",
        default="/data/lux70/plaid/artifacts/samples/by_length/5j007z42/f2219_o3617/ddim/ive5gowq/inverse_generated/structures",
        type=str,
    )
    parser.add_argument(
        "--output_fasta_path",
        default="/data/lux70/plaid/artifacts/samples/by_length/5j007z42/f2219_o3617/ddim/ive5gowq/phantom_generated/sequences.fasta",
        type=str,
    )
    args = parser.parse_args()
    return args


def run(args):
    """Hydra configurable instantiation, for imports in full pipeline."""
    inverse_fold = InverseFoldPipeline(
        pdb_dir=Path(args.pdb_dir),
        output_fasta_path=Path(args.output_fasta_path),
    )
    inverse_fold.run()


if __name__ == "__main__":
    args = get_args()
    run(args)
