from pathlib import Path
import os

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import pandas as pd

from plaid.typed import PathLike
from plaid.esmfold import esmfold_v1
from plaid.evaluation import (
    RITAPerplexity,
    foldseek_easycluster,
    foldseek_easysearch,
    mmseqs_easysearch,
    mmseqs_easycluster,
)
from plaid.constants import FUNCTION_IDX_TO_GO_TERM
from plaid.pipeline import run_analysis, move_designable
from plaid.utils import calc_sequence_identity


def default(val, default_val):
    return val if val is not None else default_val


def sample_run(sample_cfg: DictConfig):
    sample_latent = hydra.utils.instantiate(sample_cfg)
    sample_latent = sample_latent.run()  # returns npz path
    with open(sample_latent.outdir / "sample.yaml", "w") as f:
        OmegaConf.save(sample_cfg, f)
    return sample_latent


def decode_run(
    decode_cfg: DictConfig,
    npz_path: PathLike,
    output_root_dir: PathLike,
    esmfold=None,
    max_seq_len=None,
):
    decode_latent = hydra.utils.instantiate(
        decode_cfg, npz_path=npz_path, output_root_dir=output_root_dir, esmfold=esmfold
    )
    seq_strs, pdb_paths = decode_latent.run()
    with open(output_root_dir / "decode_config.yaml", "w") as f:
        OmegaConf.save(decode_cfg, f)
    return seq_strs, pdb_paths


def inverse_generate_structure_run(
    inverse_generate_structure_cfg: DictConfig,
    fasta_file,
    outdir,
    esmfold: torch.nn.Module = None,
):
    fold_pipeline = hydra.utils.instantiate(
        inverse_generate_structure_cfg,
        fasta_file=fasta_file,
        outdir=outdir,
        esmfold=esmfold,
    )
    fold_pipeline.run()
    with open(outdir / "fold_config.yaml", "w") as f:
        OmegaConf.save(inverse_generate_structure_cfg, f)


def inverse_generate_sequence_run(
    inverse_generate_sequence_cfg: DictConfig, pdb_dir=None, output_fasta_path=None
):
    """Hydra configurable instantiation, for imports in full pipeline."""
    inverse_fold = hydra.utils.instantiate(
        inverse_generate_sequence_cfg,
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
    )
    inverse_fold.run()
    with open(Path(output_fasta_path).parent / "inverse_fold_config.yaml", "w") as f:
        OmegaConf.save(inverse_generate_sequence_cfg, f)


def phantom_generate_structure_run(sample_dir):
    # generated structure to inverse-generated sequence
    cmd = f"""omegafold {str(sample_dir / "inverse_generated/sequences.fasta")} {str(sample_dir / "phantom_generated/structures")} --subbatch_size 64"""
    os.system(cmd)


def phantom_generate_sequence_run(
    phantom_generate_sequence_cfg, pdb_dir, output_fasta_path
):
    inverse_fold = hydra.utils.instantiate(
        phantom_generate_sequence_cfg,
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
    )
    inverse_fold.run()
    with open(
        Path(output_fasta_path).parent / "phantom_generate_sequence_config.yaml", "w"
    ) as f:
        OmegaConf.save(phantom_generate_sequence_cfg, f)
    return inverse_fold


@hydra.main(config_path="../configs/inference", config_name="full")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    sample_cfg = cfg.sample
    decode_cfg = cfg.decode
    inverse_generate_sequence_cfg = cfg.inverse_generate_sequence
    inverse_generate_structure_cfg = cfg.inverse_generate_structure
    phantom_generate_sequence_cfg = cfg.phantom_generate_sequence
    # NOTE: no phantom generate structure config, using all OmegaFold defaults.

    # ===========================
    # Sample
    # ===========================

    sample_latent = sample_run(sample_cfg)

    outdir = sample_latent.outdir
    uid = sample_latent.uid
    x = sample_latent.x
    cond_code = sample_latent.cond_code
    go_term = FUNCTION_IDX_TO_GO_TERM[str(cond_code.split("_")[0][1:])]

    # ===========================
    # Decode
    # ===========================

    if not cfg.run_decode:
        exit(0)

    esmfold = esmfold_v1()
    esmfold.eval().requires_grad_(False)
    esmfold.cuda()

    with torch.no_grad():
        seq_strs, pdb_paths = decode_run(
            decode_cfg,
            npz_path=outdir / "latent.npz",
            output_root_dir=outdir / "generated",
            esmfold=esmfold,
        )

    assert len(seq_strs) == len(pdb_paths) == x.shape[0]

    # ===========================
    # Inverse generations
    # ===========================

    if cfg.run_cross_consistency:
        # run ProteinMPNN for generated structures
        input_pdb_dir = outdir / "generated/structures"
        output_fasta_path = outdir / "inverse_generated/sequences.fasta"
        inverse_generate_sequence_run(
            inverse_generate_sequence_cfg,
            pdb_dir=input_pdb_dir,
            output_fasta_path=output_fasta_path,
        )

        # run ESMFold for generated sequences
        input_fasta_file = outdir / "generated" / "sequences.fasta"
        structure_outdir = outdir / "inverse_generated" / "structures"
        inverse_generate_structure_run(
            inverse_generate_structure_cfg,
            fasta_file=input_fasta_file,
            outdir=structure_outdir,
            esmfold=esmfold,
        )
    else:
        pass

    # ===========================
    # Phantom generations
    # ===========================

    if cfg.run_self_consistency:
        # run ProteinMPNN on the structure predictions of our generated sequences to look at self-consistency sequence recovery
        if not (outdir / "phantom_generated").exists():
            Path(outdir / "phantom_generated").mkdir(parents=True)

        input_pdb_dir = outdir / "inverse_generated/structures"
        output_fasta_path = outdir / "phantom_generated/sequences.fasta"
        phantom_generate_sequence_run(
            phantom_generate_sequence_cfg,
            pdb_dir=input_pdb_dir,
            output_fasta_path=output_fasta_path,
        )

        # uses OmegaFold to fold the inverse-fold sequence predictions of generated structures to look at scTM and scRMSD
        phantom_generate_structure_run(outdir)
    else:
        pass

    # ===========================
    # Analysis
    # ===========================

    if run_analysis:
        # TODO: make analysis step also optional

        # run and save result CSV
        rita_perplexity = RITAPerplexity()
        df = run_analysis(outdir, rita_perplexity=rita_perplexity)

        if cfg.log_to_wandb:
            # add wandb molecule object:
            wandb.init(
                project="plaid-sampling2",
                config=OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True),
                id=uid,
                resume="allow",
                name=f"{cond_code} {go_term}",
            )
            df["structure"] = [wandb.Molecule(str(pdbpath)) for pdbpath in pdb_paths]
            df["pdbpath"] = [str(pdbpath) for pdbpath in pdb_paths]
            df["cond_code"] = [cond_code] * len(df)
            df["go_term"] = [go_term] * len(df)
            df["length"] = [len(seq) for seq in seq_strs]

            wandb.log({"generations": wandb.Table(dataframe=df)})

        # ===========================
        # Foldseek
        # ===========================
        # this moves everything into a "designable" subdir
        move_designable(
            df,
            delete_original=False,
            original_dir_prefix="generated/structures",
            target_dir_prefix="",
        )

        if cfg.use_designability_filter:
            subdir_name = "designable"
        else:
            subdir_name = "generated/structures"

        # novelty search for structure conservation
        foldseek_easysearch_outpath = foldseek_easysearch(outdir, subdir_name)
        if cfg.log_to_wandb:
            from plaid.evaluation._foldseek import (
                EASY_SEARCH_OUTPUT_COLS as FOLDSEEK_SEARCH_COLS,
            )

            df = pd.read_csv(foldseek_easysearch_outpath, sep="\t", header=None)
            df.columns = FOLDSEEK_SEARCH_COLS
            try:
                df["theader"] = df.theader.map(lambda x: " ".join(x.split(" ")[1:]))
            except:
                pass
            df = df.sort_values("evalue", ascending=True)
            df = df.groupby("query").head(5)
            wandb.log({"foldseek_easysearch": wandb.Table(dataframe=df)})

        # diversity clustering
        _ = foldseek_easycluster(outdir, subdir_name)

        # ===========================
        # MMseqs
        # ===========================
        # novelty search for sequence conservation
        mmseqs_easysearch_outpath = mmseqs_easysearch(
            outdir, "generated/sequences.fasta"
        )
        mmseqs_easycluster_results = pd.read_csv(mmseqs_easysearch_outpath, sep="\t")
        if cfg.log_to_wandb:
            from plaid.evaluation._mmseqs import (
                EASY_SEARCH_OUTPUT_COLS as MMSEQS_SEARCH_COLS,
            )

            df = pd.read_csv(foldseek_easysearch_outpath, sep="\t", header=None)
            df.columns = MMSEQS_SEARCH_COLS
            try:
                df["theader"] = df.theader.map(lambda x: " ".join(x.split(" ")[1:]))
            except:
                pass
            df = df.sort_values("evalue", ascending=True)
            df = df.groupby("query").head(5)
            df["sequence_identity"] = df.apply(
                lambda row: calc_sequence_identity(row["qseq"], row["tseq"]), axis=1
            )
            wandb.log({"mmseqs_easysearch": wandb.Table(dataframe=df)})

        # diversity clustering
        _ = mmseqs_easycluster(outdir, "generated/sequences.fasta")


if __name__ == "__main__":
    main()
