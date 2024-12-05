from typing import Optional
from pathlib import Path
import glob
import os
import time

import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from safetensors import safe_open
import hydra

from plaid.esmfold import esmfold_v1
from plaid.evaluation import (
    RITAPerplexity,
    CrossConsistencyEvaluation,
    SelfConsistencyEvaluation,
)
from plaid.typed import PathLike
from plaid.utils import extract_avg_b_factor_per_residue


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
    # Sample configuration
    # ===========================

    sample_latent = sample_run(sample_cfg)
    outdir = sample_latent.outdir
    uid = sample_latent.uid

    # if the job id is repeated in an wandb init, it logs to the same entry
    wandb.init(
        project=cfg.wandb_project_name,
        config=OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True),
        id=uid,
        resume="allow",
    )

    wandb.log({"sample_latent_time": sample_latent.sampling_time})

    x = sample_latent.x

    # ===========================
    # Decode, calculate naturalness, and log
    # ===========================
    if not cfg.run_decode:
        exit(0)

    esmfold = esmfold_v1()
    esmfold.eval().requires_grad_(False)
    esmfold.cuda()

    start = time.time()
    with torch.no_grad():
        seq_strs, pdb_paths = decode_run(
            decode_cfg,
            npz_path=outdir / "latent.npz",
            output_root_dir=outdir / "generated",
            esmfold=esmfold,
        )
    end = time.time()
    wandb.log({"decode_time": end - start})

    assert len(seq_strs) == len(pdb_paths) == x.shape[0]

    perplexity_calc = RITAPerplexity()

    samples_d = {
        "structure": [],
        "sequence": [],
        "plddt": [],
        "perplexity": [],
    }

    for i in range(x.shape[0]):
        seqstr, pdbpath = seq_strs[i], pdb_paths[i]
        samples_d["structure"].append(wandb.Molecule(str(pdbpath)))
        samples_d["sequence"].append(seqstr)

        # extract pLDDT scores
        plddts = extract_avg_b_factor_per_residue(pdbpath)
        samples_d["plddt"].append(np.mean(plddts))  # average across the pLDDT scores

        # calculate perplexity
        perplexity = perplexity_calc.calc_perplexity(seqstr)
        samples_d["perplexity"].append(perplexity)

    wandb.log(
        {
            "plddt_mean": np.mean(samples_d["plddt"]),
            "plddt_std": np.std(samples_d["plddt"]),
            "plddt_median": np.median(samples_d["plddt"]),
            "plddt_hist": wandb.Histogram(samples_d["plddt"]),
        }
    )

    wandb.log(
        {
            "perplexity_mean": np.mean(samples_d["perplexity"]),
            "perplexity_std": np.std(samples_d["perplexity"]),
            "perplexity_median": np.median(samples_d["perplexity"]),
            "perplexity_hist": wandb.Histogram(samples_d["perplexity"]),
        }
    )

    del perplexity_calc

    # ===========================
    # Run consistency evals
    # ===========================

    if not cfg.run_cross_consistency:
        wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})
        exit(0)

    try:
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

        # cross_consistency_calc = CrossConsistencyEvaluation(outdir)

        # # should be already sorted in the same order
        # ccrmsd = cross_consistency_calc.cross_consistency_rmsd()
        # # ccrmspd = cross_consistency_calc.cross_consistency_rmspd()
        # cctm = cross_consistency_calc.cross_consistency_tm()
        # ccsr = cross_consistency_calc.cross_consistency_sr()

        # wandb.log(
        #     {
        #         # detailed stats for ccRMSD and ccSR as representative metrics
        #         "ccrmsd_mean": np.mean(ccrmsd),
        #         "ccrmsd_std": np.std(ccrmsd),
        #         "ccrmsd_median": np.std(ccrmsd),
        #         "ccsr_mean": np.mean(ccsr),
        #         "ccsr_std": np.std(ccsr),
        #         "ccsr_median": np.std(ccsr),
        #         # also log histogram
        #         "cctm_hist": wandb.Histogram(cctm),
        #         "ccrmsd_hist": wandb.Histogram(ccrmsd),
        #         # "ccrmspd_hist": wandb.Histogram(ccrmspd),
        #         "ccsr_hist": wandb.Histogram(ccsr),
        #     }
        # )

        # # add to the big table

        # samples_d["ccrmsd"] = ccrmsd
        # # samples_d["ccrmspd"] = ccrmspd
        # samples_d["cctm"] = cctm
        # samples_d["ccsr"] = ccsr

    except:
        # log our table with whatever we got to:
        wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})
        pass

    # ===========================
    # Phantom generations for self-consistency
    # ===========================

    if not cfg.run_self_consistency:
        wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})
        exit(0)

    try:
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

        # self_consistency_calc = SelfConsistencyEvaluation(outdir)

        # self_consistency_rmsd = self_consistency_calc.self_consistency_rmsd()
        # # self_consistency_rmspd = self_consistency_calc.cross_consistency_rmspd()
        # self_consistency_tm = self_consistency_calc.self_consistency_tm()
        # self_consistency_sr = self_consistency_calc.self_consistency_sr()

        # wandb.log(
        #     {
        #         # detailed stats for self-consistency RMSD and SR as representative metrics
        #         "self_consistency_rmsd_mean": np.mean(self_consistency_rmsd),
        #         "self_consistency_rmsd_std": np.std(self_consistency_rmsd),
        #         "self_consistency_rmsd_median": np.median(self_consistency_rmsd),
        #         "self_consistency_sr_mean": np.mean(self_consistency_sr),
        #         "self_consistency_sr_std": np.std(self_consistency_sr),
        #         "self_consistency_sr_median": np.median(self_consistency_sr),
        #         # also log histogram
        #         "self_consistency_tm_hist": wandb.Histogram(self_consistency_tm),
        #         "self_consistency_rmsd_hist": wandb.Histogram(self_consistency_rmsd),
        #         # "self_consistency_rmspd_hist": wandb.Histogram(self_consistency_rmspd),
        #         "self_consistency_sr_hist": wandb.Histogram(self_consistency_sr),
        #     }
        # )

        # # add to the big table
        # samples_d["self_consistency_rmsd"] = self_consistency_rmsd
        # # samples_d["self_consistency_rmspd"] = self_consistency_rmspd
        # samples_d["self_consistency_tm"] = self_consistency_tm
        # samples_d["self_consistency_sr"] = self_consistency_sr

    except:
        pass

    # log our big table
    wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})


if __name__ == "__main__":
    main()
