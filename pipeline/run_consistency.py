from pathlib import Path
import os

from omegaconf import DictConfig, OmegaConf
import torch
import hydra

from plaid.esmfold import esmfold_v1


def default(val, default_val):
    return val if val is not None else default_val


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


@hydra.main(config_path="../configs/inference", config_name="consistency")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    inverse_generate_sequence_cfg = cfg.inverse_generate_sequence
    inverse_generate_structure_cfg = cfg.inverse_generate_structure
    phantom_generate_sequence_cfg = cfg.phantom_generate_sequence

    outdir = Path(cfg.samples_dir)

    # ===========================
    # Inverse generations for cross-consistency
    # ===========================

    if cfg.run_inverse:
        if not (outdir / "inverse_generated").exists():
            Path(outdir / "inverse_generated").mkdir(parents=True)

        # run ProteinMPNN for generated structures
        input_pdb_dir = outdir / "generated/structures"
        output_fasta_path = outdir / "inverse_generated/sequences.fasta"
        inverse_generate_sequence_run(
            inverse_generate_sequence_cfg,
            pdb_dir=input_pdb_dir,
            output_fasta_path=output_fasta_path,
        )
        esmfold = esmfold_v1()
        esmfold.eval().requires_grad_(False)
        esmfold.cuda()

        # run ESMFold for generated sequences
        input_fasta_file = outdir / "generated" / "sequences.fasta"
        structure_outdir = outdir / "inverse_generated" / "structures"
        inverse_generate_structure_run(
            inverse_generate_structure_cfg,
            fasta_file=input_fasta_file,
            outdir=structure_outdir,
            esmfold=esmfold,
        )

    # ===========================
    # Phantom generations for self-consistency
    # ===========================

    if cfg.run_phantom:
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


if __name__ == "__main__":
    main()
