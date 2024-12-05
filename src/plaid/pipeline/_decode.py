"""
From sampled compressed latent, uncompress and generate sequence and structure using frozen decoders.
"""

from pathlib import Path
from typing import Dict
import time

import torch
import numpy as np
import pandas as pd
import pickle as pkl

from cheap.proteins import LatentToSequence, LatentToStructure
from cheap.utils import LatentScaler
from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32

from plaid.utils import outputs_to_avg_metric, npy, write_pdb_to_disk, write_to_fasta
from plaid.typed import PathLike


def ensure_exists(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


class DecodeLatent:
    """
    Given a NPZ of sampled latent, reconstruct the sequence
    """

    def __init__(
        self,
        npz_path: PathLike,
        output_root_dir: PathLike = None,
        num_recycles=4,
        batch_size=64,
        device="cuda",
        esmfold=None,
        use_compile=False,
        chunk_size=128,
        delete_esm_lm=False,
    ):
        self.npz_path = npz_path
        self.output_root_dir = Path(output_root_dir)
        print("Output root dir:", self.output_root_dir)
        ensure_exists(self.output_root_dir)

        self.esmfold = esmfold
        self.chunk_size = chunk_size
        self.use_compile = use_compile
        self.delete_esm_lm = delete_esm_lm

        self.time_log_path = self.output_root_dir / "construct.log"
        self.num_recycles = num_recycles
        self.batch_size = batch_size
        self.device = device

        self.sequence_constructor = None
        self.structure_constructor = None
        self.cheap_pipeline = CHEAP_pfam_shorten_2_dim_32()
        self.latent_scaler = LatentScaler()
        self.hourglass = self.cheap_pipeline.hourglass_model
        self.hourglass.to(device).eval()

    def load_sampled(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        x = torch.tensor(data["samples"])
        if x.dim() == 4:
            # take last timestep only
            # (B, T, L, D) -> (B, L, D)
            x = x[:, -1, :, :]
        assert x.ndim == 3
        x = x.float()  # float16 to float32
        return x

    def process_x(self, x, mask=None):
        x = x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # uncompress
        with torch.no_grad():
            x = self.hourglass.decode(x, mask)

        # unscale from "smooth" latent space back to "pathological" ESMFold latent space
        return self.latent_scaler.unscale(x)

    def construct_sequence(
        self,
        x_processed,
    ):
        device = self.device
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence()

        # forward pass
        self.sequence_constructor.to(device)
        x_processed = x_processed.to(device)
        with torch.no_grad():
            _, _, sequences = self.sequence_constructor.to_sequence(
                x_processed, return_logits=False
            )
        return sequences

    def construct_structure(self, x_processed, seq_str):
        device = self.device
        if self.structure_constructor is None:
            # warning: this implicitly creates an ESMFold inference model, can be very memory consuming
            self.structure_constructor = LatentToStructure(
                esmfold=self.esmfold,
                chunk_size=self.chunk_size,
                delete_esm_lm=self.delete_esm_lm,
                use_compile=self.use_compile,
            )

        self.structure_constructor.to(device)
        x_processed = x_processed.to(device=device)

        with torch.no_grad():
            pdb_strs = self.structure_constructor.to_structure(
                x_processed,
                sequences=seq_str,
                num_recycles=self.num_recycles,
                batch_size=self.batch_size,
                # return_raw_outputs=True
                return_raw_outputs=False,
            )
        return pdb_strs

    def write_structures_to_disk(self, pdb_strs):
        ensure_exists(self.output_root_dir)
        assert not self.output_root_dir is None
        paths = []
        for i, pdbstr in enumerate(pdb_strs):
            outpath = Path(self.output_root_dir) / "structures" / f"sample{i}.pdb"
            outpath = write_pdb_to_disk(pdbstr, outpath)
            paths.append(outpath)
        return paths

    # def write_structure_metrics_to_disk(self, struct_log_dict: Dict[str, np.ndarray]):
    #     ensure_exists(self.output_root_dir)
    #     # pickle dump
    #     with open(Path(self.output_root_dir) / "structure_metrics.pkl", "wb") as f:
    #         pkl.dump(struct_log_dict, f)

    #     # write a dataframe summary
    #     means = {f"{k}_mean": v.mean() for k, v in struct_log_dict.items()}
    #     stds = {f"{k}_std": v.std() for k, v in struct_log_dict.items()}
    #     d = means | stds
    #     with open(Path(self.output_root_dir) / "structure_metrics.txt", "w") as f:
    #         for k, v in d.items():
    #             f.write(f"{k}: {v:.4f}\n")

    def write_sequences_to_disk(self, sequences, fasta_name, headers=None):
        ensure_exists(self.output_root_dir)
        write_to_fasta(
            sequences=sequences,
            outpath=Path(self.output_root_dir) / fasta_name,
            headers=headers,
        )

    def run(self):
        print("Loading latent samples from", self.npz_path)
        x = self.load_sampled(self.npz_path)

        print("Decompressing latent samples")
        start = time.time()
        x_processed = self.process_x(x)
        del self.cheap_pipeline  # free up memory
        end = time.time()
        with open(self.time_log_path, "w") as f:
            f.write(f"Decompress time: {end - start:.2f} seconds.\n")

        print(f"Constructing sequences and writing to {str(self.output_root_dir)}")
        start = time.time()
        seq_strs = self.construct_sequence(x_processed)
        headers = [f"sample{i}" for i in range(len(seq_strs))]
        self.write_sequences_to_disk(seq_strs, "sequences.fasta", headers=headers)

        del self.sequence_constructor  # free up memory
        end = time.time()
        with open(self.time_log_path, "a") as f:
            f.write(f"Sequence construction time: {end - start:.2f} seconds.\n")

        print(
            f"Constructing structures and writing to {str(self.output_root_dir / 'structures')}"
        )
        start = time.time()
        pdb_strs = self.construct_structure(x_processed, seq_strs)
        pdb_paths = self.write_structures_to_disk(pdb_strs)
        end = time.time()
        with open(self.time_log_path, "a") as f:
            f.write(f"Structure construction time: {end - start:.2f} seconds.\n")

        return seq_strs, pdb_paths
