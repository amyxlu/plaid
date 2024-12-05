from typing import Optional, List, Tuple
from pathlib import Path
import os

import torch
from tqdm import tqdm

from evo.dataset import FastaDataset

from ..esmfold import output_to_pdb, esmfold_v1
from ..typed import PathLike, DeviceLike
from ..utils import save_pdb_strs_to_disk


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class FoldPipeline:
    def __init__(
        self,
        fasta_file: PathLike,
        outdir: Optional[PathLike],
        esmfold: Optional[torch.nn.Module] = None,
        num_recycles: int = 4,
        batch_size: int = -1,
        max_seq_len: Optional[int] = None,
        save_as_single_pdb_file: bool = False,
        device: DeviceLike = torch.device("cuda"),
        max_num_batches: Optional[int] = None,
        shuffle: bool = False,
    ):
        self.fasta_file = fasta_file
        self.outdir = Path(outdir)
        self.num_recycles = num_recycles
        self.max_seq_len = max_seq_len
        self.device = device
        self.save_as_single_pdb_file = save_as_single_pdb_file

        # tidy up some default values
        self.ds = FastaDataset(fasta_file)
        if batch_size == -1:
            batch_size = len(self.ds)
        self.batch_size = batch_size

        self.dataloader = torch.utils.data.DataLoader(
            self.ds, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        if max_num_batches is None:
            max_num_batches = len(self.dataloader)
        self.max_num_batches = max_num_batches

        # create esmfold
        if esmfold is None:
            esmfold = esmfold_v1()
        self.esmfold = esmfold
        self.esmfold.to(self.device)
        self.esmfold.eval().requires_grad_(False)

    def to(self, device):
        self.device = device
        self.esmfold.to(device)
        return self

    def fold_batch(self, sequences) -> Tuple[List[str], List[float]]:
        # if not self.max_seq_len is None:
        #     sequences = get_random_sequence_crop_batch(sequences, self.max_seq_len, min_len=30)
        if not self.max_seq_len is None:
            sequences = [seq[: self.max_seq_len] for seq in sequences]

        with torch.no_grad():
            # output = self.esmfold.infer(sequences)
            # plddts = output['plddt'].cpu().numpy()
            # assert plddts.ndim == 3
            return self.esmfold.infer_pdbs(sequences, num_recycles=self.num_recycles)

    def run(self):
        if self.outdir is not None:
            print(
                "Warning: make sure there are no repeated headers in the FASTA file.\n"
                "Otherwise, the resulting structures will be overwritten."
            )

        ensure_exists(self.outdir)

        all_headers = []
        all_pdb_strs = []

        num_batches = min(len(self.ds) // self.batch_size, self.max_num_batches)
        # cur_num = 0

        print("Saving structures to", self.outdir)

        for i, batch in tqdm(enumerate(self.dataloader), total=num_batches):
            if self.max_num_batches is not None and i >= self.max_num_batches:
                print(f"Stopping at {self.max_num_batches} batches.")
                break

            headers, sequences = batch
            pdb_strs = self.fold_batch(sequences)
            all_headers.extend(headers)
            all_pdb_strs.extend(pdb_strs)

            # write for this batch:
            for i in range(len(pdb_strs)):
                # sometimes the header is saved as a path, in which case lets remove the backslash
                name = headers[i].split("/")[-1]
                with open(self.outdir / f"{name}.pdb", "w") as f:
                    f.write(pdb_strs[i])
                # cur_num += 1

        return all_pdb_strs
