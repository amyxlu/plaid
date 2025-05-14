"""
Misc dataloaders not used for the final model, included for completeness.
"""

import math

import einops
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import lmdb

from safetensors.torch import load_file
from pathlib import Path
import pickle
import typing as T
import lightning as L

from ..transforms import (
    mask_from_seq_lens,
    trim_or_pad_length_first,
)
from ..constants import ACCEPTED_LM_EMBEDDER_TYPES

from ._datasets import FastaDataset


class TensorShardDataset(Dataset):
    """Loads entire dataset as one Safetensor dataset. Returns the embedding, mask, and pdb id."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: str = "/shared/amyxlu/data/cath/sequences.pkl",
        max_seq_len: int = 64,
        dtype: str = "bf16",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.seq_len = max_seq_len
        self.shard_dir = Path(shard_dir)
        self.header_to_seq = pickle.load(open(header_to_sequence_file, "rb"))
        self.embs, self.masks, self.ordered_headers = self.load_partition(split)

    def load_partition(self, split: T.Optional[str] = None):
        if not split is None:
            assert split in ("train", "val")
            datadir = self.shard_dir / split
        else:
            datadir = self.shard_dir
        datadir = datadir / f"seqlen_{self.seq_len}" / self.dtype
        data = load_file(datadir / "shard0000.pt")
        assert data.keys() == set(("embeddings", "seq_len"))
        emb, seqlen = data["embeddings"], data["seq_len"]
        mask = mask_from_seq_lens(emb, seqlen)

        ordered_headers = open(datadir / "../shard0000.txt").readlines()
        ordered_headers = [h.rstrip("\n") for h in ordered_headers]
        return emb, mask, ordered_headers

    def __len__(self):
        return self.embs.size(0)

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor, str]:
        header = self.ordered_headers[idx]
        if "|" in header:
            header = header.split("|")[-1].split("/")[0]
        return (
            self.embs[idx, ...],
            self.masks[idx, ...],
            self.header_to_seq[header],
        )


class H5ShardDataset(Dataset):
    """Loads H5 dataset, which is able to actually store strings, but is
    not able to support bf16 storage."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        embedder: str = "esmfold",
        max_seq_len: int = 64,
        dtype: str = "fp32",
        filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
        # ids_to_drop: T.Optional[T.List[str]] = None,
    ):
        super().__init__()
        self.filtered_ids_list = filtered_ids_list
        # self.ids_to_drop = ids_to_drop
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.shard_dir = Path(shard_dir)
        self.embedder = embedder
        self.max_num_samples = max_num_samples

        self.data = self.load_partition(
            split, embedder, max_num_samples, filtered_ids_list
        )
        pdb_ids = list(self.data.keys())

        self.pdb_ids = list(pdb_ids)

    def drop_protein(self, pid):
        drop = False
        # fixed chain parsing issue, shouldn't have to drop proteins now.
        # if not (self.ids_to_drop is None) and (pid in self.ids_to_drop):
        #     drop = True
        return drop

    def load_partition(
        self,
        split: T.Optional[str] = None,
        embedder: T.Optional[str] = None,
        max_num_samples: T.Optional[int] = None,
        filtered_ids_list: T.Optional[T.List[str]] = None,
    ):
        """
        2024/02/15: path format:
        ${shard_dir}/${split}/${embedder}/${seqlen}/${precision}/shard0000.h5
        """
        # make sure that the specifications are valid
        datadir = self.shard_dir
        if not split is None:
            assert split in ("train", "val")
            datadir = datadir / split

        if not embedder is None:
            assert embedder in ACCEPTED_LM_EMBEDDER_TYPES
            datadir = datadir / embedder

        datadir = datadir / f"seqlen_{self.max_seq_len}" / self.dtype
        outdict = {}

        # load the shard hdf5 file
        with h5py.File(datadir / "shard0000.h5", "r") as f:
            emb = torch.from_numpy(np.array(f["embeddings"]))
            sequence = list(f["sequences"])
            pdb_ids = list(f["pdb_id"])

            # if prespecified a set of pdb ids, only load those
            if not filtered_ids_list is None:
                pdb_ids = set(pdb_ids).intersection(set(filtered_ids_list))
                disjoint = set(filtered_ids_list) - set(pdb_ids)
                print(
                    f"Did not find {len(disjoint)} IDs, including {list(disjoint)[:3]}, etc."
                )
                pdb_ids = list(pdb_ids)

            # possible trim to a subset to enable faster loading
            if not max_num_samples is None:
                pdb_ids = pdb_ids[:max_num_samples]

            # loop through and decode the protein string one by one
            for i in range(len(pdb_ids)):
                pid = pdb_ids[i].decode()
                if not self.drop_protein(pid):
                    outdict[pid] = (emb[i, ...], sequence[i].decode())
        return outdict

    def __len__(self):
        return len(self.pdb_ids)

    def get(self, idx: int) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # return (self.embs[idx, ...], self.sequences[idx], self.pdb_id[idx])
        assert isinstance(self.pdb_ids, list)
        pid = self.pdb_ids[idx]
        return pid, self.data[pid]

    def __getitem__(
        self, idx: int
    ) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # wrapper for non-structure dataloaders, rearrange output tuple
        pdb_id, (emb, seq) = self.get(idx)
        return emb, seq, pdb_id


class CATHStructureDataset(H5ShardDataset):
    """Wrapper around H5 shard dataset. Returns actual structure features as well."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        pdb_path_dir: str = "/shared/amyxlu/data/cath/full/dompdb",
        embedder: str = "esmfold",
        max_seq_len: int = 64,
        dtype: str = "fp32",
        path_to_filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
    ):
        if not path_to_filtered_ids_list is None:
            with open(path_to_filtered_ids_list, "r") as f:
                filtered_ids_list = f.read().splitlines()
        else:
            filtered_ids_list = None

        super().__init__(
            split=split,
            shard_dir=shard_dir,
            embedder=embedder,
            max_seq_len=max_seq_len,
            dtype=dtype,
            filtered_ids_list=filtered_ids_list,
            max_num_samples=max_num_samples,
        )

        from plaid.utils import StructureFeaturizer

        self.structure_featurizer = StructureFeaturizer()
        self.pdb_path_dir = Path(pdb_path_dir)
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx: int):
        pdb_id, (emb, seq) = self.get(idx)
        pdb_path = self.pdb_path_dir / pdb_id
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
        # try:
        #     structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        #     return emb, seq, structure_features
        # except KeyError as e:
        #     with open("bad_ids.txt", "a") as f:
        #         print(pdb_id, e)
        #         f.write(f"{pdb_id}\n")
        #     pass
        structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        return emb, seq, structure_features


class TokenDataset(Dataset):
    def __init__(
        self,
        split,
        compress_model_id,  # = "2024-03-05T06-20-52",  # soft-violet
        token_dir,  # = "/data/lux70/data/cath/tokens/",
        max_seq_len=512,  # = 128,
    ):
        self.split = split
        self.compress_model_id = compress_model_id
        self.token_dir = Path(token_dir)
        self.max_seq_len = max_seq_len

        self.tokens = self.load_partition(split)["tokens"]

    def load_partition(self, split):
        outpath = (
            self.token_dir
            / self.compress_model_id
            / split
            / f"seqlen_{self.max_seq_len}"
            / "tokens.st"
        )
        return load_file(outpath)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx, ...]
        tokens = tokens.long()
        tokens = einops.rearrange(tokens, "l c -> (l c)")
        return tokens


class CompressedContinuousDataset(Dataset):
    """TODO: possibly refactor to inherit from the same base class as H5ShardDataset"""

    def __init__(
        self,
        split,
        compression_model_id,
        seq_len,
        base_data_dir="/data/lux70/data/rocklin/compressed/",
        filtered_ids_list=None,
        max_num_samples=None,
    ):
        super().__init__()
        self.datadir = (
            Path(base_data_dir)
            / split
            / f"hourglass_{compression_model_id}"
            / f"seqlen_{seq_len}"
        )
        self.filtered_ids_list = filtered_ids_list
        self.max_num_samples = max_num_samples
        self.data = self._load_filtered_partition()
        self.pdb_ids = list(self.data.keys())

    def _load_filtered_partition(self):
        outdict = {}
        # load the shard hdf5 file
        with h5py.File(self.datadir / "shard0000.h5", "r") as f:
            emb = torch.from_numpy(np.array(f["embeddings"]))
            sequence = list(f["sequences"])
            pdb_ids = list(f["pdb_id"])

            # if prespecified a set of pdb ids, only load those
            if not self.filtered_ids_list is None:
                pdb_ids = set(pdb_ids).intersection(set(self.filtered_ids_list))
                disjoint = set(self.filtered_ids_list) - set(pdb_ids)
                print(
                    f"Did not find {len(disjoint)} IDs, including {list(disjoint)[:3]}, etc."
                )
                pdb_ids = list(pdb_ids)

            # possible trim to a subset to enable faster loading
            if not self.max_num_samples is None:
                pdb_ids = pdb_ids[: self.max_num_samples]

            # loop through and decode the protein string one by one
            for i in range(len(pdb_ids)):
                pid = pdb_ids[i].decode()
                # NOTE: could also refactor such that we work on a "keep unless dropped" basis
                # if not self.drop_protein(pid):
                outdict[pid] = (emb[i, ...], sequence[i].decode())
        return outdict

    def __len__(self):
        return len(self.pdb_ids)

    def get(self, idx: int) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # define a separate fn to separate out __getitem__ logic for dataloaders
        # that load a ground truth structure vs. embedding only.
        assert isinstance(self.pdb_ids, list)
        pid = self.pdb_ids[idx]
        return pid, self.data[pid]

    def __getitem__(
        self, idx: int
    ) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # wrapper for non-structure dataloaders, rearrange output tuple
        pdb_id, (emb, seq) = self.get(idx)
        return emb, seq, pdb_id


"""
Datamodule wrappers
"""


class CompressedContinuousDataModule(L.LightningDataModule):
    def __init__(
        self,
        compression_model_id,
        seq_len,
        base_data_dir,
        batch_size,
        num_workers,
        shuffle_val_dataset=False,
        filtered_ids_list=None,
        max_num_samples=None,
    ):
        super().__init__()
        self.dataset_fn = CompressedContinuousDataset
        self.shuffle_val_dataset = shuffle_val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = {
            "compression_model_id": compression_model_id,
            "seq_len": seq_len,
            "base_data_dir": base_data_dir,
            "filtered_ids_list": filtered_ids_list,
            "max_num_samples": max_num_samples,
        }

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = self.dataset_fn("train", **self.dataset_kwargs)
            self.val_dataset = self.dataset_fn("val", **self.dataset_kwargs)
        elif stage == "predict":
            self.test_dataset = self.dataset_fn("val", **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CATHShardedDataModule(L.LightningDataModule):
    def __init__(
        self,
        storage_type: str = "hdf5",
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        embedder: str = "esmfold",
        header_to_sequence_file: T.Optional[str] = None,
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        dtype: str = "fp32",
        shuffle_val_dataset: bool = False,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.embedder = embedder
        self.dtype = dtype
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.storage_type = storage_type
        self.shuffle_val_dataset = shuffle_val_dataset

        assert storage_type in ("safetensors", "hdf5")
        if storage_type == "safetensors":
            assert not header_to_sequence_file is None
            self.header_to_sequence_file = header_to_sequence_file
            self.dataset_fn = TensorShardDataset
        elif storage_type == "hdf5":
            self.dataset_fn = H5ShardDataset

    def setup(self, stage: str = "fit"):
        kwargs = {}
        if self.storage_type == "safetensors":
            kwargs["header_to_sequence_file"] = self.header_to_sequence_file
        if self.storage_type == "hdf5":
            kwargs["embedder"] = self.embedder

        if stage == "fit":
            self.train_dataset = self.dataset_fn(
                "train",
                self.shard_dir,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
            self.val_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CATHStructureDataModule(L.LightningDataModule):
    """Lightning datamodule that loads cached CATH tensors and structure features.
    Returns a tuple of (embedding, sequence_strings, dictionary_of_structure_features)

    Note: the structure features includes a list of keys that includes 'sequence'; this is the sequence
    parsed from the PDB file, and includes the full, untrimmed sequence.
    The sequence strings are saved from the original sequences that produced the
    embeddings, and always trimmed to self.seq_len. The string identity should be usually the
    same, unless in cases where the FASTA and PDB files in the CATH database differs in sidechain identity.
    """

    def __init__(
        self,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        pdb_path_dir: str = "/shared/amyxlu/data/cath/full/dompdb",
        embedder: str = "esmfold",
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        path_to_filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
        shuffle_val_dataset: bool = False,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.pdb_path_dir = pdb_path_dir
        self.embedder = embedder
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype = "fp32"
        self.path_to_filtered_ids_list = path_to_filtered_ids_list
        self.max_num_samples = max_num_samples
        self.shuffle_val_dataset = shuffle_val_dataset

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = CATHStructureDataset(
                split="train",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
            self.val_dataset = CATHStructureDataset(
                "val",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
        elif stage == "predict":
            self.test_dataset = CATHStructureDataset(
                "val",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class FastaDataModule(L.LightningDataModule):
    def __init__(
        self,
        fasta_file: str,
        batch_size: int,
        train_frac: float = 0.8,
        num_workers: int = 0,
        shuffle_val_dataset: bool = False,
        seq_len: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.fasta_file = fasta_file
        self.train_frac, self.val_frac = train_frac, 1 - train_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset
        # hack, cropping is actually done in ESMFold emb
        self.seq_len = seq_len

    def setup(self, stage: str = "fit"):
        ds = FastaDataset(self.fasta_file, cache_indices=True)
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            ds, [self.train_frac, self.val_frac], generator=seed
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class TokenDataModule(L.LightningDataModule):
    def __init__(
        self,
        compress_model_id="2024-03-05T06-20-52",  # soft-violet
        token_dir="/data/lux70/data/cath/tokens/",
        max_seq_len=128,
        batch_size=256,
        num_workers=0,
        shuffle_val_dataset=False,
    ):
        super().__init__()
        self.compress_model_id = compress_model_id
        self.token_dir = Path(token_dir)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = TokenDataset(
                "train",
                compress_model_id=self.compress_model_id,
                token_dir=self.token_dir,
                max_seq_len=self.max_seq_len,
            )
            self.val_dataset = TokenDataset(
                "val",
                compress_model_id=self.compress_model_id,
                token_dir=self.token_dir,
                max_seq_len=self.max_seq_len,
            )
        elif stage == "predict":
            self.test_dataset = TokenDataset(
                "val",
                compress_model_id=self.compress_model_id,
                token_dir=self.token_dir,
                max_seq_len=self.max_seq_len,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CompressedLMDBDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.all_headers = pickle.loads(txn.get(b"all_headers"))
            self.max_seq_len = int.from_bytes(txn.get(b"max_seq_len"))
            self.shorten_factor = int.from_bytes(txn.get(b"shorten_factor"))
            self.hid_dim = int.from_bytes(txn.get(b"hid_dim"))

        self.pad_idx = 0
        # TODO: parse sequences into dictionary to also load sequences
        # only needed if using sequence auxiliary loss for diffusion

    def __len__(self):
        return len(self.all_headers)

    def __getitem__(self, idx):
        header_bytes = self.all_headers[idx]
        header = header_bytes.decode()
        with self.env.begin(write=False) as txn:
            if txn.get(header_bytes) is None:
                print(f"Key {header_bytes} not found.")
            else:
                print(f"Key {header_bytes} found.")
                emb = np.frombuffer(txn.get(header_bytes), dtype=np.float32)
        emb = torch.tensor(emb.reshape(-1, self.hid_dim))
        L, C = emb.shape
        emb = trim_or_pad_length_first(emb, self.max_seq_len, self.pad_idx)
        mask = torch.arange(self.max_seq_len) < L
        sequence = ""  # TODO: once sequences are parsed, can return this properly
        return emb, mask, header, sequence

    def __del__(self):
        self.env.close()


class CompressedLMDBDataModule(L.LightningDataModule):
    def __init__(
        self,
        compression_model_id="jzlv54wl",
        lmdb_root_dir="/data/lux70/data/pfam/compressed/subset_5000",
        max_seq_len=512,
        batch_size=128,
        num_workers=8,
        shuffle_val_dataset=False,
    ):
        super().__init__()
        self.compression_model_id = compression_model_id
        self.lmdb_root_dir = lmdb_root_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset

        base_dir = (
            Path(lmdb_root_dir)
            / f"hourglass_{compression_model_id}"
            / f"seqlen_{max_seq_len}"
        )
        self.train_lmdb_path = base_dir / "train.lmdb"
        self.val_lmdb_path = base_dir / "val.lmdb"

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = CompressedLMDBDataset(self.train_lmdb_path)
            self.val_dataset = CompressedLMDBDataset(self.val_lmdb_path)
        elif stage == "predict":
            self.test_dataset = CompressedLMDBDataset(self.val_lmdb_path)
        else:
            return ValueError(f"Stage must be 'fit' or 'predict', got {stage}.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CompressedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        fh = h5py.File(h5_path, "r")
        self.max_seq_len = fh.attrs["max_seq_len"]
        self.shorten_factor = fh.attrs["shorten_factor"]
        self.compressed_hid_dim = fh.attrs["compressed_hid_dim"]
        self.dataset_size = fh.attrs["dataset_size"]
        self.fh = fh

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        ds = self.fh[str(idx)]
        emb, seq = torch.tensor(ds[:]), ds.attrs["sequence"]
        L, C = emb.shape
        emb = trim_or_pad_length_first(emb, self.max_seq_len)
        mask = torch.arange(self.max_seq_len) < L
        return {"emb": emb, "mask": mask, "sequence": seq}


class MayClanCompressedDataset(CompressedH5Dataset):
    def __init__(self, h5_path):
        super().__init__(h5_path)

    def __getitem__(self, idx):
        group = self.fh[str(idx)]
        emb, clan, original_l = group[:], group.attrs["clan"], group.attrs["len"]
        emb, clan, original_l = tuple(
            map(lambda x: torch.tensor(x), (emb, clan, original_l))
        )
        s = self.shorten_factor
        effective_max_l = math.ceil(self.max_seq_len / s)
        effective_cur_l = math.ceil(original_l[0] / s)
        emb = trim_or_pad_length_first(emb, effective_max_l)
        mask = torch.arange(len(emb)) < effective_cur_l
        return {"emb": emb, "mask": mask, "clan": clan}


class CompressedH5DataModule(L.LightningDataModule):
    def __init__(
        self,
        compression_model_id,
        h5_root_dir,
        max_seq_len,
        batch_size=128,
        num_workers=8,
        shuffle_val_dataset=False,
        mayclan_version=False,
        aprclan_version=False,
    ):

        super().__init__()
        self.compression_model_id = compression_model_id
        self.h5_root_dir = h5_root_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset

        if mayclan_version:
            self.dataset_fn = MayClanCompressedDataset
        elif aprclan_version:
            self.dataset_fn = CompressedH5ClansDataset
        else:
            self.dataset_fn = CompressedH5Dataset

        base_dir = (
            Path(h5_root_dir)
            / f"hourglass_{compression_model_id}"
            / f"seqlen_{max_seq_len}"
        )
        self.train_h5_path = base_dir / "train.h5"
        self.val_h5_path = base_dir / "val.h5"

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = self.dataset_fn(self.train_h5_path)
            self.val_dataset = self.dataset_fn(self.val_h5_path)
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(self.val_h5_path)
        else:
            return ValueError(f"Stage must be 'fit' or 'predict', got {stage}.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class CompressedH5ClansDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        fh = h5py.File(h5_path, "r")
        self.max_seq_len = fh.attrs["max_seq_len"]
        self.shorten_factor = fh.attrs["shorten_factor"]
        self.compressed_hid_dim = fh.attrs["compressed_hid_dim"]
        self.dataset_size = fh.attrs["dataset_size"]
        self.fh = fh

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        group = self.fh[str(idx)]
        emb, clan, original_l = group[:], group.attrs["clan"], group.attrs["len"]
        emb, clan, original_l = tuple(
            map(lambda x: torch.tensor(x), (emb, clan, original_l))
        )
        s = self.shorten_factor
        effective_max_l = math.ceil(self.max_seq_len / s)
        effective_cur_l = math.ceil(original_l[0] / s)
        emb = trim_or_pad_length_first(emb, effective_max_l)
        mask = torch.arange(len(emb)) < effective_cur_l
        return {"emb": emb, "mask": mask, "clan": clan}


if __name__ == "__main__":
    dm = CompressedH5DataModule(
        compression_model_id="j1v1wv6w",
        h5_root_dir="/data/lux70/data/pfam/compressed/subset_1M_with_clan/fp16/",
        max_seq_len=512,
        batch_size=16,
        mayclan_version=True,
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    import IPython

    IPython.embed()
