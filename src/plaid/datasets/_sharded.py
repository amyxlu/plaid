"""
Dataloaders for WebDataset cached tensors, used only for training.
"""

import io
import json
from pathlib import Path
import typing as T

import torch
import webdataset as wds
import numpy as np
import lightning as L

from ._metadata_helpers import MetadataParser
from ..typed import PathLike


def default(value, default):
    return default if value is None else value


def pad_or_trim(arr, max_L):
    original_length = arr.shape[0]
    if original_length <= max_L:
        padding = max_L - original_length
        arr = np.pad(arr, ((0, padding), (0, 0)), mode="constant")
    elif original_length > max_L:
        arr = arr[:max_L]
    return arr


def decode_header(header):
    return header.decode()


def decode_numpy(npy, max_length):
    x = np.load(io.BytesIO(npy)).astype(np.float32)
    original_length = min(max_length, x.shape[0])
    mask = np.ones(original_length)
    x = pad_or_trim(x, max_length)
    mask = np.pad(mask, ((0, max_length - original_length))).astype(bool)
    return x, mask


def make_sample(sample, max_length, metadata_parser):
    """From a loaded sample from a webdataset, extract the relevant conditioning fields."""
    embedding, mask = decode_numpy(sample["embedding.npy"], max_length)
    header = decode_header(sample["header.txt"])
    sample_id = sample["__key__"]
    local_path = sample["__local_path__"]

    pfam_id = metadata_parser.header_to_pfam_id(header)
    go_idx = int(metadata_parser.header_to_go_idx(header))
    organism_idx = int(metadata_parser.header_to_organism_idx(header))
    return embedding, mask, go_idx, organism_idx, pfam_id, sample_id, local_path


class FunctionOrganismDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_shards: str,  # should follow brace expansion format
        val_shards: str,
        config_file: PathLike,
        go_metadata_fpath: PathLike,
        organism_metadata_fpath: PathLike,
        cache_dir: T.Optional[PathLike] = None,
        train_epoch_num_batches: T.Optional[
            int
        ] = None,  # optionally use only some batches for training
        val_epoch_num_batches: T.Optional[
            int
        ] = None,  # optionally use only some batches for val
        shuffle_buffer: int = 1000,
        shuffle_initial: int = 1000,
        max_length: int = 512,
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = shuffle_initial
        self.prefetch_factor = prefetch_factor

        # actual dataset size as stored on disk
        with open(config_file, "r") as f:
            config = json.load(f)
            self.actual_train_data_size = config["train_dataset_size"]
            self.actual_val_data_size = config["val_dataset_size"]

        # number of samples to use in an epoch
        self.train_epoch_num_batches = default(
            train_epoch_num_batches, self.actual_train_data_size // batch_size
        )
        self.val_epoch_num_batches = default(
            val_epoch_num_batches, self.actual_val_data_size // batch_size
        )

        # create local cache dir if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        # metadata parser that turns header into conditioning indices
        self.metadata_parser = MetadataParser(
            go_metadata_fpath=go_metadata_fpath,
            organism_metadata_fpath=organism_metadata_fpath,
        )

    def make_sample(self, sample):
        return make_sample(sample, self.max_length, self.metadata_parser)

    def make_dataloader(self, split):
        assert split in ["train", "val"]
        if split == "train":
            path = self.train_shards
            num_epoch_batches = self.train_epoch_num_batches
            shuffle_buffer = self.shuffle_buffer
            shuffle_initial = self.shuffle_initial
        else:
            path = self.val_shards
            num_epoch_batches = self.val_epoch_num_batches
            shuffle_buffer = 0
            shuffle_initial = 0

        # batching for improving worker-to-loader speed
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                shardshuffle=(split == "train"),
                cache_dir=self.cache_dir,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            .shuffle(shuffle_buffer, initial=shuffle_initial)
            .map(lambda x: self.make_sample(x))
            .batched(self.batch_size)
            .with_epoch(self.batch_size * self.train_epoch_num_batches)
        )

        # unbatch, shuffle, and form final batch for SGD (if training)
        dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": None,
            "pin_memory": True,
            "prefetch_factor": self.prefetch_factor,
        }

        dataloader = (
            wds.WebLoader(dataset, **dataloader_kwargs)
            .unbatched()
            .shuffle(shuffle_buffer)
            .batched(self.batch_size)
            .with_length(num_epoch_batches)
        )

        return dataset, dataloader

    def setup(self, stage=None):
        self.train_ds, self.train_dl = self.make_dataloader("train")
        self.val_ds, self.val_dl = self.make_dataloader("val")

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl


def get_test_sharded_batch(batch_size=1024):
    datamodule = FunctionOrganismDataModule(
        train_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/train/shard{0000..4423}.tar",
        val_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/val/shard{0000..0863}.tar",
        config_file="/data/lux70/data/pfam/compressed/j1v1wv6w/config.json",
        go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
        cache_dir="/homefs/home/lux70/cache/plaid_data_cache/j1v1wv6w",
        train_epoch_num_batches=100_000,
        val_epoch_num_batches=1_000,
        shuffle_buffer=10_000,
        shuffle_initial=10_000,
        num_workers=4,
        batch_size=batch_size,
    )

    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    batch = next(iter(val_dataloader))

    print(
        "embedding, mask, go_idx, organism_idx, pfam_id, sample_id, local_path = batch"
    )
    return batch
