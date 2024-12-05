import typing as T

import torch

from cheap.proteins import LatentToSequence, LatentToStructure

from .functions import masked_token_cross_entropy_loss, masked_token_accuracy
from ..openfold_utils import backbone_loss
from ..utils import outputs_to_avg_metric


class SequenceAuxiliaryLoss:
    def __init__(
        self,
        sequence_constructor: LatentToSequence,
        weight: float = 1.0,
        loss_fn: T.Callable = masked_token_cross_entropy_loss,
    ):
        self.sequence_constructor = sequence_constructor
        self.loss_fn = loss_fn
        self.weight = weight

    def __call__(
        self,
        latent,
        aatype,
        mask,
        cur_weight=None,
        return_reconstructed_sequences: bool = False,
    ):
        """If cur weight is specified, it will override self.weight."""
        device = latent.device
        self.sequence_constructor.to(device)
        aatype, mask = aatype.to(device), mask.to(device)

        # grab logits and calculate masked cross entropy (must pass non-default arguments)
        logits, _, recons_strs = self.sequence_constructor.to_sequence(
            latent, mask, return_logits=True, drop_mask_idx=False
        )
        loss = self.loss_fn(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask)
        weight = self.weight if cur_weight is None else cur_weight
        logdict = {
            "seq_loss": loss.item(),
            "seq_acc": acc.item(),
        }
        if return_reconstructed_sequences:
            return weight * loss, logdict, recons_strs
        else:
            return (
                weight * loss,
                logdict,
            )

        # if log_recons_strs:
        #     # wandb logging deep inside a module is suboptimal but lightning logging wandb tables integration is weird
        #     tbl = {"reconstructed": recons_strs}
        #     if not original_sequences is None:
        #         tbl['sequences'] = , "original": sequences}
        #     tbl = pd.DataFrame()
        #     wandb.log({"recons_strs_tbl": wandb.Table(dataframe=tbl)})
        # # TODO: anneal weight by step in outer loop?


class BackboneAuxiliaryLoss:
    def __init__(self, structure_constructor: LatentToStructure, weight=1.0):
        self.structure_constructor = structure_constructor
        self.weight = weight

    def __call__(
        self,
        latent,
        gt_structures,
        sequences,
        num_recycles=1,
        inner_batch_size=None,
        cur_weight=None,
    ):
        device = latent.device
        self.structure_constructor.to(device)

        # check shapes
        batch_size, seq_len, _ = latent.shape
        assert gt_structures["backbone_rigid_tensor"].shape == torch.Size(
            [batch_size, seq_len, 4, 4]
        )
        assert gt_structures["backbone_rigid_mask"].shape == torch.Size(
            [batch_size, seq_len]
        )

        # todo: maybe also log pdb strs
        # pred_structures = self.trunk.from_seq_feat(true_aa, latent)[0]
        pred_pdb_strs, pred_raw_outputs = self.structure_constructor.to_structure(
            latent,
            sequences,
            num_recycles,
            batch_size=inner_batch_size,
            return_raw_features=True,
        )
        assert pred_raw_outputs["frames"].shape == torch.Size(
            [8, batch_size, seq_len, 7]
        )

        loss = backbone_loss(
            backbone_rigid_tensor=gt_structures["backbone_rigid_tensor"].to(device),
            backbone_rigid_mask=gt_structures["backbone_rigid_mask"].to(device),
            traj=pred_raw_outputs["frames"],
        )

        weight = self.weight if cur_weight is None else cur_weight
        metrics = outputs_to_avg_metric(pred_raw_outputs)
        logdict = {"backbone_loss": loss.item()} | metrics
        return weight * loss, logdict


if __name__ == "__main__":
    ####
    # test sequence loss
    ####
    # from plaid.datasets import CATHShardedDataModule
    # from plaid.utils import load_sequence_decoder

    # device = torch.device("cuda:1")

    # # datadir = "/data/lux70/data/cath/shards/"
    # # pklfile = "/data/lux70/data/cath/sequences.pkl"
    # datadir = "/shared/amyxlu/data/cath/shards/"
    # pklfile = "/shared/amyxlu/data/cath/sequences.pkl"

    # dm = CATHShardedDataModule(
    #     # storage_type="hdf5",
    #     seq_len=64,
    #     shard_dir=datadir,
    #     header_to_sequence_file=pklfile,
    #     # dtype="fp32"
    # )
    # dm.setup("fit")
    # train_dataloader = dm.train_dataloader()
    # batch = next(iter(train_dataloader))
    # x, mask, sequence = batch
    # x, mask = x.to(device=device, dtype=torch.float32), mask.to(
    #     device, dtype=torch.float32
    # )
    # sequence = [s[:64] for s in sequence]

    # sequence_decoder = load_sequence_decoder(eval_mode=False, device=device)
    # sequence_loss_fn = SequenceAuxiliaryLoss(sequence_decoder)
    # loss, logdict = sequence_loss_fn(x, sequence, 0.98)
    # import IPython; IPython.embed()

    ####
    # test structure loss
    ####
    from plaid.datasets import CATHStructureDataModule

    shard_dir = "/data/lux70/data/cath/shards/"
    pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
    dm = CATHStructureDataModule(
        shard_dir, pdb_dir, seq_len=64, batch_size=32, num_workers=0
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
