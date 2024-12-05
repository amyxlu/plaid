# """
# Custom callbacks sampling and evaluation.
# """

# import typing as T
# import os
# import json
# from pathlib import Path

# import wandb
# import torch
# from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.utilities import rank_zero_only
# import safetensors.torch as st
# import pandas as pd
# import numpy as np
# from tqdm import tqdm, trange

# from cheap.model import HourglassProteinCompressionTransformer
# from cheap.proteins import LatentToStructure, LatentToSequence
# from cheap.utils import LatentScaler

# from ..diffusion import FunctionOrganismDiffusion
# from ..evaluation import RITAPerplexity, parmar_fid, parmar_kid
# from ..utils import write_pdb_to_disk, npy


# def maybe_print(msg):
#     if rank_zero_only.rank == 0:
#         print(msg)


# # def _wandb_log(logger, log_dict):
# #     for k, v in log_dict.items():
# #         if "_hist" in k:
# #             log_dict[k] = wandb.Histogram(np_histogram=v)
# #         if "_df" in k:
# #             log_dict[k] = wandb.Table(dataframe=v)
# #     logger.log(log_dict)


# class SampleCallback(Callback):
#     """
#     Calls the `p_sample_loop` functions in the diffusion module,
#     which samples `x` (i.e. normalized and compressed). Then, using the
#     `process_x_to_latent` method in the diffusion module, process x back into
#     the unnormalized and uncompressed version of the latent and construct the sequence
#     and structure.
#     """

#     def __init__(
#         self,
#         diffusion: FunctionOrganismDiffusion,
#         hourglass_model: HourglassProteinCompressionTransformer,
#         n_to_sample: int = 16,
#         n_to_construct: int = 4,
#         gen_seq_len: int = 64,
#         batch_size: int = -1,
#         log_to_wandb: bool = False,
#         calc_structure: bool = True,
#         calc_sequence: bool = True,
#         calc_fid: bool = True,
#         calc_sequence_properties: bool = False,
#         fid_holdout_tensor_fpath: str = "",
#         calc_perplexity: bool = True,
#         save_generated_structures: bool = False,
#         num_recycles: int = 4,
#         outdir: str = "sampled",
#         sequence_decode_temperature: float = 1.0,
#         sequence_constructor: T.Optional[LatentToSequence] = None,
#         structure_constructor: T.Optional[LatentToStructure] = None,
#         run_every_n_steps: int = 1000,
#         n_structures_to_log: T.Optional[int] = None,
#     ):
#         super().__init__()
#         self.outdir = Path(outdir)
#         self.diffusion = diffusion
#         self.log_to_wandb = log_to_wandb
#         self.gen_seq_len = gen_seq_len
#         self.hourglass_model = hourglass_model

#         self.calc_fid = calc_fid
#         self.calc_structure = calc_structure
#         self.calc_sequence = calc_sequence
#         self.calc_perplexity = calc_perplexity
#         self.calc_sequence_properties = calc_sequence_properties
#         self.save_generated_structures = save_generated_structures
#         if calc_perplexity:
#             assert calc_sequence
#         if calc_sequence_properties:
#             assert calc_sequence
#         if save_generated_structures:
#             assert calc_structure

#         self.fid_holdout_tensor_fpath = fid_holdout_tensor_fpath
#         self.n_to_sample = n_to_sample
#         self.num_recycles = num_recycles
#         self.sequence_decode_temperature = sequence_decode_temperature

#         self.is_save_setup = False
#         self.is_fid_setup = False
#         self.is_perplexity_setup = False
#         self.sequence_constructor = sequence_constructor
#         self.structure_constructor = structure_constructor
#         self.scaler = LatentScaler()

#         batch_size = self.n_to_sample if batch_size == -1 else batch_size
#         n_to_construct = self.n_to_sample if n_to_construct == -1 else n_to_construct
#         self.batch_size = batch_size
#         self.n_to_construct = n_to_construct
#         self.gen_seq_len = gen_seq_len
#         self.run_every_n_steps = run_every_n_steps
#         self.n_structures_to_log = n_structures_to_log

#     def _save_setup(self):
#         if not self.outdir.exists():
#             self.outdir.mkdir(parents=True)
#             print("Created directory", self.outdir)
#         self.is_save_setup = True

#     def _perplexity_setup(self, device):
#         self.perplexity_calc = RITAPerplexity(device)
#         self.is_perplexity_setup = True

#     def _load_saved_features(self, device):
#         pass

#     def _fid_setup(self, device):
#         def load_saved_features(location, device="cpu"):
#             return st.load_file(location)["features"].to(device)

#         # def load_saved_features(location, device="cpu"):
#         #     import h5py
#         #     with h5py.File(location, "r") as f:
#         #         tensor = f['embeddings'][()]
#         #     tensor = torch.from_numpy(tensor).to(device)
#         #     return tensor.mean(dim=1)

#         # load saved features (unnormalized)
#         self.real_features = load_saved_features(self.fid_holdout_tensor_fpath, device=device)

#         # take a subset to make the n more comparable between reference and sampled
#         self.real_features = self.real_features[
#             torch.randperm(self.real_features.size(0))[: self.n_to_sample]
#         ]

#         # calculate FID/KID in the normalized space
#         self.real_features = self.scaler.scale(self.real_features)

#         print(
#             "FID reference tensor mean/std:",
#             self.real_features.mean().item(),
#             self.real_features.std().item(),
#         )
#         self.is_fid_setup = True

#     def sample_latent(self, shape, model_kwargs={}):
#         all_samples, n_samples = [], 0
#         for _ in trange(0, self.n_to_sample, shape[0]):
#             x_sampled = self.diffusion.p_sample_loop(shape, clip_denoised=True, progress=True)
#             x_sampled = x_sampled.detach().cpu()
#             all_samples.append(x_sampled)
#             n_samples += x_sampled.shape[0]

#         log_dict = {"sampled/unscaled_latent_hist": np.histogram(x_sampled.numpy().flatten())}
#         all_samples = torch.cat(all_samples)
#         return all_samples, log_dict

#     def calculate_fid(self, x_uncompressed, device):
#         """
#         Calculate FID in the uncompressed but still normalized space, with mean across feature dim.
#         """
#         if not self.is_fid_setup:
#             self._fid_setup(device)

#         fake_features = x_uncompressed.mean(dim=1)

#         # just to be consistent, but not necessary
#         indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
#         real_features = self.real_features[indices]
#         assert real_features.ndim == fake_features.ndim == 2

#         fake_features = fake_features.cpu().numpy()
#         real_features = real_features.cpu().numpy()

#         fid = parmar_fid(fake_features, real_features)
#         kid = parmar_kid(fake_features, real_features)

#         log_dict = {f"sampled/fid": fid, f"sampled/kid": kid}
#         return log_dict

#     def construct_sequence(
#         self,
#         x_processed,
#         device,
#     ):
#         if self.sequence_constructor is None:
#             self.sequence_constructor = LatentToSequence()

#         # forward pass
#         self.sequence_constructor.to(device)
#         x_processed = x_processed.to(device)
#         with torch.no_grad():
#             probs, _, strs = self.sequence_constructor.to_sequence(x_processed, return_logits=False)

#         # organize results for logging
#         sequence_results = pd.DataFrame(
#             {
#                 "sequences": strs,
#                 "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
#             }
#         )

#         if self.calc_sequence_properties:
#             from plaid.utils import calculate_df_protein_property_mp

#             sequence_results = calculate_df_protein_property_mp(df=sequence_results, sequence_col="sequences")

#         log_dict = {f"sampled/sequences_df": sequence_results}
#         log_dict[f"sampled/molecular_weights"] = np.histogram(
#             sequence_results['molecular_weight'].values,
#             bins=min(len(sequence_results), 50)
#         )
#         log_dict[f"sampled/isoelectric_point"] = np.histogram(
#             sequence_results['isoelectric_point'].values,
#             bins=min(len(sequence_results), 50)
#         )

#         if self.calc_perplexity:
#             if not self.is_perplexity_setup:
#                 self._perplexity_setup(device)
#             perplexities = self.perplexity_calc.batch_eval(strs, return_mean=False)
#             print(f"Mean perplexity: {np.mean(perplexities):.3f}")
#             log_dict[f"sampled/perplexity_mean"] = np.mean(perplexities)
#             log_dict[f"sampled/perplexity_hist"] = np.histogram(
#                 np.array(perplexities).flatten(), bins=min(len(perplexities), 50)
#             )
#         return strs, log_dict

#     def construct_structure(self, x_processed, seq_str, device):
#         if self.structure_constructor is None:
#             # warning: this implicitly creates an ESMFold inference model, can be very memory consuming
#             self.structure_constructor = LatentToStructure()

#         self.structure_constructor.to(device)
#         x_processed = x_processed.to(device=device)

#         with torch.no_grad():
#             pdb_strs, outputs = self.structure_constructor.to_structure(
#                 x_processed,
#                 sequences=seq_str,
#                 num_recycles=self.num_recycles,
#                 batch_size=self.batch_size,
#                 return_raw_outputs=True
#             )

#         from plaid.utils import outputs_to_avg_metric

#         metrics = outputs_to_avg_metric(outputs)

#         # `outputs` values are numpy arrays on CPU
#         plddt_per_position = npy(outputs["plddt"].flatten())
#         pae_per_position = npy(outputs["predicted_aligned_error"].flatten())

#         plddt = npy(metrics["plddt"].flatten())
#         pae = npy(metrics["predicted_aligned_error"].flatten())
#         ptm = npy(metrics["ptm"].flatten())

#         N = x_processed.shape[0]

#         log_dict = {
#             f"sampled/plddt_mean": np.mean(plddt),
#             f"sampled/plddt_std": np.std(plddt),
#             f"sampled/plddt_hist": np.histogram(plddt, bins=min(N, 50)),
#             f"sampled/plddt_per_position_hist": np.histogram(plddt_per_position, bins=min(N, 50)),
#             f"sampled/pae_mean": np.mean(pae),
#             f"sampled/pae_std": np.std(pae),
#             f"sampled/pae_hist": np.histogram(pae, bins=min(pae.shape[0], 50)),
#             f"sampled/pae_per_position_hist": np.histogram(pae_per_position, bins=min(N, 50)),
#             f"sampled/ptm_mean": np.mean(pae),
#             f"sampled/ptm_std": np.std(pae),
#             f"sampled/ptm_hist": np.histogram(ptm, bins=min(ptm.shape[0], 50)),
#         }
#         return pdb_strs, metrics, log_dict

#     def on_val_epoch_start(self, *_):
#         print("val epoch start")

#     def _run(self, pl_module, shape, log_to_wandb=True):
#         # set up
#         torch.cuda.empty_cache()
#         log_to_wandb = self.log_to_wandb and log_to_wandb

#         device = pl_module.device
#         if log_to_wandb:
#             logger = pl_module.logger.experiment

#         # sample latent (compressed and standardized)
#         maybe_print("sampling latent...")
#         x, log_dict = self.sample_latent(shape)  # compressed, i.e. (N, L, C_compressed)
#         if log_to_wandb:
#             _wandb_log(logger, log_dict)

#         # uncompress
#         uncompressed = self.hourglass_model.uncompress(x).detach()

#         # calculate FID with uncompressed (but still standardized!)
#         if self.calc_fid:
#             maybe_print("calculating FID...")
#             log_dict = self.calculate_fid(uncompressed, device)
#             if log_to_wandb:
#                 _wandb_log(logger, log_dict)

#         # unstandardize
#         latent = self.scaler.unscale(uncompressed)

#         # subset, and perhaps calculate sequence and structure
#         if not self.n_to_construct == -1:
#             maybe_print(f"subsampling to only reconstruct {self.n_to_construct} samples...")
#             latent = latent[torch.randperm(x.shape[0])][: self.n_to_construct]

#         if self.calc_sequence:
#             maybe_print("constructing sequence...")
#             seq_str, log_dict = self.construct_sequence(latent, device)
#             if log_to_wandb:
#                 _wandb_log(logger, log_dict)

#         if self.calc_structure:
#             maybe_print("constructing structure...")
#             pdb_strs, metrics, log_dict = self.construct_structure(latent, seq_str, device)
#             if log_to_wandb:
#                 _wandb_log(logger, log_dict)

#             if self.save_generated_structures:
#                 all_pdb_paths = self.save_structures_to_disk(pdb_strs, pl_module.global_step)

#                 if not self.n_structures_to_log is None:
#                     df = pd.DataFrame(metrics)
#                     df["pdb_path"] = [str(x) for x in all_pdb_paths]
#                     df = df.sort_values(by="plddt", ascending=False)
#                     df = df.iloc[: self.n_structures_to_log, :]
#                     df["protein"] = df["pdb_path"].map(lambda x: wandb.Molecule(str(x)))
#                     wandb.log({"sampled_proteins_df": df})

#     def save_structures_to_disk(self, pdb_strs, cur_step: int):
#         paths = []
#         for i, pdbstr in enumerate(pdb_strs):
#             outpath = self.outdir / f"step-{cur_step}" / f"sample{i}.pdb"
#             outpath = write_pdb_to_disk(pdbstr, outpath)
#             paths.append(outpath)
#         return paths

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if (pl_module.global_step % self.run_every_n_steps == 0) and not (pl_module.global_step == 0):
#             shape = (self.batch_size, self.gen_seq_len, self.diffusion.model.input_dim)
#             self._run(pl_module, shape, log_to_wandb=True)
#             torch.cuda.empty_cache()
#         else:
#             pass
