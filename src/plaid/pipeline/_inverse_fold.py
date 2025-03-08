"""
Wrapper for running ProteinMPNN, based on:
https://colab.research.google.com/github/dauparas/ProteinMPNN/blob/main/colab_notebooks/quickdemo.ipynb
https://github.com/dauparas/ProteinMPNN/blob/main/colab_notebooks/ca_only_quickdemo.ipynb
"""

import glob
import os
import typing as T
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import re
import copy

from .ProteinMPNN.protein_mpnn_utils import (
    _scores,
    _S_to_seq,
    tied_featurize,
    parse_PDB,
    StructureDatasetPDB,
    ProteinMPNN,
)

from ..utils import write_to_fasta
from ..typed import PathLike, DeviceLike


def ensure_exists(path):
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


class InverseFoldPipeline:
    """Monomer-only, CA-only ProteinMPNN wrapper for inverse folding."""

    def __init__(
        self,
        pdb_dir: PathLike,  # list to PDB files, will inverse fold all files with .pdb extension
        output_fasta_path: PathLike = None,
        # ProteinMPNN backbone-only defaults:
        model_name="v_48_020",  # @param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
        verbose: bool = False,
        hidden_dim=128,
        num_layers=3,
        base_dir_to_model_weights: str = os.path.join(
            os.path.dirname(__file__), "ProteinMPNN"
        ),
        ca_only: bool = True,
        batch_size: int = 1,  # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
        max_length: int = 512,  # Max sequence length
        num_seq_per_target: int = 8,  # Sequences to generate per structure, must be a power of 2
        sampling_temp: str = "0.1",  # Sampling temperature, one of ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]
        designed_chain: str = "A",  # IMPORTANT: only supports monomers for now!
        fixed_chain: str = "",
        backbone_noise: float = 0.00,  # Standard deviation of Gaussian noise to add to backbone atoms
        homomer: bool = True,
        omit_AAs: str = "X",  # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.
        save_score: bool = False,  # save score=-log_prob to ny files
        save_probs: bool = False,  # save MPNN predicted probabilities per position
        score_only: bool = False,  # score input backbone-sequence pairs
        conditional_probs_only: bool = False,  # output conditional probabilities p(s_i given the rest of the sequence and backbone)
        conditional_probs_only_backbone: bool = False,  # if true output conditional probabilities p(s_i given backbone)
        pssm_multi: float = 0.0,  # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
        pssm_threshold: float = 0.0,  # A value between -inf and +inf to restrict per position AAs
        pssm_log_odds_flag: bool = False,
        pssm_bias_flag: bool = False,
        device: DeviceLike = "cuda",
    ):
        self.pdb_dir = Path(pdb_dir)
        self.verbose = verbose
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_dir_to_model_weights = base_dir_to_model_weights
        self.ca_only = ca_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_seq_per_target = num_seq_per_target
        self.sampling_temp = sampling_temp
        self.designed_chain = designed_chain
        self.fixed_chain = fixed_chain
        self.backbone_noise = backbone_noise
        self.homomer = homomer
        self.omit_AAs = omit_AAs
        self.save_score = save_score
        self.save_probs = save_probs
        self.score_only = score_only
        self.conditional_probs_only = conditional_probs_only
        self.conditional_probs_only_backbone = conditional_probs_only_backbone
        self.pssm_multi = pssm_multi
        self.pssm_threshold = pssm_threshold
        self.pssm_log_odds_flag = pssm_log_odds_flag
        self.pssm_bias_flag = pssm_bias_flag

        if output_fasta_path is None:
            output_fasta_path = self.pdb_dir / "../inverse_generate/sequences.fasta"
        self.output_fasta_path = Path(output_fasta_path)

        self.device = device
        self.model = self.load_protein_mpnn_model()
        self.model = self.model.to(self.device)

    def to(self, device: DeviceLike):
        self.device = device
        self.model = self.model.to(self.device)
        return self

    def load_protein_mpnn_model(self):
        if self.ca_only:
            checkpoint_path = (
                Path(self.base_dir_to_model_weights)
                / "ca_model_weights"
                / f"{self.model_name}.pt"
            )
        else:
            checkpoint_path = (
                Path(self.base_dir_to_model_weights)
                / "vanilla_model_weights"
                / f"{self.model_name}.pt"
            )

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print("Number of edges:", checkpoint["num_edges"])
        noise_level_print = checkpoint["noise_level"]
        print(f"Training noise level: {noise_level_print}A")

        # Load the CA only model
        # model = ProteinMPNN(num_letters=21, node_features=self.hidden_dim, edge_features=self.hidden_dim, hidden_dim=self.hidden_dim, num_encoder_layers=self.num_layers, num_decoder_layers=self.num_layers, augment_eps=self.backbone_noise, k_neighbors=checkpoint['num_edges'])
        model = ProteinMPNN(
            ca_only=self.ca_only,
            num_letters=21,
            node_features=self.hidden_dim,
            edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            augment_eps=self.backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("ProteinMPNN model loaded")
        model.to(self.device)
        return model

    def _make_tied_positions_for_homomers(self, pdb_dict_list):
        my_dict = {}
        for result in pdb_dict_list:
            all_chain_list = sorted(
                [item[-1:] for item in list(result) if item[:9] == "seq_chain"]
            )  # A, B, C, ...
            tied_positions_list = []
            chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
            for i in range(1, chain_length + 1):
                temp_dict = {}
                for j, chain in enumerate(all_chain_list):
                    temp_dict[chain] = [i]  # needs to be a list
                tied_positions_list.append(temp_dict)
            my_dict[result["name"]] = tied_positions_list
        return my_dict

    def inverse_fold(
        self, pdb_path: PathLike, return_outdict: bool = False, verbose: bool = False
    ) -> T.Dict[str, np.ndarray]:
        if self.designed_chain == "":
            designed_chain_list = []
        else:
            designed_chain_list = re.sub("[^A-Za-z]+", ",", self.designed_chain).split(
                ","
            )

        if self.fixed_chain == "":
            fixed_chain_list = []
        else:
            fixed_chain_list = re.sub("[^A-Za-z]+", ",", self.fixed_chain).split(",")

        chain_list = list(set(designed_chain_list + fixed_chain_list))

        NUM_BATCHES = self.num_seq_per_target // self.batch_size
        BATCH_COPIES = self.batch_size
        temperatures = [float(item) for item in self.sampling_temp.split()]
        omit_AAs_list = self.omit_AAs
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"

        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(
            np.float32
        )

        chain_id_dict = None
        fixed_positions_dict = None
        pssm_dict = None
        omit_AA_dict = None
        bias_AA_dict = None
        tied_positions_dict = None
        bias_by_res_dict = None
        bias_AAs_np = np.zeros(len(alphabet))

        ###############################################################
        pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
        dataset_valid = StructureDatasetPDB(
            pdb_dict_list, truncate=None, max_length=self.max_length
        )

        print("Dataset valid", len(dataset_valid))
        if len(dataset_valid) == 0:
            raise ValueError(
                f"No valid chains found in the input PDB file less than specified max length {self.max_length}"
            )

        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]["name"]] = (
            designed_chain_list,
            fixed_chain_list,
        )

        for chain in chain_list:
            l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
        if verbose:
            print(chain_id_dict)
            print(f"Length of chain {chain} is {l}")

        if self.homomer:
            tied_positions_dict = self._make_tied_positions_for_homomers(pdb_dict_list)
        else:
            tied_positions_dict = None

        with torch.no_grad():
            if verbose:
                print("Generating sequences...")

            for ix, protein in enumerate(dataset_valid):
                score_list = []
                all_probs_list = []
                all_log_probs_list = []
                S_sample_list = []
                sequences = []
                batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    fixed_positions_dict,
                    omit_AA_dict,
                    tied_positions_dict,
                    pssm_dict,
                    bias_by_res_dict,
                    ca_only=True,
                )
                pssm_log_odds_mask = (
                    pssm_log_odds_all > self.pssm_threshold
                ).float()  # 1.0 for true, 0.0 for false
                name_ = batch_clones[0]["name"]

                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = self.model(
                    X,
                    S,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn_1,
                )
                mask_for_loss = mask * chain_M * chain_M_pos
                # https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py#L39
                scores = _scores(S, log_probs, mask_for_loss)
                native_score = scores.cpu().data.numpy()

                for temp in temperatures:  # default: single run
                    for j in range(NUM_BATCHES):  # num of sequences to sample
                        randn_2 = torch.randn(chain_M.shape, device=X.device)
                        if tied_positions_dict == None:
                            sample_dict = self.model.sample(
                                X,
                                randn_2,
                                S,
                                chain_M,
                                chain_encoding_all,
                                residue_idx,
                                mask=mask,
                                temperature=temp,
                                omit_AAs_np=omit_AAs_np,
                                bias_AAs_np=bias_AAs_np,
                                chain_M_pos=chain_M_pos,
                                omit_AA_mask=omit_AA_mask,
                                pssm_coef=pssm_coef,
                                pssm_bias=pssm_bias,
                                pssm_multi=self.pssm_multi,
                                pssm_log_odds_flag=bool(self.pssm_log_odds_flag),
                                pssm_log_odds_mask=pssm_log_odds_mask,
                                pssm_bias_flag=bool(self.pssm_bias_flag),
                                bias_by_res=bias_by_res_all,
                            )
                            S_sample = sample_dict["S"]
                        else:
                            sample_dict = self.model.tied_sample(
                                X,
                                randn_2,
                                S,
                                chain_M,
                                chain_encoding_all,
                                residue_idx,
                                mask=mask,
                                temperature=temp,
                                omit_AAs_np=omit_AAs_np,
                                bias_AAs_np=bias_AAs_np,
                                chain_M_pos=chain_M_pos,
                                omit_AA_mask=omit_AA_mask,
                                pssm_coef=pssm_coef,
                                pssm_bias=pssm_bias,
                                pssm_multi=self.pssm_multi,
                                pssm_log_odds_flag=bool(self.pssm_log_odds_flag),
                                pssm_log_odds_mask=pssm_log_odds_mask,
                                pssm_bias_flag=bool(self.pssm_bias_flag),
                                tied_pos=tied_pos_list_of_lists_list[0],
                                tied_beta=tied_beta,
                                bias_by_res=bias_by_res_all,
                            )
                            # Compute scores
                            S_sample = sample_dict["S"]
                        log_probs = self.model(
                            X,
                            S_sample,
                            mask,
                            chain_M * chain_M_pos,
                            residue_idx,
                            chain_encoding_all,
                            randn_2,
                            use_input_decoding_order=True,
                            decoding_order=sample_dict["decoding_order"],
                        )
                        mask_for_loss = mask * chain_M * chain_M_pos
                        scores = _scores(S_sample, log_probs, mask_for_loss)
                        scores = scores.cpu().data.numpy()
                        all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                        all_log_probs_list.append(log_probs.cpu().data.numpy())
                        S_sample_list.append(S_sample.cpu().data.numpy())
                        for b_ix in range(BATCH_COPIES):
                            masked_chain_length_list = masked_chain_length_list_list[
                                b_ix
                            ]
                            masked_list = masked_list_list[b_ix]
                            seq_recovery_rate = torch.sum(
                                torch.sum(
                                    torch.nn.functional.one_hot(S[b_ix], 21)
                                    * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                                    axis=-1,
                                )
                                * mask_for_loss[b_ix]
                            ) / torch.sum(mask_for_loss[b_ix])
                            seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                            sequences.append(seq)

                            score = scores[b_ix]
                            score_list.append(score)
                            native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                            if b_ix == 0 and j == 0 and temp == temperatures[0]:
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(native_seq[start:end])
                                    start = end
                                native_seq = "".join(
                                    list(np.array(list_of_AAs)[np.argsort(masked_list)])
                                )
                                l0 = 0
                                for mc_length in list(
                                    np.array(masked_chain_length_list)[
                                        np.argsort(masked_list)
                                    ]
                                )[:-1]:
                                    l0 += mc_length
                                    native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                                    l0 += 1
                                sorted_masked_chain_letters = np.argsort(
                                    masked_list_list[0]
                                )
                                print_masked_chains = [
                                    masked_list_list[0][i]
                                    for i in sorted_masked_chain_letters
                                ]
                                sorted_visible_chain_letters = np.argsort(
                                    visible_list_list[0]
                                )
                                print_visible_chains = [
                                    visible_list_list[0][i]
                                    for i in sorted_visible_chain_letters
                                ]
                                native_score_print = np.format_float_positional(
                                    np.float32(native_score.mean()),
                                    unique=False,
                                    precision=4,
                                )
                                line = ">{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n".format(
                                    name_,
                                    native_score_print,
                                    print_visible_chains,
                                    print_masked_chains,
                                    self.model_name,
                                    native_seq,
                                )
                                if verbose:
                                    print(line.rstrip())
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(seq[start:end])
                                start = end

                            seq = "".join(
                                list(np.array(list_of_AAs)[np.argsort(masked_list)])
                            )
                            l0 = 0
                            for mc_length in list(
                                np.array(masked_chain_length_list)[
                                    np.argsort(masked_list)
                                ]
                            )[:-1]:
                                l0 += mc_length
                                seq = seq[:l0] + "/" + seq[l0:]
                                l0 += 1
                            score_print = np.format_float_positional(
                                np.float32(score), unique=False, precision=4
                            )
                            seq_rec_print = np.format_float_positional(
                                np.float32(seq_recovery_rate.detach().cpu().numpy()),
                                unique=False,
                                precision=4,
                            )
                            line = ">T={}, sample={}, score={}, seq_recovery={}\n{}\n".format(
                                temp, b_ix, score_print, seq_rec_print, seq
                            )
                            if verbose:
                                print(line.rstrip())

            all_probs_concat = np.concatenate(all_probs_list)
            all_log_probs_concat = np.concatenate(all_log_probs_list)
            S_sample_concat = np.concatenate(S_sample_list)

            best_seq = S_sample_concat[np.argmin(score_list)]
            best_seq = _S_to_seq(best_seq, chain_M[0])

            outdict = {
                "all_probs_concat": all_probs_concat,
                "all_log_probs_concat": all_log_probs_concat,
                "S_sample_concat": S_sample_concat,
                "score_list": np.array(score_list),
                "sequences": np.array(sequences),
                "best_seq": best_seq,
            }

            if return_outdict:
                return outdict
            else:
                return best_seq

    def inverse_fold_batch(
        self, write_on_the_fly=False
    ) -> T.Union[T.List[str], T.List[T.Dict[str, np.ndarray]]]:
        pdb_paths = glob.glob(str(self.pdb_dir / "*.pdb"))
        sequences = []
        headers = []

        for pdb_path in tqdm(pdb_paths):
            header = Path(pdb_path).stem
            seq = self.inverse_fold(
                pdb_path, return_outdict=False, verbose=self.verbose
            )

            sequences.append(seq)
            headers.append(header)

            if write_on_the_fly:
                ensure_exists(self.output_fasta_path)
                with open(self.output_fasta_path, "a") as f:
                    f.write(f">{header}\n{seq}\n")

        return sequences, headers

    def run(self):
        sequences, headers = self.inverse_fold_batch(write_on_the_fly=True)
        return sequences, headers
