import math
import torch
import numpy as np
import typing as T
from torch.nn.functional import nll_loss

from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils import to_tensor


class RITAPerplexity:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            "lightonai/RITA_xl", trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

    def calc_perplexity(self, sequence):
        """Calculates the perplexity under RITA for a single model"""
        input_ids = torch.tensor(self.tokenizer.encode(sequence)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)

    def batch_eval(self, all_sequences, batch_size: int = None, *args, **kwargs):
        """Calculates the average perplexity under RITA for a batch of strings"""
        if not len(set([len(s) for s in all_sequences])) == 1:
            raise NotImplementedError(
                "Batched calculation only supports sequences of the same length at the moment."
            )

        batch_size = len(all_sequences) if not batch_size else batch_size
        all_perplexities = []
        for i in range(0, len(all_sequences), batch_size):
            sequences = all_sequences[i : i + batch_size]
            input_ids = self.tokenizer.batch_encode_plus(sequences)["input_ids"]
            input_ids = to_tensor(input_ids, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            all_perplexities.append(torch.exp(loss).item())

        return np.mean(all_perplexities)


class ESMPseudoPerplexity:
    """Follows the per-sequence definition in the Lin et al., 2022 appendix."""

    def __init__(self, device, esm_model_name: str = "esm2_t48_15B_UR50D"):
        self.device = device
        model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
        self.pad_idx = alphabet.padding_idx
        self.nlayers = int(esm_model_name.split("_")[1][1:])
        self.batch_converter = alphabet.get_batch_converter()
        self.model = model.to(device)
        self.model.eval()

    def batch_calc_perplexity(self, sequences: T.List[str]):
        labels, strs, tokens = self.batch_converter(sequences)
        B, L, _ = tokens.shape
        perplexities = []

        # at each position, replace the token with a mask token and calculate the "perplexity"
        for pos in range(len(L)):
            tokens_ = tokens.clone()
            tokens_[:, pos] = torch.where(
                tokens[:, pos] == self.pad_idx, self.pad_idx, self.mask_idx
            )
            with torch.no_grad():
                results = self.model(
                    tokens_.to(self.device),
                    repr_layers=[self.nlayers - 1],
                    return_contacts=False,
                )
                nll = nll_loss(
                    results["logits"], labels.to(self.device), ignore_index=self.pad_idx
                )
                perplexities.append(torch.exp(nll).item())
        return perplexities
