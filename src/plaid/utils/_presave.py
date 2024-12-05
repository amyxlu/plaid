import time
import torch

from ..transforms import get_random_sequence_crop_batch
from ._misc import get_model_device


def make_embedder(lm_embedder_type):
    start = time.time()
    print(f"making {lm_embedder_type}...")

    if "esmfold" in lm_embedder_type:
        # from plaid.denoisers.esmfold import ESMFold
        from plaid.esmfold import esmfold_v1

        embedder = esmfold_v1()
        alphabet = None
    else:
        print("loading LM from torch hub")
        embedder, alphabet = torch.hub.load(
            "facebookresearch/esm:main", lm_embedder_type
        )

    embedder = embedder.eval().to("cuda")

    for param in embedder.parameters():
        param.requires_grad = False

    end = time.time()
    print(f"done loading model in {end - start:.2f} seconds.")

    return embedder, alphabet


def embed_batch_esmfold(
    esmfold, sequences, max_len=512, embed_result_key="s", return_seq_lens=True
):
    with torch.no_grad():
        # don't disgard short sequences since we're also saving headers
        sequences = get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=0
        )
        seq_lens = [len(seq) for seq in sequences]
        embed_results = esmfold.infer_embedding(sequences, return_intermediates=True)
        feats = embed_results[embed_result_key].detach()
        masks = embed_results["mask"].detach()
        seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)
    if return_seq_lens:
        return feats, seq_lens, sequences
    else:
        return feats, masks, sequences


def embed_batch_esm(
    embedder, sequences, batch_converter, repr_layer, max_len=512, return_seq_lens=True
):
    sequences = get_random_sequence_crop_batch(sequences, max_len=max_len, min_len=0)
    seq_lens = [len(seq) for seq in sequences]
    seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)

    batch = [("", seq) for seq in sequences]
    _, _, tokens = batch_converter(batch)
    device = get_model_device(embedder)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = embedder(tokens, repr_layers=[repr_layer], return_contacts=False)
        feats = results["representations"][repr_layer]

    if return_seq_lens:
        return feats, seq_lens, sequences
    else:
        return feats, masks, sequences
