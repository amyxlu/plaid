import os
import re
from contextlib import contextmanager
from functools import lru_cache, reduce
import hashlib
import math
from pathlib import Path
import shutil
import threading
import urllib

import safetensors
import torch
import typing as T
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PathLike = T.Union[str, Path]
ArrayLike = T.Union[np.ndarray, torch.Tensor]


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """Downloads a file if it does not exist, optionally checking its SHA-256 hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, "wb") as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest != file_digest:
            raise OSError(f"hash of {path} (url: {url}) failed to validate")
    return path


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


stratified_settings = threading.local()


@contextmanager
def enable_stratified(group=0, groups=1, disable=False):
    """A context manager that enables stratified sampling."""
    try:
        stratified_settings.disable = disable
        stratified_settings.group = group
        stratified_settings.groups = groups
        yield
    finally:
        del stratified_settings.disable
        del stratified_settings.group
        del stratified_settings.groups


@contextmanager
def enable_stratified_accelerate(accelerator, disable=False):
    """A context manager that enables stratified sampling, distributing the strata across
    all processes and gradient accumulation steps using settings from Hugging Face Accelerate.
    """
    try:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        acc_steps = accelerator.gradient_state.num_steps
        acc_step = accelerator.step % acc_steps
        group = rank * acc_steps + acc_step
        groups = world_size * acc_steps
        with enable_stratified(group, groups, disable=disable):
            yield
    finally:
        pass


def stratified_with_settings(shape, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution, using settings from a context
    manager."""
    if not hasattr(stratified_settings, "disable") or stratified_settings.disable:
        return torch.rand(shape, dtype=dtype, device=device)
    return stratified_uniform(
        shape,
        stratified_settings.group,
        stratified_settings.groups,
        dtype=dtype,
        device=device,
    )


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        stratified_with_settings(shape, device=device, dtype=torch.float64)
        * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device="cpu", dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (
        stratified_with_settings(shape, device=device, dtype=dtype)
        * (max_value - min_value)
        + min_value
    ).exp()


def rand_v_diffusion(
    shape,
    sigma_data=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = (
        stratified_with_settings(shape, device=device, dtype=dtype)
        * (max_cdf - min_cdf)
        + min_cdf
    )
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_cosine_interpolated(
    shape,
    image_d,
    noise_d_low,
    noise_d_high,
    sigma_data=1.0,
    min_value=1e-3,
    max_value=1e3,
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(
        t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max
    ):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max
        )
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max
        )
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_with_settings(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max
    )
    return torch.exp(-logsnr / 2) * sigma_data


def rand_split_log_normal(
    shape, loc, scale_1, scale_2, device="cpu", dtype=torch.float32
):
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


# def dct(x):
#     if x.ndim == 3:
#         return df.dct(x)
#     if x.ndim == 4:
#         return df.dct2(x)
#     if x.ndim == 5:
#         return df.dct3(x)
#     raise ValueError(f"Unsupported dimensionality {x.ndim}")


@lru_cache
def freq_weight_1d(n, scales=0, dtype=None, device=None):
    ramp = torch.linspace(0.5 / n, 0.5, n, dtype=dtype, device=device)
    weights = -torch.log2(ramp)
    if scales >= 1:
        weights = torch.clamp_max(weights, scales)
    return weights


@lru_cache
def freq_weight_nd(shape, scales=0, dtype=None, device=None):
    indexers = [
        [slice(None) if i == j else None for j in range(len(shape))]
        for i in range(len(shape))
    ]
    weights = [
        freq_weight_1d(n, scales, dtype, device)[ix] for n, ix in zip(shape, indexers)
    ]
    return reduce(torch.minimum, weights)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """A context manager that sets whether TF32 is allowed on cuDNN or matmul."""
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old


def get_safetensors_metadata(path):
    """Retrieves the metadata from a safetensors file."""
    return safetensors.safe_open(path, "pt").metadata()


def npy(x: ArrayLike):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


def to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x


def set_random_seed(seed: int) -> None:
    if seed == -1:
        # pseudorandom-ception
        seed = random.randint(0, 1000000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(
    x: ArrayLike, minv: T.Union[float, ArrayLike], maxv: T.Union[float, ArrayLike]
) -> ArrayLike:
    return (x - minv) / (maxv - minv)


def generate_square_subsequent_mask(sz: int):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


def count_parameters(model, require_grad_only=True):
    if require_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def random_hash():
    gen_hash = random.getrandbits(128)
    gen_hash = "%032x" % gen_hash
    return gen_hash


def remove_all_pdb_files(outdir: PathLike):
    outdir = Path(outdir)
    for fpath in outdir.glob("*.pdb"):
        fpath.unlink()


def timestamp():
    return datetime.now().strftime("%y%m%d_%H%M")


def write_pdb_to_disk(pdb_str, outpath):
    outpath = Path(outpath)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True, exist_ok=False)
    with open(outpath, "w") as f:
        f.write(pdb_str)
    return outpath


def save_pdb_strs_to_disk(
    pdb_strs, outdir, identifiers_list=None, save_as_single_file=False
):
    """Allows the option of saving all structures as models in a single file, useful for making animations."""
    if identifiers_list is None:
        identifiers_list = [f"protein{i}" for i in range(len(pdb_strs))]

    outdir = Path(outdir)
    outpaths = []

    if save_as_single_file:
        print("Writing all PDBs to a single file.")
        outpath = outdir / "all.pdb"
        with open(outpath, "w") as f:
            for i, (header, pdb_str) in enumerate(zip(identifiers_list, pdb_strs)):
                f.write(f"MODEL {i+1}\n")
                f.write(pdb_str)
                f.write("ENDMDL\n")
            outpaths.append(outpath)

    else:
        for i, (header, pdb_str) in enumerate(zip(identifiers_list, pdb_strs)):
            outpath = outdir / f"{header}.pdb"
            with open(outpath, "w") as f:
                f.write(pdb_str)
                outpaths.append(outpath)

    return outpaths


def write_to_fasta(sequences, outpath, headers: T.Optional[T.List[str]] = None):
    outpath = Path(outpath)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = [f"sequence_{i}" for i in range(len(sequences))]
    assert len(headers) == len(sequences)

    with open(outpath, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{headers[i]}\n")
            f.write(f"{seq}\n")
    print(f"Wrote {len(sequences)} sequences to {outpath}.")


def extract_avg_b_factor_per_residue(pdb_file: PathLike) -> T.List[float]:
    """Mostly used for OmegaFold."""
    b_factors = []

    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    b_factor = float(line[60:66].strip())
                    b_factors.append(b_factor)
                except ValueError:
                    pass

    return b_factors


def print_cuda_memory_usage():
    if torch.cuda.is_available():
        print(
            f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
        )
        print(
            f"Current CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
        )


def print_cuda_info():
    # Number of visible CUDA GPUs
    print("=" * 10, "\n")
    num_gpus = torch.cuda.device_count()
    print(f"Number of visible CUDA GPUs: {num_gpus}")

    # Current GPU ID device number
    if num_gpus > 0:
        current_gpu_id = torch.cuda.current_device()
        print(f"Current GPU ID device number: {current_gpu_id}")
    else:
        print("No CUDA GPU available.")

    print_cuda_memory_usage()

    # Number of visible CPUs
    num_cpus = torch.get_num_threads()
    print(f"Number of visible CPUs: {num_cpus}")
    print("=" * 10, "\n")


def view_py3Dmol(pdbstr):
    import py3Dmol

    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(pdbstr)
    view.setStyle({"model": -1}, {"cartoon": {"color": "green"}})
    view.zoomTo()
    view.show()


def parse_sequence_from_structure(pdb_data=None, pdb_path=None, id="") -> str:
    """Returns a string where chains are separated by : and assumes one model per PDB file"""
    from Bio.PDB import PDBParser
    import io

    if pdb_data:
        # Using StringIO to create a file-like object from the string
        pdb_io = io.StringIO(pdb_data)

        # Parsing the PDB data
        parser = PDBParser()
        structure = parser.get_structure(id, pdb_io)
    else:
        assert not pdb_path is None
        parser = PDBParser()
        with open(pdb_path, "r") as f:
            structure = parser.get_structure(id, f)

    # Extracting the sequence
    if len(structure) > 1:
        print(f"WARNING: More than one model found in {id}. Using the first model.")
    model = structure[0]
    chains = []
    for chain in model:
        sequence = ""
        for residue in chain.get_residues():
            res_name = residue.get_resname()
            sequence += restype_3to1[res_name]
        chains.append(sequence)
    return ":".join(chains)


def output_to_pdb(output: T.Dict) -> T.List[str]:
    """https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/misc.py#L93"""
    from openfold.np.protein import Protein as OFProtein
    from openfold.np.protein import to_pdb
    from openfold.utils.feats import atom14_to_atom37

    """Returns the pbd (file) string from the model given the model output."""
    # atom14_to_atom37 must be called first, as it fails on latest numpy if the
    # input is a numpy array. It will work if the input is a torch tensor.
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
            chain_index=output["chain_index"][i] if "chain_index" in output else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def build_contact_map(pdb_file, threshold=5.0):
    from Bio.PDB import PDBParser

    def calculate_distance(residue1, residue2):
        """Calculate the distance between the center of masses of two residues."""
        diff_vector = residue1["CA"].coord - residue2["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    """Build a contact map from a PDB file with a specified distance threshold."""
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Assuming we are working with the first model

    residues = [
        residue for residue in model.get_residues() if residue.get_id()[0] == " "
    ]
    n = len(residues)
    contact_map = np.zeros((n, n), dtype=int)

    for i, residue1 in enumerate(residues):
        for j, residue2 in enumerate(residues):
            if calculate_distance(residue1, residue2) < threshold:
                contact_map[i, j] = 1

    return contact_map


def pdb_path_to_biotite_atom_array(file_path):
    from biotite.structure.io import pdb

    pdb_file = pdb.PDBFile.read(file_path)
    atom_array_stack = pdb.get_structure(pdb_file)
    return atom_array_stack


# Filter for alpha carbon atoms
def alpha_carbons_from_atom_array(atom_array):
    return atom_array[atom_array.atom_name == "CA"]


def outputs_to_avg_metric(outputs):
    avg_metrics = {}
    metrics_to_log = [
        "plddt",
        "ptm",
        "aligned_confidence_probs",
        "predicted_aligned_error",
    ]

    for metric in metrics_to_log:
        value = npy(outputs[metric])

        if value.ndim == 1:
            median = value
        elif value.ndim == 2:
            median = np.median(value, axis=1)
        else:
            assert value.ndim > 2
            median = np.median(value, axis=tuple(range(1, value.ndim)))

        avg_metrics[metric] = median

    return avg_metrics


def calc_sequence_recovery(
    pred_seq: ArrayLike, orig_seq: ArrayLike, mask: T.Optional[ArrayLike] = None
):
    if isinstance(pred_seq[0], str):
        assert isinstance(orig_seq[0], str)
        pred_seq = np.array([ord(x) for x in pred_seq])
        orig_seq = np.array([ord(x) for x in orig_seq])

    if not mask is None:
        pred_seq, orig_seq = pred_seq[mask], orig_seq[mask]

    assert len(pred_seq) == len(orig_seq)
    return np.sum(npy(pred_seq) == npy(orig_seq)) / len(pred_seq)


def calc_sequence_identity(query_seq, target_seq):
    """Given two sequences, which could be of different lengths,
    aligns the sequences with global alignment and returns the sequence identity
    as a global alignment score divided by the query sequence."""
    from Bio import pairwise2

    alignments = pairwise2.align.globalxx(query_seq, target_seq)
    similarity = alignments[0].score
    return similarity / len(query_seq)


def read_sequences_from_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, "r") as f:
        for line in f.readlines():
            if line[0] == ">":
                header = line.rstrip("\n")[1:]
            else:
                sequence = line.rstrip("\n")
                sequences[header] = sequence
    return sequences


def find_latest_checkpoint(folder):
    checkpoint_files = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    checkpoint_files = list(filter(lambda x: "EMA" not in x, checkpoint_files))
    if not checkpoint_files:
        return None

    latest_checkpoint = max(checkpoint_files, key=lambda x: extract_step(x))
    return latest_checkpoint


def extract_step(checkpoint_file):
    match = re.search(r"(\d+)-(\d+)\.ckpt", checkpoint_file)
    if match:
        return int(match.group(2))
    return -1


def round_to_multiple(x, multiple):
    return int(multiple * round(x / multiple))


def filename_to_suffix_number(path):
    x = str(path).split("/")[-1]
    return int(re.findall(r"\d+", x)[-1])


def sort_by_suffix(lst):
    import re

    sorted_list = sorted(lst, key=lambda path: filename_to_suffix_number(path))
    return sorted_list


def get_pfam_length(pfam_id, full_pfam_hmm_file):
    idx = full_pfam_hmm_file.find(pfam_id)

    if idx != -1:
        # Extract lines in that neighborhood
        subcontent = full_pfam_hmm_file[idx : idx + 1000]

        # filter for lines with the length identifier
        lines = subcontent.split("\n")
        line = list(filter(lambda line: "#=GF ML" in line, lines))[0]

        if "#=GF ML" in line:
            length = int(line.split()[-1])

    return length


######
# The following is adapted from pseudocode in Chen et al.,
# https://arxiv.org/abs/2301.10972
######


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.0)


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)
