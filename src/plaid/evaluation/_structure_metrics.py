import torch
from biotite.structure import rmspd
from biotite.structure.io import pdb
import numpy as np
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor
from glob import glob


def pdb_path_to_biotite_atom_array(file_path):
    """Returns an atom array stack - if only one chain, can take just the first of the tuple to get the atom array."""
    pdb_file = pdb.PDBFile.read(file_path)
    atom_array = pdb.get_structure(pdb_file)
    return atom_array


"""
RMSPD calculation from biotite
"""


def rmspd_from_pdb_paths(path1, path2):
    aarray_recons = pdb_path_to_biotite_atom_array(path1)
    aarray_orig = pdb_path_to_biotite_atom_array(path2)
    return rmspd(aarray_recons, aarray_orig)


def batch_rmspd_from_pdb_paths(pdb_paths1, pdb_paths2):
    all_rmspds = []
    for a, b in zip(pdb_paths1, pdb_paths2):
        result = rmspd_from_pdb_paths(a, b)
        all_rmspds.append(result[0])
        # print(result)
    # print("mean: ", np.mean(all_rmspds))
    # print("median: ", np.median(all_rmspds))
    return all_rmspds


# https://github.com/RosettaCommons/RFDesign/blob/98f7435944068f0b8a864eef3029a0bad8e530ca/hallucination/util/metrics.py


def lDDT(ca0, ca, s=0.001):
    """smooth lDDT:
    s=0.35  - good for training (smooth)
    s=0.001 - (or smaller) good for testing
    """
    L = ca0.shape[0]
    # Computes batched the p-norm distance between each pair of the two collections of row vectors.
    d0 = torch.cdist(ca0, ca0)
    d0 = d0 + 999.9 * torch.eye(L, device=ca0.device)  # exclude diagonal
    i, j = torch.where(d0 < 15.0)
    d = torch.cdist(ca, ca)
    dd = torch.abs(d0[i, j] - d[i, j]) + 1e-3

    def f(x, m, s):
        return 0.5 * torch.erf((torch.log(dd) - np.log(m)) / (s * 2**0.5)) + 0.5

    lddt = torch.stack([f(dd, m, s) for m in [0.5, 1.0, 2.0, 4.0]], dim=-1).mean()
    return 1.0 - lddt


def RMSD(P, Q):
    """Kabsch algorthm"""

    def rmsd(V, W):
        return torch.sqrt(torch.sum((V - W) * (V - W)) / len(V))

    def centroid(X):
        return X.mean(axis=0)

    cP = centroid(P)
    cQ = centroid(Q)
    P = P - cP
    Q = Q - cQ

    # Computation of the covariance matrix
    C = torch.mm(P.T, Q)

    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3, 3], device=P.device)
    d[:, -1] = torch.sign(torch.det(V) * torch.det(W))

    # Rotation matrix U
    U = torch.mm(d * V, W.T)

    # Rotate P
    rP = torch.mm(P, U)

    # get RMS
    rms = rmsd(rP, Q)

    return rms  # , rP


def KL(P, Q, eps=1e-6):
    """KL-divergence between two sets of 6D coords"""
    kl = [
        (Pi * torch.log((Pi + eps) / (Qi + eps))).sum(0).mean() for Pi, Qi in zip(P, Q)
    ]
    kl = torch.stack(kl).mean()
    return kl


# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/validation_metrics.py
#
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def drmsd(structure_1, structure_2, mask=None):
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d**2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd**2
    if mask is not None:
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.0)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if mask is not None:
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)

    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1.0, 2.0, 4.0, 8.0])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1.0, 2.0, 4.0])


def calculate_rmsd(pdb_path_1, pdb_path_2, ca_only=True):
    """Aligns and calculates the RMSD for the first chain in the first model of each PDB path."""
    # Initialize the PDB parser
    parser = PDB.PDBParser(QUIET=True)

    # Load the PDB structures
    structure1 = parser.get_structure("structure1", pdb_path_1)
    structure2 = parser.get_structure("structure2", pdb_path_2)

    # Select the chains
    chain1 = next(structure1[0].get_chains())
    chain2 = next(structure2[0].get_chains())

    # Extract the alpha carbons (CA) for alignment
    if ca_only:
        atoms1 = [atom for atom in chain1.get_atoms() if atom.get_id() == "CA"]
        atoms2 = [atom for atom in chain2.get_atoms() if atom.get_id() == "CA"]
    else:
        atoms1 = list(chain1.get_atoms())
        atoms2 = list(chain2.get_atoms())

    # Ensure both chains have the same number of alpha carbons
    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"CA atom count mismatch between {pdb_path_1} and {pdb_path_2}. Cannot calculate RMSD."
        )

    # Perform the superimposition
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)

    # Calculate RMSD
    rmsd = super_imposer.rms

    return rmsd


def process_pdb_pair(pdb_pair):
    pdb_path_1, pdb_path_2, ca_only = pdb_pair
    return calculate_rmsd(pdb_path_1, pdb_path_2, ca_only=ca_only)


def batch_rmsd_calculation(pdb_paths1, pdb_paths2, ca_only=True):
    assert len(pdb_paths1) == len(pdb_paths2), "The number of PDB paths must be equal."
    ca_only = [ca_only] * len(pdb_paths1)
    pdb_pairs = zip(pdb_paths1, pdb_paths2, ca_only)
    results = []

    # Use ProcessPoolExecutor to parallelize the RMSD calculations
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_pdb_pair, pdb_pairs):
            results.append(result)

    return results
