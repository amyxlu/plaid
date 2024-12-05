""" Calculate additional protein attributes from sequence for conditioning.
"""

import os
import multiprocessing as mp
from functools import partial

import numpy as np
import typing as T
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import PDB

from plaid.constants import RESTYPES


# Calculated using a random subset of 40399 UniRef90 sequences,
# subcropped to 128 amino acids, and uses the bin edges and fitted using
# sklearn.preprocessing.KBinsDiscretizer with n_bins=6, encode='ordinal', strategy='quantile'
# (see training/uniref/plot_uniref_chunks_len128.ipynb)

SS_BOUNDARIES = {
    "uniref": {
        "helix": [0.1647, 0.3294, 0.4941, 0.6589, 0.8236],
        "turn": [0.1628, 0.3255, 0.4883, 0.6510, 0.8138],
        "sheet": [0.1524, 0.3048, 0.4571, 0.6095, 0.7619],
    },
    "cath": {
        "helix": [0.2109375, 0.25, 0.2734375, 0.296875, 0.3203125],
        "turn": [0.171875, 0.2109375, 0.234375, 0.265625, 0.3046875],
        "sheet": [0.1875, 0.2265625, 0.2578125, 0.296875, 0.3359375],
    },
}


DEFAULT_PROPERTIES = [
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "charge_at_pH",
]


def _quantize_frac(arr: np.ndarray, boundaries: np.ndarray):
    """
    0: < arr[0]
    1: arr[0] to arr[1]
    2: ...
    3: ...
    4: ...
    5: > arr[-1]
    """
    return np.digitize(arr, boundaries)


def _quantize_secondary_structure_fractions(fracs, origin_dataset: str = "uniref"):
    helix_fracs, turn_fracs, sheet_fracs = fracs[:, 0], fracs[:, 1], fracs[:, 2]
    helix_quantized = _quantize_frac(
        helix_fracs, SS_BOUNDARIES[origin_dataset]["helix"]
    )
    turn_quantized = _quantize_frac(turn_fracs, SS_BOUNDARIES[origin_dataset]["turn"])
    sheet_quantized = _quantize_frac(
        sheet_fracs, SS_BOUNDARIES[origin_dataset]["sheet"]
    )
    return np.stack([helix_quantized, turn_quantized, sheet_quantized], axis=1)


def _sequence_to_secondary_structure_frac(
    sequence: str,
) -> T.Tuple[float, float, float]:
    sequence = str(sequence).replace(">", "")
    sequence = sequence.replace("<", "")
    X = ProteinAnalysis(str(sequence))
    return X.secondary_structure_fraction()


def sequences_to_secondary_structure_fracs(
    sequences: T.List[str], quantized: bool = True, origin_dataset: str = "uniref"
) -> T.List[T.Tuple[float, float, float]]:
    fracs = [_sequence_to_secondary_structure_frac(s) for s in sequences]
    fracs = np.stack(fracs, axis=0)
    if quantized:
        return _quantize_secondary_structure_fractions(fracs, origin_dataset)
    else:
        return np.array(fracs)


"""
Per-sequence protein properties
"""


def protein_property(protein_sequence, prop):
    protein_sequence = "".join(
        list(filter(lambda char: char in RESTYPES, protein_sequence))
    )
    analyzer = ProteinAnalysis(protein_sequence)
    if prop == "charge_at_pH":
        return getattr(analyzer, prop)(pH=7)
    else:
        return getattr(analyzer, prop)()


def calculate_df_protein_property(
    df, sequence_col="sequences", properties=DEFAULT_PROPERTIES
):
    for prop in properties:
        df[prop] = df[sequence_col].map(lambda seq: protein_property(seq, prop))
    return df


def process_chunk(chunk, prop, sequence_col):
    return chunk[sequence_col].map(lambda seq: protein_property(seq, prop))


def calculate_df_protein_property_mp(
    df, sequence_col="sequences", properties=DEFAULT_PROPERTIES
):
    num_processes = min(mp.cpu_count(), df.shape[0])

    for prop in properties:
        chunk_size = len(df) // num_processes  # Size of each chunk
        chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        pool = mp.Pool(processes=num_processes)
        fn = partial(process_chunk, prop=prop, sequence_col=sequence_col)
        results = pool.map(fn, chunks)
        pool.close()
        pool.join()
        df[prop] = pd.concat(results)

    return df


def analyze_secondary_structure(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    dssp = PDB.DSSP(structure[0], pdb_file)

    secondary_structure = {"Helix": 0, "Sheet": 0, "Turn": 0, "Other": 0}

    for key in dssp.keys():
        ss = dssp[key][2]
        if ss == "H" or ss == "G" or ss == "I":
            secondary_structure["Helix"] += 1
        elif ss == "E" or ss == "B":
            secondary_structure["Sheet"] += 1
        elif ss == "T" or ss == "S":
            secondary_structure["Turn"] += 1
        else:
            secondary_structure["Other"] += 1

    total_residues = sum(secondary_structure.values())

    result = {
        "file": pdb_file,
        "total_residues": total_residues,
        "Helix": secondary_structure["Helix"],
        "Helix_percentage": (secondary_structure["Helix"] / total_residues) * 100,
        "Sheet": secondary_structure["Sheet"],
        "Sheet_percentage": (secondary_structure["Sheet"] / total_residues) * 100,
        "Turn": secondary_structure["Turn"],
        "Turn_percentage": (secondary_structure["Turn"] / total_residues) * 100,
        "Other": secondary_structure["Other"],
        "Other_percentage": (secondary_structure["Other"] / total_residues) * 100,
    }

    return result


def analyze_secondary_structure_mp(pdb_files, num_cpus):
    """
    Example usage:

    pdb_files = ['file1.pdb', 'file2.pdb', 'file3.pdb']  # Replace with your list of PDB file paths
    num_cpus = os.cpu_count()  # Use all available CPUs, or set to a specific number
    results = analyze_multiple_pdb_files(pdb_files, num_cpus)

    for result in results:
        print(f"File: {result['file']}")
        print(f"Total residues: {result['total_residues']}")
        print(f"Helix: {result['Helix']} ({result['Helix_percentage']:.2f}%)")
        print(f"Sheet: {result['Sheet']} ({result['Sheet_percentage']:.2f}%)")
        print(f"Turn: {result['Turn']} ({result['Turn_percentage']:.2f}%)")
        print(f"Other: {result['Other']} ({result['Other_percentage']:.2f}%)\n")
    """
    with mp.Pool(num_cpus) as pool:
        results = pool.map(analyze_secondary_structure, pdb_files)
    return results
