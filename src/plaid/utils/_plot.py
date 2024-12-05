# from colabfold

from pathlib import Path
import os
import re
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

from ._misc import read_sequences_from_fasta


def plot_predicted_alignment_error(
    jobname: str, num_models: int, outs: dict, result_dir: Path, show: bool = False
):
    plt.figure(figsize=(3 * num_models, 2), dpi=100)
    for n, (model_name, value) in enumerate(outs.items()):
        plt.subplot(1, num_models, n + 1)
        plt.title(model_name)
        plt.imshow(value["pae"], label=model_name, cmap="bwr", vmin=0, vmax=30)
        plt.colorbar()
    plt.savefig(result_dir.joinpath(jobname + "_PAE.png"))
    if show:
        plt.show()
    plt.close()


def plot_msa_v2(feature_dict, sort_lines=True, dpi=100):
    seq = feature_dict["msa"][0]
    if "asym_id" in feature_dict:
        Ls = [0]
        k = feature_dict["asym_id"][0]
        for i in feature_dict["asym_id"]:
            if i == k:
                Ls[-1] += 1
            else:
                Ls.append(1)
            k = i
    else:
        Ls = [len(seq)]
    Ln = np.cumsum([0] + Ls)

    try:
        N = feature_dict["num_alignments"][0]
    except:
        N = feature_dict["num_alignments"]

    msa = feature_dict["msa"][:N]
    gap = msa != 21
    qid = msa == seq
    gapid = np.stack([gap[:, Ln[i] : Ln[i + 1]].max(-1) for i in range(len(Ls))], -1)
    lines = []
    Nn = []
    for g in np.unique(gapid, axis=0):
        i = np.where((gapid == g).all(axis=-1))
        qid_ = qid[i]
        gap_ = gap[i]
        seqid = np.stack(
            [qid_[:, Ln[i] : Ln[i + 1]].mean(-1) for i in range(len(Ls))], -1
        ).sum(-1) / (g.sum(-1) + 1e-8)
        non_gaps = gap_.astype(float)
        non_gaps[non_gaps == 0] = np.nan
        if sort_lines:
            lines_ = non_gaps[seqid.argsort()] * seqid[seqid.argsort(), None]
        else:
            lines_ = non_gaps[::-1] * seqid[::-1, None]
        Nn.append(len(lines_))
        lines.append(lines_)

    Nn = np.cumsum(np.append(0, Nn))
    lines = np.concatenate(lines, 0)
    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.title("Sequence coverage")
    plt.imshow(
        lines,
        interpolation="nearest",
        aspect="auto",
        cmap="rainbow_r",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(0, lines.shape[1], 0, lines.shape[0]),
    )
    for i in Ln[1:-1]:
        plt.plot([i, i], [0, lines.shape[0]], color="black")
    for j in Nn[1:-1]:
        plt.plot([0, lines.shape[1]], [j, j], color="black")

    plt.plot((np.isnan(lines) == False).sum(0), color="black")
    plt.xlim(0, lines.shape[1])
    plt.ylim(0, lines.shape[0])
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    return plt


def plot_msa(msa, query_sequence, seq_len_list, total_seq_len, dpi=100):
    # gather MSA info
    prev_pos = 0
    msa_parts = []
    Ln = np.cumsum(np.append(0, [len for len in seq_len_list]))
    for id, l in enumerate(seq_len_list):
        chain_seq = np.array(query_sequence[prev_pos : prev_pos + l])
        chain_msa = np.array(msa[:, prev_pos : prev_pos + l])
        seqid = np.array(
            [
                np.count_nonzero(chain_seq == msa_line[prev_pos : prev_pos + l])
                / len(chain_seq)
                for msa_line in msa
            ]
        )
        non_gaps = (chain_msa != 21).astype(float)
        non_gaps[non_gaps == 0] = np.nan
        msa_parts.append((non_gaps[:] * seqid[:, None]).tolist())
        prev_pos += l
    lines = []
    lines_to_sort = []
    prev_has_seq = [True] * len(seq_len_list)
    for line_num in range(len(msa_parts[0])):
        has_seq = [True] * len(seq_len_list)
        for id in range(len(seq_len_list)):
            if np.sum(~np.isnan(msa_parts[id][line_num])) == 0:
                has_seq[id] = False
        if has_seq == prev_has_seq:
            line = []
            for id in range(len(seq_len_list)):
                line += msa_parts[id][line_num]
            lines_to_sort.append(np.array(line))
        else:
            lines_to_sort = np.array(lines_to_sort)
            lines_to_sort = lines_to_sort[np.argsort(-np.nanmax(lines_to_sort, axis=1))]
            lines += lines_to_sort.tolist()
            lines_to_sort = []
            line = []
            for id in range(len(seq_len_list)):
                line += msa_parts[id][line_num]
            lines_to_sort.append(line)
        prev_has_seq = has_seq
    lines_to_sort = np.array(lines_to_sort)
    lines_to_sort = lines_to_sort[np.argsort(-np.nanmax(lines_to_sort, axis=1))]
    lines += lines_to_sort.tolist()

    # Nn = np.cumsum(np.append(0, Nn))
    # lines = np.concatenate(lines, 1)
    xaxis_size = len(lines[0])
    yaxis_size = len(lines)

    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.title("Sequence coverage")
    plt.imshow(
        lines[::-1],
        interpolation="nearest",
        aspect="auto",
        cmap="rainbow_r",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(0, xaxis_size, 0, yaxis_size),
    )
    for i in Ln[1:-1]:
        plt.plot([i, i], [0, yaxis_size], color="black")
    # for i in Ln_dash[1:-1]:
    #    plt.plot([i, i], [0, lines.shape[0]], "--", color="black")
    # for j in Nn[1:-1]:
    #    plt.plot([0, lines.shape[1]], [j, j], color="black")

    plt.plot((np.isnan(lines) == False).sum(0), color="black")
    plt.xlim(0, xaxis_size)
    plt.ylim(0, yaxis_size)
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")

    return plt


def show_pdb(
    pdb_str,
    show_sidechains=False,
    show_mainchains=False,
    color="pLDDT",
    chains=None,
    vmin=50,
    vmax=90,
    size=(800, 480),
    hbondCutoff=4.0,
    Ls=None,
    animate=False,
):
    """https://github.com/sokrypton/ColabFold/blob/main/ESMFold.ipynb"""

    if chains is None:
        chains = 1 if Ls is None else len(Ls)
    view = py3Dmol.view(
        js="https://3dmol.org/build/3Dmol.js", width=size[0], height=size[1]
    )
    if animate:
        view.addModelsAsFrames(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})
    else:
        view.addModel(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})
    if color == "pLDDT":
        view.setStyle(
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": vmin,
                        "max": vmax,
                    }
                }
            }
        )
    elif color == "rainbow":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif color == "chain":
        raise NotImplementedError

    if show_sidechains:
        BB = ["C", "O", "N"]
        view.addStyle(
            {
                "and": [
                    {"resn": ["GLY", "PRO"], "invert": True},
                    {"atom": BB, "invert": True},
                ]
            },
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
            {"sphere": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "PRO"}, {"atom": ["C", "O"], "invert": True}]},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
    if show_mainchains:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB}, {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}}
        )
    view.zoomTo()
    if animate:
        view.animate()
    return view


def create_sample_id_from_designability_df(row):
    s = row.pdb_paths.split("/")[-1]
    sample_id = re.findall(r"-?\d+", s)[0]
    length = len(row.sequences)
    return f"length{length}_sample{sample_id}"


def create_sample_id_from_cluster_dict(k, v):
    sample_id = re.findall(r"-?\d+", k)[0]
    length = len(v)
    return f"length{length}_sample{sample_id}"


def load_df(sample_dir, use_designability_filter=False):
    # load the dataframe and do some column manipulations
    df = pd.read_csv(sample_dir / "designability.csv")
    df = df.rename({"ccTM": "cctm"}, axis=1)
    df = df.reset_index(drop=True)
    df = df.drop(["Unnamed: 0"], axis=1)
    df["sample_id"] = df.apply(
        lambda row: create_sample_id_from_designability_df(row), axis=1
    )
    df["lengths"] = df.sequences.str.len()

    # load the cluster dictionaries and get the unique clusters
    sequence_clusters_dict = read_sequences_from_fasta(
        sample_dir / "mmseqs_easycluster.m8_rep_seq.fasta"
    )
    if use_designability_filter:
        structure_clusters_dict = read_sequences_from_fasta(
            sample_dir / "foldseek_easycluster.m8_rep_seq.fasta"
        )
    else:
        structure_clusters_dict = read_sequences_from_fasta(
            sample_dir / "no_filter_foldseek_easycluster.m8_rep_seq.fasta"
        )

    sequence_clusters = [
        create_sample_id_from_cluster_dict(k, v)
        for (k, v) in sequence_clusters_dict.items()
    ]
    structure_clusters = [
        create_sample_id_from_cluster_dict(k, v)
        for (k, v) in structure_clusters_dict.items()
    ]

    # add a boolean to say if this sample is a representative cluster
    df["is_sequence_rep"] = df.sample_id.isin(sequence_clusters)
    df["is_structure_rep"] = df.sample_id.isin(structure_clusters)
    return df


def load_results_for_method(sample_dir, by_length=True, use_designability_filter=False):
    all_stats = {}

    if by_length:
        lengths = os.listdir(sample_dir)
        lengths.sort()
        # print("Lengths:")
        # print(lengths)

        for l in lengths:
            try:
                df = load_df(sample_dir / l, use_designability_filter)
                all_stats[l] = df

            except Exception as e:
                print(e)
                pass

        df = pd.concat(list(all_stats.values()))

    else:
        df = load_df(sample_dir, use_designability_filter)

    return df
