"""
From https://github.com/microsoft/foldingdiff/blob/main/foldingdiff/tmalign.py
Short and easy wrapper for TMalign
"""

import os
import re
import itertools
import shutil
import subprocess
import multiprocessing
import logging
from typing import List, Tuple

import numpy as np
import io
from pathlib import Path


def write_tmp_file(data, fname) -> str:
    outpath = Path(os.environ["HOME"]) / ".tmp" / fname
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)
    with open(outpath, "w") as f:
        f.write(data)
    return str(outpath)


def run_tmalign(
    query: str, reference: str, fast: bool = False, delete_tmp=True, verbose=False
) -> float:
    """
    Run TMalign on the two given input pdb files
    """
    uses_tmp_files = False
    if not os.path.isfile(query):
        query = write_tmp_file(query, "query.pdb")
        if verbose:
            print("saved query to", query)
        uses_tmp_files = True
    if not os.path.isfile(reference):
        reference = write_tmp_file(reference, "reference.pdb")
        if verbose:
            print("saved reference to", reference)
        uses_tmp_files = True

    # Check if TMalign is installed
    exec = shutil.which("TMalign")
    if not exec:
        raise FileNotFoundError("TMalign not found in PATH")

    # Build the command
    cmd = f"{exec} {query} {reference}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"Tmalign failed on {query}|{reference}, returning NaN")
        return np.nan

    # Parse the output
    score_lines = []
    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            score_lines.append(line)

    if uses_tmp_files and delete_tmp:
        try:
            os.remove(query)
            os.remove(reference)
        except:
            pass
    # Fetch the chain number
    key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    return results_dict["Chain_2"]  # Normalize by reference length


def max_tm_across_refs(
    query: str,
    references: List[str],
    n_threads: int = multiprocessing.cpu_count(),
    fast: bool = True,
    chunksize: int = 10,
    parallel: bool = True,
) -> Tuple[float, str]:
    """
    Compare the query against each of the references in parallel and return the maximum score
    along with the corresponding reference
    This is typically a lot of comparisons so we run with fast set to True by default
    """
    logging.debug(
        f"Matching against {len(references)} references using {n_threads} workers with fast={fast}"
    )
    args = [(query, ref, fast) for ref in references]
    if parallel:
        n_threads = min(n_threads, len(references))
        pool = multiprocessing.Pool(n_threads)
        values = list(pool.starmap(run_tmalign, args, chunksize=chunksize))
        pool.close()
        pool.join()
    else:
        values = list(itertools.starmap(run_tmalign, args))

    return np.nanmax(values), references[np.argmax(values)]


def main():
    """On the fly testing"""
    run_tmalign("data/7PFL.pdb", "data/7ZYA.pdb")


if __name__ == "__main__":
    main()
