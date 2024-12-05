import os
from pathlib import Path
import subprocess
import shutil


# query,target,evalue,gapopen,pident,fident,nident,qstart,qend,qlen
# tstart,tend,tlen,alnlen,raw,bits,cigar,qseq,tseq,qheader,theader,qaln,taln,mismatch,qcov,tcov
# qset,qsetid,tset,tsetid,taxid,taxname,taxlineage,
# lddt,lddtfull,qca,tca,t,u,qtmscore,ttmscore,alntmscore,rmsd,prob
# complexqtmscore,complexttmscore,complexu,complext,complexassignid

EASY_SEARCH_OUTPUT_COLS = [
    "query",
    "target",
    "theader",
    "evalue",
    "alntmscore",
    "rmsd",
    "lddt",
    "prob",
    "taxid",
    "taxname",
    "taxlineage",
]


def foldseek_easysearch(
    sample_dir: str,
    structure_subdir_name="designable",
    output_file_name="foldseek_easysearch.m8",
):
    sample_dir = Path(sample_dir)
    structures_dir = sample_dir / structure_subdir_name
    output_file = sample_dir / output_file_name
    tmp_dir = sample_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # PDB_DATABASE_PATH = "/data/bucket/robins21/pdb"
    PDB_DATABASE_PATH = os.environ.get("PDB_DATABASE_PATH")
    if PDB_DATABASE_PATH is None:
        raise ValueError(
            "PDB_DATABASE_PATH environment variable not set. Please use"
            "`foldseek databases PDB <o:db> <tmpDir>` to download the PDB database"
            "and set the PDB_DATABASE_PATH environment variable to the output path <o:db>."
        )

    try:
        cmd = [
            "foldseek",
            "easy-search",
            str(structures_dir),
            PDB_DATABASE_PATH,
            str(output_file),
            str(tmp_dir),
            # "--alignment-type",
            # "1",  # TM mode align
            "--format-output",
            " ".join(EASY_SEARCH_OUTPUT_COLS),
        ]
        subprocess.run(cmd)

    except:
        shutil.rmtree(tmp_dir)

    return output_file


def foldseek_easycluster(
    sample_dir: str,
    structure_subdir_name="designable",
    output_file_name="foldseek_easycluster.m8",
):
    structures_dir = sample_dir / structure_subdir_name
    output_file = sample_dir / output_file_name
    tmp_dir = sample_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    try:
        cmd = [
            "foldseek",
            "easy-cluster",
            str(structures_dir),
            str(output_file),
            str(tmp_dir),
            # "--alignment-type",
            # "2",
            # "--tmscore-threshold",
            # "0.5",
        ]
        subprocess.run(cmd)

    except:
        shutil.rmtree(tmp_dir)

    return output_file
