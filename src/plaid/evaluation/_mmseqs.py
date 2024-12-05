import os
from pathlib import Path
import shutil
import subprocess


EASY_SEARCH_OUTPUT_COLS = [
    "query",
    "target",
    "evalue",
    "alnlen",
    "qseq",
    "tseq",
    "qheader",
    "theader",
    "taxid",
    "taxname",
    "taxlineage",
]

# query,target,evalue,gapopen,pident,fident,nident,qstart,qend,qlen
# tstart,tend,tlen,alnlen,raw,bits,cigar,qseq,tseq,qheader,theader,qaln,taln,qframe,tframe,mismatch,qcov,tcov
# qset,qsetid,tset,tsetid,taxid,taxname,taxlineage,qorfstart,qorfend,torfstart,torfend


def mmseqs_easysearch(
    sample_dir,
    fasta_file_name="generated/sequences.fasta",
    output_file_name="mmseqs_easysearch.m8",
):
    sample_dir = Path(sample_dir)
    fasta_file = sample_dir / fasta_file_name
    output_file = sample_dir / output_file_name
    tmp_dir = sample_dir / "tmp"

    # UNIREF_DATABASE_PATH = "/data/lux70/data/uniref50db/uniref50db"
    UNIREF_DATABASE_PATH = os.environ.get("UNIREF_DATABASE_PATH")
    if UNIREF_DATABASE_PATH is None:
        raise ValueError(
            "UNIREF_DATABASE_PATH environment variable not set. Please use"
            "`mmseqs databases UniRef50 <o:db> <tmpDir>` to download the UniRef50 database"
            "and set the UNIREF_DATABASE_PATH environment variable to the output path <o:db>."
        )

    try:
        cmd = [
            "mmseqs",
            "easy-search",
            str(fasta_file),
            UNIREF_DATABASE_PATH,
            str(output_file),
            str(tmp_dir),
            "--format-output",
            ",".join(EASY_SEARCH_OUTPUT_COLS),
            "--split-memory-limit",
            "10G",
        ]
        subprocess.run(cmd)

    except:
        shutil.rmtree(tmp_dir)

    return output_file


def mmseqs_easycluster(
    sample_dir,
    fasta_file_name="generated/sequences.fasta",
    output_file_name="mmseqs_easycluster.m8",
):
    sample_dir = Path(sample_dir)
    fasta_file = sample_dir / fasta_file_name
    output_file = sample_dir / output_file_name
    tmp_dir = sample_dir / "tmp"

    try:
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(fasta_file),
            str(output_file),
            str(tmp_dir),
        ]
        subprocess.run(cmd)

    except:
        shutil.rmtree(tmp_dir)

    return output_file
