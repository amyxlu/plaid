from pathlib import Path
from plaid.pipeline import run_analysis, move_designable
from plaid.evaluation import (
    foldseek_easycluster,
    foldseek_easysearch,
    mmseqs_easycluster,
    mmseqs_easysearch,
)

import sys
from plaid.evaluation import RITAPerplexity


sample_dir = Path(sys.argv[1])
# sample_dir = Path("/data/lux70/plaid/baselines/proteingenerator/100_200_300")
rita_perplexity = RITAPerplexity()
df = run_analysis(sample_dir, rita_perplexity=rita_perplexity)

# ===========================
# Foldseek and MMseqs
# ===========================
# this moves everything into a "designable" subdir
move_designable(
    df,
    delete_original=False,
    original_dir_prefix="generated/structures",
    target_dir_prefix="",
)

subdir_name = "designable"

foldseek_easycluster(sample_dir, subdir_name)
foldseek_easysearch(sample_dir, subdir_name)
mmseqs_easycluster(sample_dir, "generated/sequences.fasta")
mmseqs_easysearch(sample_dir, "generated/sequences.fasta")


# root_sample_dir = Path("/data/lux70/plaid/artifacts/samples/by_length/")

# rita_perplexity = RITAPerplexity()

# for length in os.listdir(root_sample_dir):
#     sample_dir = root_sample_dir / str(length)
#     print("=====================================")
#     print(f"Running analysis on {sample_dir}")
#     print("=====================================")
#     run_analysis(sample_dir, rita_perplexity=rita_perplexity)
