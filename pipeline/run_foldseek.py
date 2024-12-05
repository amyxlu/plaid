import argparse
from pathlib import Path

from plaid.evaluation import foldseek_easysearch, foldseek_easycluster

parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples_dir", default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist/"
)
parser.add_argument("--structure_subdir_name", default="None")
parser.add_argument("--use_designability_filter", action="store_true")
args = parser.parse_args()

print(args)

# Unless a specific subdir is specified, use the default subdir name based on designability filter
if args.structure_subdir_name == "None":
    if args.use_designability_filter:
        subdir_name = "designable"
    else:
        subdir_name = "generated/structures"
else:
    subdir_name = args.structure_subdir_name

# use the default output file name based on designability filter
if args.use_designability_filter:
    easysearch_output_file_name = "foldseek_easysearch.m8"
    easycluster_output_file_name = "foldseek_easycluster.m8"
else:
    easysearch_output_file_name = "no_filter_foldseek_easysearch.m8"
    easycluster_output_file_name = "no_filter_foldseek_easycluster.m8"

subdir_name = Path(subdir_name)
samples_dir = Path(args.samples_dir)


print("==========================================")
print("Running foldseek_easysearch and foldseek_easycluster")
print("Structures:", samples_dir / subdir_name)
print("Use designability filter", args.use_designability_filter)
print("==========================================")

foldseek_easysearch(samples_dir, subdir_name, easysearch_output_file_name)
foldseek_easycluster(samples_dir, subdir_name, easycluster_output_file_name)
