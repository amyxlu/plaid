import argparse
from pathlib import Path

from plaid.evaluation import mmseqs_easysearch, mmseqs_easycluster

parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples_dir",
    default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist/f989_o1326_l144_s3",
)
parser.add_argument("--fasta_file_name", default="generated/sequences.fasta")
args = parser.parse_args()

samples_dir = Path(args.samples_dir)

print("==========================================")
print("Running mmseqs_easysearch and mmseqs_easycluster")
print("Sequences:", samples_dir / args.fasta_file_name)
print("==========================================")

mmseqs_easysearch(samples_dir, args.fasta_file_name)
mmseqs_easycluster(samples_dir, args.fasta_file_name)
