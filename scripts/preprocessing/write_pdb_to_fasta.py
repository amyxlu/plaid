import os
from pathlib import Path
from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm

path = Path("/data/bucket/freyn6/data/structures/pdb/")
pdb_files = os.listdir(path)
# len(pdb_files)

parser = PDBParser(QUIET=True)
outfile = "/data/lux70/data/pdb/all_chains.fasta"

minlen = 10

with open(outfile, "w") as f:
    for pdb_file in tqdm(pdb_files):
        pdb_name = pdb_file.rstrip(".pdb")
        filename = path / pdb_file

        try:
            structure = parser.get_structure("protein", filename)
            ppb = PPBuilder()
            for model in structure:
                for chain in model:
                    sequence = ''
                    for pp in ppb.build_peptides(chain):
                        sequence += str(pp.get_sequence())
                        if len(sequence) > minlen:
                            f.write(f">{pdb_name} | Chain {chain.id}\n")
                            f.write(f"{sequence}\n") 

        except Exception as e:
            print(e)
            pass