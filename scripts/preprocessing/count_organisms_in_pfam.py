filepath = "/data/lux70/data/pfam/Pfam-A.fasta"

organism_counts = {}

with open(filepath, "r") as f:
    for i, line in enumerate(f):
        # if i > 1000:
        #     break

        if line[0] == ">":
            organism = line.split("/")[0].split("_")[-1]
            if not organism in organism_counts.keys():
                organism_counts[organism] = 1
            else:
                organism_counts[organism] += 1

import pandas as pd

organism_counts = pd.DataFrame.from_dict(organism_counts, orient="index").reset_index()
organism_counts.columns = ["organism_id", "counts"]
# organism_counts
organism_counts.to_csv("/data/lux70/data/pfam/organism_counts.csv", index=True)
