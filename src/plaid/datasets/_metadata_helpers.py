"""
Helpers during training and evaluation for loading GO term and organism metadata.
"""

import pandas as pd
import os


NUM_FUNCTION_CLASSES = 2219
NUM_ORGANISM_CLASSES = 3617
HUMAN_IDX = 1326


GO_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../../assets", "pfam2go.csv"
)

ORGANISM_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../../assets", "organism_counts.csv"
)


class MetadataParser:
    def __init__(
        self,
        go_metadata_fpath=GO_METADATA_PATH,
        organism_metadata_fpath=ORGANISM_METADATA_PATH,
    ):
        """
        Class to extract relevant information from Pfam headers using specific data structures.

        go_metadata is created with notebooks/DATA_pfam_to_GO.ipynb
        organism indexing is created with notebooks/DATA_organism_counts.ipynb
        """
        self.go_metadata = pd.read_csv(go_metadata_fpath)
        self.organism_metadata = pd.read_csv(organism_metadata_fpath)

        # the last index is reserved for dummy values, and number of class is 1-indexed, hence the dummy value should equal the number of classes
        assert (
            self.go_metadata["GO_idx"].max() + 1 == NUM_FUNCTION_CLASSES
        ), f"GO_idx max is {self.go_metadata['GO_idx'].max()}, expected {NUM_FUNCTION_CLASSES}"
        assert (
            self.organism_metadata["organism_index"].max() + 1 == NUM_ORGANISM_CLASSES
        ), f"organism_index max is {self.organism_metadata['organism_index'].max()}, expected {NUM_ORGANISM_CLASSES}"

        self.dummy_go_idx = self.go_metadata["GO_idx"].max() + 1
        self.dummy_organism_idx = self.organism_metadata["organism_index"].max() + 1

    def header_to_pfam_id(self, header: str):
        return header.split(" ")[-1].split(".")[0]

    def header_to_organism(self, header: str):
        return header.split("/")[0].split("_")[-1]

    def header_to_go_idx(self, header: str):
        pfam_id = self.header_to_pfam_id(header)
        try:
            return self.go_metadata[self.go_metadata["pfam_id"] == pfam_id][
                "GO_idx"
            ].item()
        except ValueError:
            return self.dummy_go_idx

    def header_to_organism_idx(self, header: str):
        organism = self.header_to_organism(header)
        try:
            return self.organism_metadata[
                self.organism_metadata.organism_id == organism
            ]["organism_index"].item()
        except ValueError:
            return self.dummy_organism_idx


if __name__ == "__main__":
    metadata_parser = MetadataParser(
        go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
    )

    sample_headers = [
        "A0A418XQZ6_9BURK/10-72 A0A418XQZ6.1 PF00392.25;GntR;",
        "A0A1E7JVW7_9ACTN/15-77 A0A1E7JVW7.1 PF00392.25;GntR;",
        "A0A4Y9SLN2_9BURK/19-82 A0A4Y9SLN2.1 PF00392.25;GntR;",
    ]
    import IPython

    IPython.embed()
