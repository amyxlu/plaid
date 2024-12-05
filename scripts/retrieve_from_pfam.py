# import webdataset as wds
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Set, Optional
# import json

# from plaid.datasets import FunctionOrganismDataModule

# class WebDatasetFilter:
#     def __init__(self, dm: FunctionOrganismDataModule):
#         """
#         Initialize filter for WebDataset format data

#         Parameters:
#         -----------
#         shards_pattern: str
#             Pattern for WebDataset shards (e.g., "path/to/shard_{0000..4423}.tar")
#         go_metadata_fpath: str
#             Path to GO term metadata CSV
#         organism_metadata_fpath: str
#             Path to organism metadata CSV
#         config_file: str
#             Path to dataset config JSON
#         cache_dir: Optional[str]
#             Directory for caching WebDataset shards
#         """
#         self.dm = dm
#         self.dataloader = dm.train_dataloader()
#         self.go_metadata = dm.metadata_parser.go_metadata
#         self.organism_metadata = dm.metadata_parser.organism_metadata

#     def filter_dataset(
#             self,
#             go_indices: List[int],
#             organism_indices: List[int],
#             output_file: Optional[str] = None
#         ) -> pd.DataFrame:
#         """
#         Filter WebDataset based on GO terms and organisms
#         """
#         # Convert terms and organisms to indices
#         import pdb;pdb.set_trace()

#         results = []

#         import IPython;IPython.embed()
#         # Process batches
#         import tqdm
#         for batch in tqdm.tqdm(self.dataloader):
#             embedding, mask, go_idx, organism_idx, pfam_id, sample_id, local_path = batch

#             # Convert to numpy for easier filtering
#             go_idx = go_idx.numpy()
#             organism_idx = organism_idx.numpy()

#             # Find matches
#             go_matches = np.isin(go_idx, list(go_indices))
#             org_matches = np.isin(organism_idx, list(organism_indices))
#             matches = go_matches & org_matches

#             # Collect matching samples
#             for i in np.where(matches)[0]:
#                 results.append({
#                     'embedding': embedding[i],
#                     'mask': mask[i],
#                     'go_idx': int(go_idx[i]),
#                     'organism_idx': int(organism_idx[i]),
#                     'pfam_id': pfam_id[i],
#                     'sample_id': sample_id[i],
#                     'local_path': local_path[i],
#                     'go_term': self.go_metadata.iloc[go_idx[i]]['go_id'],
#                     'organism': self.organism_metadata.iloc[organism_idx[i]]['organism']
#                 })

#         import IPython;IPython.embed()

# # Example usage
# if __name__ == "__main__":
#     from plaid.datasets import FunctionOrganismDataModule

#     dm = FunctionOrganismDataModule(
#         train_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/train/shard{0000..4423}.tar",
#         val_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/val/shard{0000..0863}.tar",
#         config_file="/data/lux70/data/pfam/compressed/j1v1wv6w/config.json",
#         go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
#         organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
#         cache_dir="/homefs/home/lux70/cache/plaid_data_cache/j1v1wv6w",
#         train_epoch_num_batches=100_000,
#         val_epoch_num_batches=1_000,
#         shuffle_buffer=10_000,
#         shuffle_initial=10_000,
#         num_workers=1,
#         batch_size=16,
#     )
#     dm.setup()

#     filter = WebDatasetFilter(dm)

#     # Filter dataset
#     filtered_data = filter.filter_dataset(
#         go_indices=[166,28],  # example: DNA-binding transcription factor activity
#         organism_indices=[1326,818],
#     )

#     print(f"Found {len(filtered_data)} matching sequences")
#     print("\nOrganism distribution:")
#     print(filtered_data['organism'].value_counts())
#     print("\nGO term distribution:")
#     print(filtered_data['go_term'].value_counts())
