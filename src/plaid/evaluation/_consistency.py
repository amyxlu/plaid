# from pathlib import Path
# import glob
# import re

# from plaid.evaluation import batch_rmsd_calculation, batch_rmspd_from_pdb_paths, run_tmalign
# from plaid.utils import read_sequences_from_fasta, calc_sequence_recovery

# import re

# def filename_to_suffix_number(path):
#     x = str(path).split("/")[-1]
#     return int(re.findall(r"\d+", x)[-1])

# def sort_by_suffix(lst):
#     import re
#     sorted_list = sorted(lst, key=lambda path: filename_to_suffix_number(path))
#     return sorted_list


# class CrossConsistencyEvaluation:
#     def __init__(self, experiment_dir: Path):
#         self.experiment_dir = Path(experiment_dir)

#         generated_pdb_dir = self.experiment_dir / "generated" / "structures"
#         self.generated_pdb_paths = glob.glob(str(generated_pdb_dir / "*.pdb"))
#         self.generated_pdb_paths = sort_by_suffix(self.generated_pdb_paths)

#         self.generated_fasta_path = self.experiment_dir / "generated" / "sequences.fasta"

#         inverse_generated_pdb_dir = self.experiment_dir / "inverse_generated" / "structures"
#         self.inverse_generated_pdb_paths = glob.glob(str(inverse_generated_pdb_dir / "*.pdb"))
#         self.inverse_generated_pdb_paths = sort_by_suffix(self.inverse_generated_pdb_paths)

#         self.inverse_generated_fasta_path = self.experiment_dir / "inverse_generated" / "sequences.fasta"

#     def cross_consistency_rmsd(self):
#         return batch_rmsd_calculation(self.generated_pdb_paths, self.inverse_generated_pdb_paths)

#     def cross_consistency_rmspd(self):
#         return batch_rmspd_from_pdb_paths(self.generated_pdb_paths, self.inverse_generated_pdb_paths)

#     def cross_consistency_tm(self):
#         return [run_tmalign(p1, p2) for (p1, p2) in zip(self.generated_pdb_paths, self.inverse_generated_pdb_paths)]

#     def cross_consistency_sr(self):
#         gen_seqs_dict = read_sequences_from_fasta(self.generated_fasta_path)
#         inv_gen_seqs_dict = read_sequences_from_fasta(self.inverse_generated_fasta_path)
#         gen_seqs = [gen_seqs_dict[k] for k in sorted(gen_seqs_dict)]
#         inv_gen_seqs = [inv_gen_seqs_dict[k] for k in sorted(inv_gen_seqs_dict)]
#         return [calc_sequence_recovery(s1, s2) for (s1, s2) in zip(gen_seqs, inv_gen_seqs)]


# class SelfConsistencyEvaluation:
#     def __init__(self, experiment_dir: Path):
#         self.experiment_dir = Path(experiment_dir)

#         inverse_generated_pdb_dir = self.experiment_dir / "inverse_generated" / "structures"
#         self.inverse_generated_pdb_paths = glob.glob(str(inverse_generated_pdb_dir / "*.pdb"))
#         self.inverse_generated_pdb_paths = sort_by_suffix(self.inverse_generated_pdb_paths)

#         self.inverse_generated_fasta_path = self.experiment_dir / "inverse_generated" / "sequences.fasta"

#         phantom_generated_pdb_dir = self.experiment_dir / "phantom_generated" / "structures"
#         self.phantom_generated_pdb_paths = glob.glob(str(phantom_generated_pdb_dir / "*.pdb"))
#         self.phantom_generated_pdb_paths = sort_by_suffix(self.phantom_generated_pdb_paths)

#         self.phantom_generated_fasta_path = self.experiment_dir / "phantom_generated" / "sequences.fasta"

#     def self_consistency_rmsd(self):
#         return batch_rmsd_calculation(self.generated_pdb_paths, self.phantom_generated_pdb_paths)

#     def self_consistency_rmspd(self):
#         return batch_rmspd_from_pdb_paths(self.generated_pdb_paths, self.phantom_generated_pdb_paths)

#     def self_consistency_tm(self):
#         return [run_tmalign(p1, p2) for (p1, p2) in zip(self.generated_pdb_paths, self.phantom_generated_pdb_paths)]

#     def self_consistency_sr(self):
#         gen_seqs_dict = read_sequences_from_fasta(self.generated_fasta_path)
#         inv_gen_seqs_dict = read_sequences_from_fasta(self.phantom_generated_fasta_path)
#         gen_seqs = [gen_seqs_dict[k] for k in sorted(gen_seqs_dict)]
#         inv_gen_seqs = [inv_gen_seqs_dict[k] for k in sorted(inv_gen_seqs_dict)]
#         return [calc_sequence_recovery(s1, s2) for (s1, s2) in zip(gen_seqs, inv_gen_seqs)]
