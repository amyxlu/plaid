from plaid.evaluation import ConditionalDistributionDistance


# make reference embedding
fid_calc = ConditionalDistributionDistance(
    function_ids=None,
    organism_idx=None,
    max_seq_len=512,
    max_eval_samples=512,
    min_samples=None,
)