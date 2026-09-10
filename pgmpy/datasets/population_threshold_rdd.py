from pgmpy.datasets._base import BaseDataset


class PopulationThresholdRDDItaly(BaseDataset):
    """Italian municipality sample from the Eggers et al. population-threshold RDD data.

    Each row is a municipality-year compared to a nearby population cutoff.
    ``pop_dev`` is the running variable (census population minus ``cut``);
    ``pop_dev >= 0`` means the municipality is above the threshold.
    ``placebo`` marks placebo cutoffs; ``salary`` and ``council`` mark whether
    mayor pay or council size change at that cutoff.

    Source: Harvard Dataverse, doi:10.7910/DVN/PGXO5O
    Original archive: ``replication_materials_submitted_20170403.zip``
    (MD5 ``df3c2bd36a9f5fcd9d1e86d5b8e31f66``).
    Packaged file SHA256
    ``608e33e1907f573e991d206ea72827b7b65c674f514d3a5f37ae754ba1430846``.
    License: CC0 1.0.

    References
    ----------
    - :footcite:t:`eggers_2018_rdd`
    - :footcite:t:`eggers_2018_rdd_dataset`
    """

    _tags = {
        "name": "population_threshold_rdd_italy",
        "n_variables": 19,
        "n_samples": 10409,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": True,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "population-threshold-rdd"

    data_url = "data/population-threshold-rdd.italy.mixed.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    missing_values_marker = "NA"

    categorical_variables = ["macro_area"]
    ordinal_variables = dict()
