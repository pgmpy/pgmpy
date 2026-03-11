from pgmpy.datasets._base import _BaseDataset


class SachsMixed(_BaseDataset):
    _tags = {
        "name": "sachs_mixed",
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url
        + "data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = [
        "cd3_cd28",
        "icam2",
        "aktinhib",
        "g0076",
        "psitect",
        "u0126",
        "ly",
        "pma",
        "b2camp",
    ]
    ordinal_variables = dict()


class SachsContinuous(_BaseDataset):
    _tags = {
        "name": "sachs_continuous",
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()


class SachsDiscrete(_BaseDataset):
    _tags = {
        "name": "sachs_discrete",
        "n_variables": 11,
        "n_samples": 5400,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.discrete.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = [
        "raf",
        "mek",
        "plc",
        "pip2",
        "pip3",
        "erk",
        "akt",
        "pka",
        "pkc",
        "p38",
        "jnk",
    ]
    ordinal_variables = dict()


class SachsContinuousLogScale(_BaseDataset):
    _tags = {
        "name": "sachs_continuous_logscale",
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.logxplus10.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()


class SachsContinuousJitteredLogScale(_BaseDataset):
    _tags = {
        "name": "sachs_continuous_jittered_logscale",
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url + "data/sachs.2005.logxplus10.jittered.eperimental.continuous.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()


class SachsContinuousJittered(_BaseDataset):
    _tags = {
        "name": "sachs_continuous_jittered",
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.with.jittered.experimental.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"

    categorical_variables = []
    ordinal_variables = dict()
