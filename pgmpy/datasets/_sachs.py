from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class SachsMixed(_BaseDataset):
    name = "sachs_mixed"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url
        + "data/sachs.2005.continuous.discrete.experimental.mixed.maximum.2.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuous(_BaseDataset):
    name = "sachs_continuous"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsDiscrete(_BaseDataset):
    name = "sachs_discrete"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.discrete.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousLogScale(_BaseDataset):
    name = "sachs_continuous_logscale"
    tags = {
        "n_variables": 11,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.logxplus10.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousJitteredLogScale(_BaseDataset):
    name = "sachs_continuous_jittered_logscale"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = (
        base_url + "data/sachs.2005.logxplus10.jittered.eperimental.continuous.txt"
    )
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"


@register_dataset_class
class SachsContinuousJittered(_BaseDataset):
    name = "sachs_continuous_jittered"
    tags = {
        "n_variables": 20,
        "n_samples": 7466,
        "has_ground_truth": True,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/refs/heads/main/real/sachs/"

    data_url = base_url + "data/sachs.2005.with.jittered.experimental.continuous.txt"
    ground_truth_url = base_url + "ground.truth/sachs.2005.ground.truth.graph.txt"
    expert_knowledge_url = base_url + "ground.truth/sachs.2005.knowledge.txt"
