from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class AppleWatchFitbit(_BaseDataset):
    name = "apple_watch_fitbit"
    tags = {
        "n_variables": 17,
        "n_samples": 6264,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "has_missing_data": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/apple-watch-fitbit/"
    )

    data_url = base_url + "data/aw-fb-pruned18.data.mixed.maximum.6.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/aw-fb-pruned18.knowledge.txt"
