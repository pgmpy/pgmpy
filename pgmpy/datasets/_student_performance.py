from pgmpy.datasets import register_dataset_class
from pgmpy.datasets._base import _BaseDataset


@register_dataset_class
class StudentPerformance(_BaseDataset):
    name = "student_performance"

    tags = {
        "n_variables": 33,
        "n_samples": 395,
        "has_ground_truth": False,
        "has_expert_knowledge": True,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": True,
        "is_ordinal": False,
    }

    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/student-performance/"
    )

    data_url = base_url + "data/student-performance.data.mixed.maximum.3.txt"
    ground_truth_url = None
    expert_knowledge_url = base_url + "ground.truth/student-performance.knowledge.txt"
