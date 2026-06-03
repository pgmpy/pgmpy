from pgmpy.datasets._base import _CausalityChallengeMixin, _BaseDataset

class Lucap0(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucap0",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }

class Lucap1(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucap1",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }

class Lucap2(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucap2",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }
