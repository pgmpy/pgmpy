from pgmpy.datasets._base import _CausalityChallengeMixin, _BaseDataset

class Cina0(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "cina0",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": True,
        "is_mixed": True,
    }

class Cina1(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "cina1",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": True,
        "is_mixed": True,
    }

class Cina2(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "cina2",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": True,
        "is_mixed": True,
    }
