from pgmpy.datasets._base import _CausalityChallengeMixin, _BaseDataset

class Marti0(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "marti0",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": False,
        "is_continuous": True,
    }

class Marti1(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "marti1",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": False,
        "is_continuous": True,
    }

class Marti2(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "marti2",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": False,
        "is_continuous": True,
    }
