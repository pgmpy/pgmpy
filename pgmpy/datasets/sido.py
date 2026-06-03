from pgmpy.datasets._base import _BaseDataset, _CausalityChallengeMixin


class Sido0(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "sido0",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": False,
    }


class Sido1(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "sido1",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": False,
    }


class Sido2(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "sido2",
        "has_ground_truth": False,
        "is_simulated": False,
        "is_discrete": True,
        "is_continuous": False,
    }
