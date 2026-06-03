from pgmpy.datasets._base import _BaseDataset, _CausalityChallengeMixin


class Lucas0(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucas0",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }


class Lucas1(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucas1",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }


class Lucas2(_CausalityChallengeMixin, _BaseDataset):
    _tags = {
        "name": "lucas2",
        "has_ground_truth": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
    }
