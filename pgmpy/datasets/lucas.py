# Currantly a WIP due to issue with wierd hosting of dataset
from pgmpy.datasets._base import _BaseDataset


class _LucasBase(_BaseDataset):
    categorical_variables = [
        "Smoking",
        "Yellow_Fingers",
        "Anxiety",
        "Peer_Pressure",
        "Genetics",
        "Attention_Disorder",
        "Born_an_Even_Day",
        "Car_Accident",
        "Fatigue",
        "Allergy",
        "Coughing",
        "Target",
    ]

    ordinal_variales = dict()

    _common_tags = {
        "n_variables": 12,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_discrete": True,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }


class Lucas0Train(_LucasBase):
    """
    References
    ----------
    .. [1] https://www.causality.inf.ethz.ch/data/LUCAS.html
    """

    _tags = {
        **_LucasBase._common_tags,
        "name": "lucas0_train",
        "n_samples": 2000,
        "has_ground_truth": True,
        "is_interventional": False,
    }

    # TODO: Add data_url and ground_truth when data hosted
    data_url = None
    ground_truth_url = None


class Lucas0Test(_LucasBase):
    """
    References
    ----------
    .. [1] https://www.causality.inf.ethz.ch/data/LUCAS.html
    """

    _tags = {
        **_LucasBase._common_tags,
        "name": "lucas0_test",
        "n_samples": 10000,
        "has_ground_truth": True,
        "is_interventional": False,
    }

    # TODO: Add data_url and ground_truth when data hosted
    data_url = None
    ground_truth_url = None


class Lucas1Test(_LucasBase):
    """
    References
    ----------
    .. [1] https://www.causality.inf.ethz.ch/data/LUCAS.html
    """

    _tags = {
        **_LucasBase._common_tags,
        "name": "lucas1_test",
        "n_samples": 10000,
        "has_ground_truth": True,
        "is_interventional": True,
    }

    # TODO: Add data_url and ground_truth when data hosted
    data_url = None
    ground_truth_url = None


class Lucas2Test(_LucasBase):
    """
    References
    ----------
    .. [1] https://www.causality.inf.ethz.ch/data/LUCAS.html
    """

    _tags = {
        **_LucasBase._common_tags,
        "name": "lucas2_test",
        "n_samples": 10000,
        "has_ground_truth": True,
        "is_interventional": True,
    }

    # TODO: Add data_url and ground_truth when data hosted
    data_url = None
    ground_truth_url = None
