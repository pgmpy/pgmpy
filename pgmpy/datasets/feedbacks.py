from pgmpy.datasets._base import _BaseDataset


_BASE_URL = "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/main/simulated/feedbacks/"

_FEEDBACKS_TAGS = {
    "has_ground_truth": False,
    "has_expert_knowledge": False,
    "has_missing_data": False,
    "has_index_col": False,
    "is_simulated": True,
    "is_interventional": False,
    "is_discrete": False,
    "is_continuous": True,
    "is_mixed": False,
    "is_ordinal": False,
    "n_variables": 5,
    "n_samples": 1000,
}

_categorical_variables = []
_ordinal_variables = dict()


class FeedbacksNetwork1Amp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 1 (amplified variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network1_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network1_amp/sim-01.Network1_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork5Amp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 5 (amplified variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network5_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network5_amp/sim-01.Network5_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork5Cont(_BaseDataset):
    """
    Simulated feedback dataset based on Network 5 (continuous variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network5_cont"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network5_cont/sim-01.Network5_cont.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork5ContP3N7(_BaseDataset):
    """
    Simulated feedback dataset based on Network 5 (continuous variant, p3n7).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network5_cont_p3n7"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network5_cont_p3n7/sim-01.Network5_cont_p3n7.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork5ContP7N3(_BaseDataset):
    """
    Simulated feedback dataset based on Network 5 (continuous variant, p7n3).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network5_cont_p7n3"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network5_cont_p7n3/sim-01.Network5_cont_p7n3.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork6Amp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 6 (amplified variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network6_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network6_amp/sim-01.Network6_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork6Cont(_BaseDataset):
    """
    Simulated feedback dataset based on Network 6 (continuous variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network6_cont"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network6_cont/sim-01.Network6_cont.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork7Amp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 7 (amplified variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network7_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network7_amp/sim-01.Network7_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork7Cont(_BaseDataset):
    """
    Simulated feedback dataset based on Network 7 (continuous variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network7_cont"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network7_cont/sim-01.Network7_cont.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork8AmpAmp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 8 (amp-amp variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network8_amp_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network8_amp_amp/sim-01.Network8_amp_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork8AmpCont(_BaseDataset):
    """
    Simulated feedback dataset based on Network 8 (amp-cont variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network8_amp_cont"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network8_amp_cont/sim-01.Network8_amp_cont.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork8ContAmp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 8 (cont-amp variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network8_cont_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network8_cont_amp/sim-01.Network8_cont_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork9AmpAmp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 9 (amp-amp variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network9_amp_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network9_amp_amp/sim-01.Network9_amp_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork9AmpCont(_BaseDataset):
    """
    Simulated feedback dataset based on Network 9 (amp-cont variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network9_amp_cont"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network9_amp_cont/sim-01.Network9_amp_cont.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables


class FeedbacksNetwork9ContAmp(_BaseDataset):
    """
    Simulated feedback dataset based on Network 9 (cont-amp variant).

    References
    ----------
    .. [1] Sanchez-Romero, R., et al. (2019). Estimating feedforward and feedback
           effective connections from fMRI time series: Assessments of statistical
           methods. Network Neuroscience, 3(2), 274-306.
    .. [2] https://github.com/pgmpy/example-causal-datasets
    """

    _tags = {**_FEEDBACKS_TAGS, "name": "feedbacks_network9_cont_amp"}
    base_url = _BASE_URL
    data_url = _BASE_URL + "data/Network9_cont_amp/sim-01.Network9_cont_amp.continuous.txt"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = _categorical_variables
    ordinal_variables = _ordinal_variables
