from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset


class _CausalChamberBase(_BaseDataset):
    """
    Base class for CausalChamber datasets.

    CausalChamber is a collection of real physical systems designed as a testbed
    for causal AI methodology. Each dataset combines interventional data with known
    ground-truth causal structures.

    Subclasses should define:
    - chamber: str - Chamber identifier (e.g., 'lt' for light tunnel, 'wt' for wind tunnel)
    - configuration: str - Configuration name (e.g., 'standard')
    - dataset_name: str - Dataset identifier (e.g., 'lt_camera_test_v1')
    - experiment_name: str - Experiment name (e.g., 'palette', 'random')

    References
    ----------
    .. [1] Gamella, J. L., et al. (2024). The Causal chambers: Real physical systems as
           a testbed for AI methodology. Nature Machine Intelligence, 6(1), 86-105.
           https://doi.org/10.1038/s42256-024-00964-x
    """

    # Subclasses must define these
    chamber = None
    configuration = None
    dataset_name = None
    experiment_name = None

    base_url = "https://github.com/juangamella/causal-chamber"
    ground_truth_url = None
    expert_knowledge_url = None
    categorical_variables = []
    ordinal_variables = dict()

    @classmethod
    def load_dataframe(cls):
        """
        Load data from the causalchamber package.

        Returns
        -------
        pd.DataFrame
            The experimental data as a pandas DataFrame.

        Raises
        ------
        ImportError
            If the causalchamber package is not installed.
        ValueError
            If required class attributes are not defined.
        """
        try:
            import causalchamber.datasets as datasets
        except ImportError:
            raise ImportError(
                "The 'causalchamber' package is required to load this dataset. "
                "Install it using: pip install causalchamber"
            )

        if cls.dataset_name is None or cls.experiment_name is None:
            raise ValueError(
                f"{cls.__name__} must define 'dataset_name' and 'experiment_name'"
            )

        dataset = datasets.Dataset(name=cls.dataset_name, root="./", download=True)
        df = dataset.get_experiment(name=cls.experiment_name).as_pandas_dataframe()

        return df

    @classmethod
    def load_ground_truth(cls) -> DAG:
        """
        Load ground truth DAG from the causalchamber package.

        Returns
        -------
        DAG
            The ground truth causal graph as a pgmpy DAG.

        Raises
        ------
        ImportError
            If the causalchamber package is not installed.
        ValueError
            If required class attributes are not defined.
        """
        if not cls.get_class_tag("has_ground_truth"):
            return None

        try:
            from causalchamber.ground_truth import graph
        except ImportError:
            raise ImportError(
                "The 'causalchamber' package is required. "
                "Install it using: pip install causalchamber"
            )

        if cls.chamber is None or cls.configuration is None:
            raise ValueError(
                f"{cls.__name__} must define 'chamber' and 'configuration'"
            )

        # Get ground truth as adjacency matrix (DataFrame)
        adj_matrix = graph(chamber=cls.chamber, configuration=cls.configuration)

        # Convert adjacency matrix to edge list
        # adj_matrix[i, j] = 1 means there's an edge from column j to row i
        edges = []
        for parent in adj_matrix.columns:
            for child in adj_matrix.index:
                if adj_matrix.loc[child, parent] == 1:
                    edges.append((parent, child))

        dag = DAG()
        dag.add_edges_from(edges)

        return dag


class LightTunnelPalette(_CausalChamberBase):
    """
    CausalChamber Light Tunnel dataset with palette experiment.

    The light tunnel is a controlled environment where colored LEDs illuminate
    a chamber, and cameras capture the resulting colors. The palette experiment
    includes interventions on different LED intensities.

    Variables include RGB values from multiple camera angles, LED control signals,
    polarizer angles, and various sensor readings.

    References
    ----------
    .. [1] Gamella, J. L., et al. (2024). The Causal chambers: Real physical systems as
           a testbed for AI methodology. Nature Machine Intelligence, 6(1), 86-105.
           https://doi.org/10.1038/s42256-024-00964-x
    """

    _tags = {
        "name": "light_tunnel_palette",
        "n_variables": 52,  # Total columns including metadata
        "n_samples": 224,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    chamber = "lt"
    configuration = "standard"
    dataset_name = "lt_camera_test_v1"
    experiment_name = "palette"


class LightTunnelPureColorsBright(_CausalChamberBase):
    """
    CausalChamber Light Tunnel dataset with pure colors bright experiment.

    The light tunnel is a controlled environment where colored LEDs illuminate
    a chamber, and cameras capture the resulting colors. The pure colors bright
    experiment tests individual LED colors at bright intensity levels.

    Variables include RGB values from multiple camera angles, LED control signals,
    polarizer angles, and various sensor readings.

    References
    ----------
    .. [1] Gamella, J. L., et al. (2024). The Causal chambers: Real physical systems as
           a testbed for AI methodology. Nature Machine Intelligence, 6(1), 86-105.
           https://doi.org/10.1038/s42256-024-00964-x
    """

    _tags = {
        "name": "light_tunnel_pure_colors_bright",
        "n_variables": 52,
        "n_samples": 3,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": True,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    chamber = "lt"
    configuration = "standard"
    dataset_name = "lt_camera_test_v1"
    experiment_name = "pure_colors_bright"
