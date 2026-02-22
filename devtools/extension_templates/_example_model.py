# This extension template provides instructions to add new example models to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to the appropriate subdirectory in `pgmpy/example_models` (e.g., `pgmpy/example_models/bnlearn/my_model.py`).
#    If adding a new model source, create a new subdirectory in `pgmpy/example_models`.
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. If you would like to contribute the model to pgmpy, please add the model name to the appropriate test list in
#   `pgmpy/tests/test_example_models/test_example_models.py` file.

# TODO: Import the appropriate mixin class for your model type:
#   - DiscreteMixin: For discrete Bayesian networks stored as gzipped BIF files
#   - BIFMixin: For discrete Bayesian networks stored as plain BIF files
#   - ContinuousMixin: For continuous Bayesian networks stored as JSON files
#   - DAGMixin: For DAGs without parameters, stored in dagitty string format
from .._base import DiscreteMixin, _BaseExampleModel

# TODO: If using a new mixin type, import it here:
# from .._base import BIFMixin, ContinuousMixin, DAGMixin


# TODO: Rename the class to match your model name (use PascalCase, e.g., MyModel, AsiaNetwork).
class YourModelName(DiscreteMixin, _BaseExampleModel):
    """
    [One line description of the model.]

    [Optional: Add a longer description if needed, explaining what the model represents,
    its purpose, or any interesting characteristics.]

    This model contains [X] nodes and [Y] edges, representing [domain description].

    References
    ----------
    ..[1] Author, A., & Author, B. (Year). Title of the paper. Journal Name, Volume(Issue), Pages.
           URL or DOI if available.
    ..[2] Additional reference if needed.
    """

    # TODO: Fill in the metadata tags for your model.
    # All tags are required. Set boolean tags to True or False as appropriate.
    _tags = {
        # Unique identifier for the model. Format: "source/model_name" (e.g., "bnlearn/asia", "dagitty/confounding")
        "name": "source/your_model_name",
        # Number of nodes (variables) in the model
        "n_nodes": None,  # TODO: Replace with integer value
        # Number of edges in the model
        "n_edges": None,  # TODO: Replace with integer value
        # Whether the model includes parameters (CPDs) or is just the structure
        "is_parameterized": True,  # TODO: Set to True if model has CPDs, False for structure only
        # Whether all variables are discrete (only applicable if is_parameterized=True)
        "is_discrete": True,  # TODO: Set appropriately
        # Whether all variables are continuous (only applicable if is_parameterized=True)
        "is_continuous": False,  # TODO: Set appropriately
        # Whether the model has both discrete and continuous variables (only applicable if is_parameterized=True)
        "is_hybrid": False,  # TODO: Set appropriately
    }

    # TODO: Set the URL or path to the model file, relative to the base URL.
    # For example:
    #   - "discrete/asia.bif.gz" for a gzipped BIF file
    #   - "bnrep/asia.bif" for a plain BIF file
    #   - "continuous/arth150.json" for a JSON file
    #   - "dagitty/confounding.txt" for a dagitty format file
    data_url = "path/to/your_model_file"

    # TODO: If you need custom loading logic that doesn't fit the standard mixins,
    # you can override the load_model_object method. Otherwise, remove this method.
    @classmethod
    def load_model_object(cls):
        """
        Custom method to load the model from the data file.

        Returns
        -------
        model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork or
               pgmpy.models.LinearGaussianBayesianNetwork or pgmpy.models.FunctionalBayesianNetwork
            The loaded model object.
        """
        # By default, the mixin classes handle loading. Only override if you need custom logic.
        # Example custom loading:
        # raw_data = cls._get_raw_data()
        # # Your custom parsing logic here
        # return model
        pass


# GUIDELINES FOR MODEL METADATA
# ==============================
#
# name:
#   - Format: "source/model_name" (lowercase with underscores)
#   - Examples: "bnlearn/asia", "bnrep/alarm", "dagitty/m_bias"
#   - Must be unique across all models
#
# n_nodes and n_edges:
#   - Count the exact number of nodes and edges in your model
#   - These will be validated against the loaded model in tests
#
# is_parameterized:
#   - True: Model includes CPDs (conditional probability distributions)
#   - False: Model is structure-only (just nodes and edges)
#
# is_discrete, is_continuous, is_hybrid:
#   - Only relevant if is_parameterized=True
#   - Exactly one must be True for parameterized models
#   - is_discrete: All variables are discrete (e.g., binary, categorical)
#   - is_continuous: All variables are continuous (e.g., Gaussian)
#   - is_hybrid: Mix of discrete and continuous variables
#
# References:
#   - Always include at least one reference in the docstring
#   - Use proper citation format (author, year, title, journal/conference, etc.)
#   - Include DOI or URL when available
#   - Multiple references can be added as [1], [2], etc.
#
# EXAMPLE IMPLEMENTATIONS:
# ========================
#
# Example 1: Discrete Bayesian Network (gzipped BIF)
# ---------------------------------------------------
# from .._base import DiscreteMixin, _BaseExampleModel
#
# class Asia(DiscreteMixin, _BaseExampleModel):
#     """
#     Asia network - A small Bayesian network for diagnosing lung diseases.
#
#     References
#     ----------
#     .. [1] Lauritzen, S., & Spiegelhalter, D. (1988). Local Computation with
#            Probabilities on Graphical Structures and their Application to Expert
#            Systems. Journal of the Royal Statistical Society: Series B, 50(2):157-224.
#     """
#     _tags = {
#         "name": "bnlearn/asia",
#         "n_nodes": 8,
#         "n_edges": 8,
#         "is_parameterized": True,
#         "is_discrete": True,
#         "is_continuous": False,
#         "is_hybrid": False,
#     }
#     data_url = "discrete/asia.bif.gz"
#
#
# Example 2: DAG without parameters
# ----------------------------------
# from .._base import DAGMixin, _BaseExampleModel
#
# class MBias(DAGMixin, _BaseExampleModel):
#     """
#     M-bias structure - A classic example of collider bias.
#
#     References
#     ----------
#     .. [1] Pearl, J. (2009). Causality: Models, Reasoning and Inference.
#            Cambridge University Press.
#     """
#     _tags = {
#         "name": "dagitty/m_bias",
#         "n_nodes": 4,
#         "n_edges": 4,
#         "is_parameterized": False,
#         "is_discrete": False,
#         "is_continuous": False,
#         "is_hybrid": False,
#     }
#     data_url = "dags/M-bias.txt"
#
#
# Example 3: Continuous Bayesian Network
# ---------------------------------------
# from .._base import ContinuousMixin, _BaseExampleModel
#
# class arth150(ContinuousMixin, _BaseExampleModel):
#     """
#     Arthritis data with 150 edges - A large continuous Bayesian network.
#
#     References
#     ----------
#     .. [1] Scutari, M., et al. (2010). Learning Bayesian Networks with the
#            bnlearn R Package. Journal of Statistical Software, 35(3), 1-22.
#     """
#     _tags = {
#         "name": "bnlearn/arth150",
#         "n_nodes": 107,
#         "n_edges": 150,
#         "is_parameterized": True,
#         "is_discrete": False,
#         "is_continuous": True,
#         "is_hybrid": False,
#     }
#     data_url = "continuous/arth150.json"
