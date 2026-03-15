# This extension template provides instructions to add new example models to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to the appropriate subdirectory in `pgmpy/example_models`
#    (e.g., `pgmpy/example_models/bnlearn/my_model.py`).
#    If adding a new model source, create a new subdirectory in `pgmpy/example_models` and add an `__init__.py`
#    file to that directory so it becomes a Python package; otherwise models in that source will not be
#    discovered by `load_model()` / `list_models()`.
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


# TODO: Rename the class to match your model name. PascalCase is recommended
# (e.g., MyModel, AsiaNetwork).
# TODO: Inherit from the appropriate Mixin class alongside _BaseExampleModel.
# For e.g., DiscreteMixin, BIFMixin, ContinuousMixin, DAGMixin.
class YourModelName(DiscreteMixin, _BaseExampleModel):
    """[Optional: Short description of the model.].

    [Note: Most existing models only include the References section below; add a description
    here only if it provides useful context for the model.]

    References
    ----------
    .. [1] Author, A., & Author, B. (Year). Title of the paper. Journal Name, Volume(Issue), Pages.
           URL or DOI if available.
    .. [2] Additional reference if needed.

    """

    # TODO: Fill in the metadata tags for your model.
    # All tags are required, except that is_discrete, is_continuous, and is_hybrid should
    # only be provided when is_parameterized=True (omit them entirely otherwise). Set
    # boolean tags to True or False as appropriate.
    _tags = {
        # Unique identifier for the model. Format: "source/model_name" (e.g., "bnlearn/asia", "dagitty/confounding")
        "name": "source/your_model_name",
        # Number of nodes (variables) in the model
        "n_nodes": None,  # TODO: Replace with integer value
        # Number of edges in the model
        "n_edges": None,  # TODO: Replace with integer value
        # Whether the model includes parameters (CPDs) or is just the structure
        "is_parameterized": True,  # TODO: Set to True if model has CPDs, False for structure only
        # TODO: Include the following three tags ONLY if is_parameterized=True; omit them entirely otherwise.
        # Exactly one of is_discrete, is_continuous, is_hybrid must be True.
        "is_discrete": True,  # TODO: Set appropriately
        "is_continuous": False,  # TODO: Set appropriately
        "is_hybrid": False,  # TODO: Set appropriately
    }

    # TODO: Set the path to the model file within the `example_models` repository.
    # For example:
    #   - "discrete/asia.bif.gz" for a gzipped BIF file
    #   - "bnrep/asia.bif" for a plain BIF file
    #   - "continuous/arth150.json" for a JSON file
    #   - "dags/confounding.txt" for a dagitty format file
    data_url = "path/to/your_model_file"

    # TODO: If you need custom loading logic that doesn't fit the standard mixins,
    # you can override the load_model_object method. Otherwise, remove this method.
    @classmethod
    def load_model_object(cls):
        """Load the model from the data file.

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
        return super().load_model_object()
