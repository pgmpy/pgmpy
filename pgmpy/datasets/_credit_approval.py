from pgmpy.datasets._base import _BaseDataset, register_dataset_class


@register_dataset_class
class CreditApproval(_BaseDataset):
    """
    Credit Approval dataset.

    This dataset contains credit approval decisions with various anonymized attributes.
    The original variable names and meanings have been deliberately obscured.

    Variables: 16 attributes (15 features + 1 target)
    Samples: 690
    """

    name = "credit_approval"
    base_url = (
        "https://raw.githubusercontent.com/pgmpy/example-causal-datasets/"
        "refs/heads/main/real/credit-approval/"
    )
    data_url = base_url + "data/crx.data.mixed.maximum.14.txt"

    tags = {
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "is_continuous": False,
        "is_discrete": False,
        "is_mixed": True,
        "is_interventional": False,
        "n_samples": 690,
        "is_ordinal": False,
        "is_simulated": False,
        "n_variables": 16,
    }