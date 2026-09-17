from skpro.distributions.base import BaseDistribution

from pgmpy.parameter._base import BaseParameter


class DistributionAdapter(BaseParameter):
    """
    Parameter adapter class for skpro distribution class.

    Parameters
    ----------
    distribution : object

    Examples
    --------
    >>> from skpro.distributions.normal import Normal
    >>> from pgmpy.parameter.adapter.DistributionAdapter import DistributionAdapter
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(seed=42)
    >>> X = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=10),
    ...     }
    ... )
    >>> normal = DistributionAdapter(Normal(0, 1))
    >>> dist = normal.predict_proba(X)
    >>> dist # doctest: +SKIP
    Normal()

    """

    _tags = {
        "parameter_type": "distribution",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_fit_joint": False,
        "missing": False,
        "can_be_root": True,
    }

    def __init__(self, distribution: BaseDistribution):
        if not isinstance(distribution, BaseDistribution):
            raise TypeError(
                f"distribution must be an instance of a skpro distribution. Received: {type(distribution).__name__}"
            )

        self.distribution = distribution
        super().__init__()
        self._is_fitted = True

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def _predict_proba(self, X):
        """Broadcast the fixed distribution to the rows of ``X``."""
        distribution_params = self.distribution.get_params(deep=False)
        if self.distribution.get_tag("broadcast_init") == "on":
            distribution_params.update(index=X.index)

            return type(self.distribution)(**distribution_params)
        else:
            # NOTE: Only `KernelMixture` and `NominalDistribution` have `"broadcast_init" == "off"`.
            # This logic handles those classes.

            if self.distribution.__class__.__name__ == "NominalDistribution":
                # NOTE: Logic for `NominalDistribution`
                distribution_params.update(
                    index=X.index,
                    probs=[self.distribution.probs[0] for i in range(len(X))],
                    categories=self.distribution.categories,
                )
            elif self.distribution.__class__.__name__ == "KernelMixture":
                # TODO: Implement logic for `skpro.distributions.KernelMixture`
                ...
            return type(self.distribution)(**distribution_params)
