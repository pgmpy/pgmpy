from collections.abc import Hashable
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _safe_import

from pgmpy.parameterization._base import BaseParameter, _checked_evidence, _equal, _parent_order
from pgmpy.parameterization.distributions import NominalDistribution

BaseDistribution = _safe_import("skpro.distributions.base.BaseDistribution")
IID = _safe_import("skpro.distributions.IID")
Normal = _safe_import("skpro.distributions.Normal")

if TYPE_CHECKING:
    from skpro.distributions.base import BaseDistribution  # noqa: F811


class DistributionAdapter(BaseParameter):
    """Parameterization with a fixed distribution, such as that of a root or of a variable set by an intervention.

    The distribution is given, not learned: ``from_values`` creates a fitted instance without data, e.g. for ``do()``,
    and ``fit`` only records the target and its parents, ignoring their values. ``predict_proba`` gives the distribution
    to every row of ``X``, and, for a root without ``X``, the distribution itself. A distribution whose parameters are
    numbers is rebuilt for the rows; any other, e.g. ``Empirical`` or ``ZeroInflated``, is repeated by skpro's ``IID``,
    which has no ``pmf``. The target is discrete for a ``NominalDistribution`` and continuous for any other
    distribution, whatever skpro's measure type of it. A ``NominalDistribution`` is a fixed table and a ``Normal`` a
    linear Gaussian with zero coefficients, so both have the ``capability:exact_inference`` tag.

    Requires the optional dependency ``skpro``.

    Parameters
    ----------
    distribution : skpro distribution
        A scalar distribution, e.g. ``skpro.distributions.Normal(mu=0, sigma=1)`` or a ``NominalDistribution``. An
        ``IID`` distribution stands for the scalar distribution it repeats.

    Attributes
    ----------
    distribution_ : skpro distribution
        A copy of the scalar distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distributions import Normal
    >>> from pgmpy.parameterization.adapters import DistributionAdapter
    >>> cpd = DistributionAdapter.from_values("A", Normal(mu=0.0, sigma=1.0))
    >>> dist = cpd.predict_proba(pd.DataFrame(index=[10, 11, 12]))
    >>> type(dist).__name__, dist.index.tolist(), dist.columns.tolist()
    ('Normal', [10, 11, 12], ['A'])
    >>> cpd.sample(n_samples=4, random_state=0).shape
    (4, 1)
    """

    _tags = {
        "name": "distribution_adapter",
        "variable_type": ["discrete", "continuous"],
        # The data is ignored, so weights change nothing.
        "supports_weighted_data": True,
        "requires_data": False,
        "python_dependencies": "skpro",
    }

    def __init__(self, distribution: "BaseDistribution") -> None:
        self.distribution = distribution
        super().__init__()

        if not isinstance(distribution, BaseDistribution):
            raise TypeError(f"distribution must be a skpro distribution, but is a {type(distribution).__name__}.")
        self._scalar = distribution.distribution if isinstance(distribution, IID) else distribution
        if self._scalar.ndim != 0:
            raise ValueError(f"distribution must be scalar, but has shape {self._scalar.shape}.")
        if isinstance(self._scalar, NominalDistribution):
            table = {"capability:factor": True, "capability:exact_inference": True}
            self.set_tags(**{"variable_type": ["discrete"], **table})
        else:
            self.set_tags(variable_type=["continuous"])
        if isinstance(self._scalar, Normal):
            gaussian = ("linearity", "additive_noise", "gaussian_noise")
            self.set_tags(**{"capability:exact_inference": True, **{f"assumption:{name}": True for name in gaussian}})

    @classmethod
    def from_values(
        cls, variable: Hashable, distribution: "BaseDistribution", evidence: list | tuple | None = None
    ) -> "DistributionAdapter":
        """Create a fitted DistributionAdapter without data.

        Parameters
        ----------
        variable : hashable
            Name of the target variable.
        distribution : skpro distribution
            The target's distribution, as for ``DistributionAdapter``.
        evidence : list or tuple, optional
            Names of the parent variables, whose values are ignored. ``None`` for a root variable.

        Returns
        -------
        DistributionAdapter
            A fitted instance. Its parents are sorted by name.
        """
        evidence = _checked_evidence(variable, evidence)
        cpd = cls(distribution)
        cpd.variable_ = variable
        cpd.evidence_ = [evidence[position] for position in _parent_order(evidence)]
        [cpd.variable_type_] = cpd.get_tag("variable_type")
        cpd.distribution_ = deepcopy(cpd._scalar)
        cpd._is_fitted = True
        return cpd

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight: np.ndarray | None) -> None:
        self.distribution_ = deepcopy(self._scalar)

    def _predict_proba(self, X: pd.DataFrame | None) -> Any:
        # Step 1: Without X, return a copy of a root's distribution itself.
        scalar = self.distribution_
        if X is None:
            return deepcopy(scalar)

        # Step 2: Repeat a NominalDistribution's probabilities for every row.
        if isinstance(scalar, NominalDistribution):
            return NominalDistribution(
                probs=np.tile(np.asarray(scalar.probs, dtype=float), (len(X), 1)),
                categories=list(scalar.categories),
                index=X.index,
                columns=[self.variable_],
            )
        # Step 3: Rebuild a distribution whose parameters are numbers with them for the rows; repeat any other by IID.
        params = {
            name: value for name, value in scalar.get_params(deep=False).items() if name not in ("index", "columns")
        }
        if all(isinstance(value, Real) and not isinstance(value, bool) for value in params.values()):
            numbers = {name: float(value) for name, value in params.items()}
            return type(scalar)(**numbers, index=X.index, columns=[self.variable_])
        return IID(deepcopy(scalar), index=X.index, columns=[self.variable_])

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or not (self.is_fitted and other.is_fitted):
            return super().__eq__(other)
        # skbase's == ignores the distributions' types, which _equal checks.
        return (self.variable_, self.evidence_) == (other.variable_, other.evidence_) and _equal(
            self.distribution_, other.distribution_
        )

    __hash__ = BaseParameter.__hash__
