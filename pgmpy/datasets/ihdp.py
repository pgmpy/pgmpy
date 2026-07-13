from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset


class IHDPDataset(BaseSimulatedDataset):
    """Semi-synthetic dataset from the Infant Health and Development Program.

    Generates outcomes using real covariates from the IHDP trial with
    simulated response surfaces, following the benchmark introduced by
    Hill (2011) :cite:p:`hill_2011`.

    The 25 pre-treatment covariates :math:`X` (6 continuous, 19 binary)
    and binary treatment indicator :math:`T` are fixed across
    replications. Potential outcomes :math:`Y(0)` and :math:`Y(1)` are
    generated from parameterized response surfaces with additive noise.

    Covariates are named ``x1``-``x25`` throughout (DataFrame columns,
    ``load_ground_truth()``'s DAG, this docstring) to match how every
    IHDP-reporting paper and package refers to them. For what each xN
    actually represents (e.g., ``x1`` = birth weight), see
    ``COVARIATE_INFO`` in the companion ``prep.py``.

    .. note::
       ``x14`` ("first" — firstborn indicator) is coded ``{1, 2}``,
       not ``{0, 1}``, matching the literature-standard NPCI/CEVAE
       convention. This is not a bug.

    Parameters
    ----------
    setting : str, default ``"B"``
        Response surface setting, following Hill (2011)'s own naming
        (also used by EconML's ``ihdp_surface_A``/``ihdp_surface_B``).
        ``"A"`` produces a linear surface with homogeneous treatment
        effect :math:`\\omega`. ``"B"`` produces a nonlinear control
        surface with heterogeneous effects — this is the surface used
        by the standard published IHDP-100/1000 benchmark replications.

        .. note::
           Dorie's NPCI R package labels this same nonlinear surface
           ``setting="A"`` in its own CLI — the letter is swapped
           relative to Hill's paper. We follow Hill (2011) and EconML's
           naming, not NPCI's internal parameter name.
    omega : float, default 4.0
        Target average treatment effect on the treated (ATT). For
        ``setting="A"``, used directly as the constant per-unit
        effect (every unit's ITE equals ``omega`` exactly). For
        ``setting="B"``, :math:`\\mu_1` is shifted by a derived
        calibration constant so the realized ATT *among treated
        units* equals ``omega`` exactly, even though individual
        effects vary — matching Hill (2011) :cite:p:`hill_2011` and
        EconML's ``ihdp_surface_B`` calibration step.
    noise : distribution object, optional
        Any object with a ``.sample(n_samples=...)`` or
        ``.rvs(size=...)`` method (e.g., ``scipy.stats`` or
        ``skpro`` distributions). When ``None``, standard
        normal :math:`\\mathcal{N}(0, 1)` noise is used.
    seed : int, optional
        Random seed for reproducible coefficient sampling and noise.

    References
    ----------
    - :cite:p:`hill_2011`
    - :cite:p:`johansson_2016`
    """

    _tags = {
        "name": "ihdp",
        "has_ground_truth": True,
        "is_continuous": True,
        "is_mixed": True,
    }

    def __init__(
        self,
        setting: str = "B",
        omega: float = 4.0,
        noise: Any = None,
        seed: int | None = None,
    ):
        if setting not in ("A", "B"):
            raise ValueError(f"Unknown setting '{setting}'. Supported: 'A', 'B'.")

        self.setting = setting
        self.omega = omega
        self.noise = noise
        self.seed = seed

        # Load real covariates + treatment (fixed across replications).
        self._covariates, self._treatment = self._load_covariates()

        # Sample coefficients (varies per seed = one "replication").
        # Distribution depends on setting — see _sample_coefficients.
        rng = np.random.default_rng(seed)
        self._beta = self._sample_coefficients(self._covariates.shape[1], rng, setting)
        self._rng = rng

        # Compute response surfaces (deterministic given X and beta).
        if setting == "A":
            self._mu0, self._mu1 = self._response_surface_a(self._covariates, self._beta, omega)
        else:
            mu0, raw_mu1 = self._response_surface_b(self._covariates, self._beta)
            # Solve for the additive shift so the realized ATT among
            # treated units equals `omega` exactly — this is the
            # calibration step from Hill (2011) / NPCI / EconML's
            # ihdp_surface_B.
            shift = (raw_mu1 - mu0)[self._treatment == 1].mean() - omega
            self._mu0, self._mu1 = mu0, raw_mu1 - shift

    def load_dataframe(self, n_samples: int | None = None) -> pd.DataFrame:
        """Generate one replication of the IHDP semi-synthetic dataset.

        Parameters
        ----------
        n_samples : int, optional
            Ignored for IHDP. Sample size is fixed at 747 (real
            covariates). A warning is issued if provided.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``treatment``, ``y_factual``,
            ``y_cfactual``, ``mu0``, ``mu1``, ``x1`` through ``x25``.
        """
        if n_samples is not None:
            warnings.warn(
                "n_samples is ignored for IHDP; sample size is fixed at 747.",
                UserWarning,
                stacklevel=2,
            )

        n = len(self._treatment)

        if self.noise is None:
            noise_0 = self._rng.normal(0, 1, size=n)
            noise_1 = self._rng.normal(0, 1, size=n)
        elif hasattr(self.noise, "sample"):
            noise_0 = np.asarray(self.noise.sample(n_samples=n)).flatten()
            noise_1 = np.asarray(self.noise.sample(n_samples=n)).flatten()
        elif hasattr(self.noise, "rvs"):
            noise_0 = np.asarray(self.noise.rvs(size=n)).flatten()
            noise_1 = np.asarray(self.noise.rvs(size=n)).flatten()
        else:
            raise TypeError(f"noise must have a .sample() or .rvs() method, got {type(self.noise).__name__}.")

        y0 = self._mu0 + noise_0
        y1 = self._mu1 + noise_1

        t = self._treatment
        y_factual = np.where(t == 1, y1, y0)
        y_cfactual = np.where(t == 1, y0, y1)

        result = pd.DataFrame(self._covariates, columns=[f"x{i}" for i in range(1, 26)])
        result.insert(0, "treatment", t)
        result.insert(1, "y_factual", y_factual)
        result.insert(2, "y_cfactual", y_cfactual)
        result.insert(3, "mu0", self._mu0)
        result.insert(4, "mu1", self._mu1)

        return result

    def load_ground_truth(self) -> DAG:
        """Return the ground-truth causal DAG with roles.

        The returned DAG has roles set for direct use with pgmpy's
        prediction estimators:

        - ``exposures``: ``"treatment"``
        - ``outcomes``: ``"y_factual"``
        - ``adjustment``: ``["x1", ..., "x25"]``

        Returns
        -------
        DAG
        """
        covariate_names = [f"x{i}" for i in range(1, 26)]
        edges = (
            [(cov, "treatment") for cov in covariate_names]
            + [(cov, "y_factual") for cov in covariate_names]
            + [("treatment", "y_factual")]
        )
        return DAG(
            ebunch=edges,
            roles={
                "exposures": "treatment",
                "outcomes": "y_factual",
                "adjustment": covariate_names,
            },
        )

    @staticmethod
    def _load_covariates() -> tuple[np.ndarray, np.ndarray]:
        """Load IHDP covariates and treatment from HuggingFace Hub.

        Continuous columns (x1-x6) are already standardized (zero mean,
        unit variance) in ``ihdp_covariates.csv`` — verified at
        data-prep time (see ``prep.py``), not re-checked on every load.

        Returns
        -------
        X : np.ndarray of shape (747, 25)
            Covariates: 6 pre-standardized continuous, 19 binary.
        treatment : np.ndarray of shape (747,)
            Binary treatment indicator.
        """
        raw = IHDPDataset._get_raw_data("IHDP/ihdp_covariates.csv")
        df = pd.read_csv(io.BytesIO(raw))
        treatment = df["treatment"].values
        X = df.drop(columns=["treatment"]).values
        return X, treatment

    @staticmethod
    def _sample_coefficients(n_features: int, rng: np.random.Generator, setting: str) -> np.ndarray:
        """Sample coefficients from the Hill (2011) distribution.

        Returns ``n_features + 1`` values: index 0 is the coefficient
        on an intercept term, indices 1..n_features are the
        per-covariate coefficients.

        Setting ``"A"`` (linear, homogeneous): {0, 1, 2, 3, 4} with
        probabilities {0.5, 0.2, 0.15, 0.1, 0.05}.

        Setting ``"B"`` (nonlinear, heterogeneous): {0, 0.1, 0.2, 0.3,
        0.4} with probabilities {0.6, 0.1, 0.1, 0.1, 0.1} — an order
        of magnitude smaller, since these coefficients feed an
        exponential and would blow up at Setting A's scale.

        Parameters
        ----------
        n_features : int
            Number of covariates.
        rng : np.random.Generator
            Random number generator.
        setting : str
            ``"A"`` or ``"B"``.

        Returns
        -------
        np.ndarray of shape (n_features + 1,)
        """
        if setting == "A":
            return rng.choice(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                size=n_features + 1,
                p=[0.5, 0.2, 0.15, 0.1, 0.05],
            )
        return rng.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4],
            size=n_features + 1,
            p=[0.6, 0.1, 0.1, 0.1, 0.1],
        )

    @staticmethod
    def _response_surface_a(X, beta, omega):
        """Setting A: linear, homogeneous treatment effect.

        An intercept column of 1s is prepended internally before
        applying ``beta``; it is never exposed to the user.

        .. math::

            \\mu_0(X) = [\\mathbf{1}, X] \\beta, \\quad
            \\mu_1(X) = \\mu_0(X) + \\omega

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
        beta : np.ndarray of shape (p + 1,)
        omega : float

        Returns
        -------
        tuple of (mu0, mu1), each np.ndarray of shape (n,)
        """
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        mu0 = X_aug @ beta
        mu1 = mu0 + omega
        return mu0, mu1

    @staticmethod
    def _response_surface_b(X, beta):
        """Setting B: nonlinear control surface, linear treated surface.

        An intercept column of 1s is prepended internally. The offset
        :math:`W = 0.5` applies only to the 25 real covariates (zero
        on the intercept) and only inside the exponential control
        surface.

        Returns the **uncalibrated** ``mu1`` — the omega shift depends
        on the treatment assignment, so calibration happens in
        ``__init__`` instead.

        .. math::

            \\mu_0(X) = \\exp\\bigl(([\\mathbf{1}, X] + W) \\beta\\bigr),
            \\quad \\mu_1^{\\text{raw}}(X) = [\\mathbf{1}, X] \\beta

        where :math:`W = (0, 0.5, \\ldots, 0.5)` — zero on the
        intercept, 0.5 on each of the 25 covariates.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
        beta : np.ndarray of shape (p + 1,)

        Returns
        -------
        tuple of (mu0, mu1_raw), each np.ndarray of shape (n,)
        """
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        offset = np.zeros_like(X_aug)
        offset[:, 1:] = 0.5
        mu0 = np.exp((X_aug + offset) @ beta)
        mu1_raw = X_aug @ beta
        return mu0, mu1_raw
