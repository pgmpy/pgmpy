from typing import Hashable, List, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np
from itertools import product
from pgmpy.factors.discrete import TabularCPD


class CanonicalInfluence(ABC):
    """
    Abstract base class for canonical influence models in Bayesian networks.

    This class defines the interface and shared behavior for all canonical
    causal influence structures (e.g., Noisy-OR, Noisy-MAX, Noisy-MIN).

    Parameters
    ----------
    variable : Hashable
        The child node affected by the influence.
    evidence : list of Hashable
        Parent nodes (causes) influencing the variable.
    mode : {'MAX', 'MIN'}, default='MAX'
        Determines the type of combination rule.
    leak : array-like of float, optional
        Leak probability vector for spontaneous activation.
    state_names : list of str or int, optional
        Names of the possible states for the variable.
    """

    def __init__(
        self,
        variable: Hashable,
        evidence: List[Hashable],
        mode: str = "MAX",
        leak: Optional[np.ndarray] = None,
        state_names: Optional[List] = None,
    ):
        self.variable = variable
        self.evidence = evidence
        self.mode = mode.upper()
        if self.mode not in {"MAX", "MIN","OR","AND"}:
            raise ValueError("mode must be 'MAX' or 'MIN', or there binaries pending 'OR' or 'AND'")
        self.isleaky = leak is not None
        self.leak = np.array(leak) if leak is not None else None
        self.state_names = state_names

    # -------------------
    # Interface
    # -------------------
    @abstractmethod
    def evaluate(self, evidence_instantiate: Dict) -> np.ndarray:
        """Compute the probability distribution P(X | evidence_instantiate)."""
        pass

    @abstractmethod
    def to_cpd(self):
        """Convert the influence model into a pgmpy TabularCPD."""
        pass

    # -------------------
    # Shared Utilities
    # -------------------
    def _validate_probs(self, arr, name="probabilities"):
        arr = np.asarray(arr)
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be between 0 and 1.")
        if not np.isclose(arr.sum(), 1.0, atol=1e-8, rtol=1e-6):
            raise ValueError(f"{name} must sum to 1.")
        return arr

    def __repr__(self):
        rep = (f"""CanonicalInfluence: \n
        variable name: {self.variable} \n
        mode: {self.mode} \n
        evidence: {self.evidence} \n
        full states representation: {self.state_names} \n
        is_leaky: {self.isleaky} \n
        """)
        if self.isleaky: rep+=f"leak intensity: {self.leak} \n"

        return rep

    def __str__(self):
        return f"{self.__class__.__name__}({self.variable}, mode={self.mode}, leaky={self.isleaky})"

class BinaryInfluenceModel(CanonicalInfluence):
    """
    Canonical influence model for binary child variables
    (Noisy-OR / Noisy-AND / Leaky variations).

    This model is equivalent to the classic "Noisy-OR" or "Noisy-AND"
    canonical activation vector (CAV) model.

    Parameters
    ----------
    variable : str
        The name of the influenced variable (child node).

    evidence : list of str
        The list of parent nodes (causes) influencing the variable.

    activation_magnitude : list or array-like
        The activation probabilities for each evidence variable, i.e.
        P(X=1 | parent_i=1, others inactive).

    mode : {'OR', 'AND'}, default='OR'
        The combination scheme:
            - 'OR'  → Noisy-OR (independent causes for activation)
            - 'AND' → Noisy-AND (all causes needed for activation)

    leak : float, optional
        Probability that the variable is activated spontaneously (without any active parent).

    isboolean_style : bool, default=False
        Whether to interpret states as boolean (`True`/`False`) or numeric (`1`/`0`).

    Examples
    --------
    >>> cav = BinaryInfluenceModel(
    ...     variable='Disease',
    ...     evidence=['Fever', 'Cough', 'Fatigue'],
    ...     activation_magnitude=[0.6, 0.4, 0.2],
    ...     leak=0.05,
    ...     mode='OR'
    ... )
    >>> cav.evaluate({'Fever': 1, 'Cough': 0, 'Fatigue': 1})
    array([0.276, 0.724])   # [P(Disease=0), P(Disease=1)]
    >>> cpd = cav.to_cpd()
    """

    def __init__(
        self,
        variable,
        evidence,
        activation_magnitude,
        mode="OR",
        leak=None,
        isboolean_style=False,
    ):
        super().__init__(variable, evidence, mode=mode, leak=np.array([leak]) if leak is not None else None)
        
        self.mode = mode.upper()
        if self.mode not in {"OR", "AND"}:
            raise ValueError("mode must be either 'OR' or 'AND'")

        self.isboolean_style = isboolean_style
        self.activation_magnitude = np.asarray(activation_magnitude, dtype=float)

        if len(self.activation_magnitude) != len(self.evidence):
            raise ValueError("Number of activation magnitudes must match number of evidence variables.")
        if np.any((self.activation_magnitude < 0) | (self.activation_magnitude > 1)):
            raise ValueError("All activation probabilities must be in [0, 1].")
        if self.isleaky and not (0 <= self.leak[0] <= 1):
            raise ValueError("Leak value must be in [0, 1].")

        # Setup state names for pgmpy TabularCPD
        self.variable_card = 2
        self.state_names = {}
        full_variables = [self.variable] + list(self.evidence)
        if self.isboolean_style:
            for v in full_variables:
                self.state_names[v] = [False, True]
        else:
            for v in full_variables:
                self.state_names[v] = [0, 1]

    # ----------------------------------------------------------------------
    def evaluate(self, evidence_instantiate: dict) -> np.ndarray:
        """
        Compute the probability distribution [P(X=0), P(X=1)]
        given a specific evidence instantiation.
        """
        if set(evidence_instantiate.keys()) != set(self.evidence):
            raise ValueError(f"Evidence mismatch. Expected {self.evidence}, got {list(evidence_instantiate.keys())}")

        active_key = True if self.isboolean_style else 1
        active_mask = np.array([evidence_instantiate[e] == active_key for e in self.evidence])
        probs = self.activation_magnitude[active_mask]

        if self.mode == "OR":
            # Independent activation model: P(X=0) = Π(1 - p_i)
            p_active = 1 - np.prod(1 - probs)
            if self.isleaky:
                p_active = 1 - (1 - p_active) * (1 - self.leak[0])
        else:  # AND
            if np.any(~active_mask):
                # Some inactive parent — can’t activate unless leaky
                p_active = self.leak[0] if self.isleaky else 0.0
            else:
                p_active = np.prod(probs)
                if self.isleaky:
                    p_active = 1 - (1 - p_active) * (1 - self.leak[0])

        return np.array([1 - p_active, p_active])

    # ----------------------------------------------------------------------
    def to_cpd(self) -> TabularCPD:
        """
        Convert the model to a pgmpy TabularCPD.

        Returns
        -------
        TabularCPD
            The pgmpy-compatible conditional probability table.
        """
        parent_states = [self.state_names[e] for e in self.evidence]
        cols = []
        for combo in product(*parent_states):
            evidence_inst = dict(zip(self.evidence, combo))
            probs = self.evaluate(evidence_inst)
            cols.append(probs)

        cpd_values = np.array(cols).T
        cpd = TabularCPD(
            variable=self.variable,
            variable_card=self.variable_card,
            values=cpd_values,
            evidence=self.evidence,
            evidence_card=[2] * len(self.evidence),
            state_names=self.state_names,
        )
        return cpd

class MultilevelInfluenceModel(CanonicalInfluence):
    """
    Canonical model for multi-valued variables (Noisy-MAX / Noisy-MIN).

    Each parent has an influence table θ[parent][value][x] giving
    P(X=x | parent=value, others inactive).

    Parameters
    ----------
    influence_tables : dict
        Nested mapping defining influence probabilities per parent/value.
    levels : int
        Number of possible levels for the child.
    leak : array-like, optional
        Leak distribution for spontaneous activation.
    """

    def __init__(self, variable, evidence, influence_tables, levels, leak=None, mode="MAX", state_names=None):
        super().__init__(variable, evidence, mode=mode, leak=leak, state_names=state_names)
        self.levels = levels
        self.influence_tables = influence_tables

        # Validation
        for parent in evidence:
            for val, probs in influence_tables[parent].items():
                self._validate_probs(probs, f"influence[{parent}={val}]")

        # Build cumulative forms for efficient evaluation
        self.cumulative_tables = {
            p: {v: np.cumsum(probs) for v, probs in table.items()}
            for p, table in influence_tables.items()
        }
        self.cumulative_leak = (
            np.cumsum(leak) if leak is not None else np.ones(levels)
        )

    def evaluate(self, evidence_instantiate: dict) -> np.ndarray:
        """Compute P(X | evidence)."""
        if set(evidence_instantiate.keys()) != set(self.evidence):
            raise ValueError(f"Evidence mismatch. Expected {self.evidence}, got {list(evidence_instantiate)}")

        if self.mode == "MAX":
            cum_prob = np.ones(self.levels)
            for parent, val in evidence_instantiate.items():
                theta = self.cumulative_tables[parent][val]
                cum_prob *= theta
            if self.isleaky:
                cum_prob *= self.cumulative_leak

        elif self.mode == "MIN":
            complement_prod = np.ones(self.levels)
            for parent, val in evidence_instantiate.items():
                theta = self.cumulative_tables[parent][val]
                complement_prod *= (1 - theta)
            if self.isleaky:
                complement_prod *= (1 - self.cumulative_leak)
            cum_prob = 1 - complement_prod

        else:
            raise ValueError("mode must be either 'MAX' or 'MIN'")

        # Ensure monotonicity and numerical safety
        cum_prob = np.maximum.accumulate(np.clip(cum_prob, 0, 1))
        probs = np.diff(np.concatenate(([0.0], cum_prob)))
        probs = np.clip(probs, 0, 1)
        return probs / probs.sum()


    def to_cpd(self):
        """Convert to pgmpy TabularCPD."""
        from pgmpy.factors.discrete import TabularCPD

        parent_states = [list(self.influence_tables[p].keys()) for p in self.evidence]
        cols = []
        for combo in product(*parent_states):
            e = dict(zip(self.evidence, combo))
            probs = self.evaluate(e)
            cols.append(probs)

        values = np.vstack(cols).T
        cpd = TabularCPD(
            variable=self.variable,
            variable_card=self.levels,
            values=values,
            evidence=self.evidence,
            evidence_card=[len(st) for st in parent_states],
        )
        return cpd
