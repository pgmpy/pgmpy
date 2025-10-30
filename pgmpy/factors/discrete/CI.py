from typing import Hashable, List, Dict, Optional
import numpy as np
from itertools import product
from pgmpy.factors.discrete import TabularCPD


class BinaryInfluenceModel(TabularCPD):
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
    >>> # The created object is a TabularCPD
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
        self.mode = mode.upper()
        if self.mode not in {"OR", "AND"}:
            raise ValueError("mode must be either 'OR' or 'AND'")

        self.isboolean_style = isboolean_style
        self.activation_magnitude = np.asarray(activation_magnitude, dtype=float)
        self.leak = np.array([leak]) if leak is not None else None
        self.isleaky = leak is not None
        self.evidence = evidence

        if len(self.activation_magnitude) != len(evidence):
            raise ValueError("Number of activation magnitudes must match number of evidence variables.")
        if np.any((self.activation_magnitude < 0) | (self.activation_magnitude > 1)):
            raise ValueError("All activation probabilities must be in [0, 1].")
        if self.isleaky and not (0 <= self.leak[0] <= 1):
            raise ValueError("Leak value must be in [0, 1].")

        # Setup state names for pgmpy TabularCPD
        variable_card = 2
        state_names = {}
        full_variables = [variable] + list(self.evidence)
        if self.isboolean_style:
            for v in full_variables:
                state_names[v] = [False, True]
        else:
            for v in full_variables:
                state_names[v] = [0, 1]

        parent_states = [state_names[e] for e in self.evidence]
        cols = []
        for combo in product(*parent_states):
            evidence_inst = dict(zip(self.evidence, combo))
            probs = self._evaluate(evidence_inst)
            cols.append(probs)

        cpd_values = np.array(cols).T
        
        super().__init__(
            variable=variable,
            variable_card=variable_card,
            values=cpd_values,
            evidence=evidence,
            evidence_card=[2] * len(evidence),
            state_names=state_names,
        )

    def _evaluate(self, evidence_instantiate: dict) -> np.ndarray:
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
            p_active = 1 - np.prod(1 - probs)
            if self.isleaky:
                p_active = 1 - (1 - p_active) * (1 - self.leak[0])
        else:  # AND
            if np.any(~active_mask):
                p_active = self.leak[0] if self.isleaky else 0.0
            else:
                p_active = np.prod(probs)
                if self.isleaky:
                    p_active = 1 - (1 - p_active) * (1 - self.leak[0])

        return np.array([1 - p_active, p_active])


class MultilevelInfluenceModel(TabularCPD):
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
        self.mode = mode.upper()
        if self.mode not in {"MAX", "MIN"}:
            raise ValueError("mode must be 'MAX' or 'MIN'")
            
        self.levels = levels
        self.influence_tables = influence_tables
        self.leak = np.array(leak) if leak is not None else None
        self.isleaky = leak is not None
        self.evidence = evidence

        for parent in self.evidence:
            for val, probs in influence_tables[parent].items():
                self._validate_probs(probs, f"influence[{parent}={val}]")

        self.cumulative_tables = {
            p: {v: np.cumsum(probs) for v, probs in table.items()}
            for p, table in influence_tables.items()
        }
        self.cumulative_leak = (
            np.cumsum(leak) if leak is not None else np.ones(levels)
        )

        parent_states = [list(self.influence_tables[p].keys()) for p in self.evidence]
        cols = []
        for combo in product(*parent_states):
            e = dict(zip(evidence, combo))
            probs = self._evaluate(e)
            cols.append(probs)

        values = np.vstack(cols).T
        
        super().__init__(
            variable=variable,
            variable_card=self.levels,
            values=values,
            evidence=evidence,
            evidence_card=[len(st) for st in parent_states],
            state_names=state_names
        )

    def _validate_probs(self, arr, name="probabilities"):
        arr = np.asarray(arr)
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be between 0 and 1.")
        if not np.isclose(arr.sum(), 1.0, atol=1e-8, rtol=1e-6):
            raise ValueError(f"{name} must sum to 1.")
        return arr

    def _evaluate(self, evidence_instantiate: dict) -> np.ndarray:
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

        cum_prob = np.maximum.accumulate(np.clip(cum_prob, 0, 1))
        probs = np.diff(np.concatenate(([0.0], cum_prob)))
        probs = np.clip(probs, 0, 1)
        return probs / probs.sum()
