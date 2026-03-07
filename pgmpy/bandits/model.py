from itertools import product

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork


class CausalBanditModel:
    """
    Represents a causal bandit environment derived from a Bayesian Network.

    Wraps a ``DiscreteBayesianNetwork`` and exposes an arm/action interface
    where each arm corresponds to a hard or soft intervention on a subset of
    variables.  The reward is the value of a designated reward variable.

    Parameters
    ----------
    model : pgmpy.models.DiscreteBayesianNetwork
        A fitted Bayesian Network with CPDs.

    reward_variable : str
        Name of the node whose value is used as the reward signal.

    intervenable_variables : list of str
        Names of nodes the agent can intervene on.

    intervention_type : str (default: ``"hard"``)
        ``"hard"`` for atomic do-interventions, ``"soft"`` for stochastic /
        virtual interventions specified via ``TabularCPD`` objects.

    reward_type : str or None (default: ``None``)
        ``"binary"`` or ``"categorical"`` or ``"continuous"``.  If ``None``,
        auto-detected from the reward variable's CPD cardinality (2 → binary,
        otherwise categorical).

    reward_mapping : dict or None (default: ``None``)
        Maps reward-variable state names to numeric values, e.g.
        ``{"low": 0, "med": 0.5, "high": 1}``.  When ``None`` and the reward
        is binary, state names ``"0"``/``"1"`` are used as-is (cast to float).

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.bandits import CausalBanditModel
    >>> model = get_example_model("asia")
    >>> cbm = CausalBanditModel(
    ...     model, reward_variable="dysp", intervenable_variables=["smoke"]
    ... )
    >>> cbm.get_possible_interventions()  # doctest: +SKIP
    [{}, {'smoke': 'yes'}, {'smoke': 'no'}]
    """

    def __init__(
        self,
        model,
        reward_variable,
        intervenable_variables,
        intervention_type="hard",
        reward_type=None,
        reward_mapping=None,
    ):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must be a DiscreteBayesianNetwork instance.")

        nodes = set(model.nodes())
        if reward_variable not in nodes:
            raise ValueError(
                f"reward_variable '{reward_variable}' is not a node in the model."
            )

        for var in intervenable_variables:
            if var not in nodes:
                raise ValueError(
                    f"intervenable variable '{var}' is not a node in the model."
                )

        if reward_variable in intervenable_variables:
            raise ValueError("reward_variable cannot also be an intervenable variable.")

        if intervention_type not in ("hard", "soft"):
            raise ValueError(
                f"intervention_type must be 'hard' or 'soft', got '{intervention_type}'."
            )

        self.model = model
        self.reward_variable = reward_variable
        self.intervenable_variables = list(intervenable_variables)
        self.intervention_type = intervention_type
        self.reward_mapping = reward_mapping
        self._ci = CausalInference(model)

        # Per-variable intervention space: maps variable → list of allowed
        # values.  For hard interventions these are state-name strings; for
        # soft interventions they are ``TabularCPD`` objects.
        self._intervention_space = {}
        model_states = model.states
        for var in self.intervenable_variables:
            if intervention_type == "hard":
                self._intervention_space[var] = list(model_states[var])
            else:
                # Soft: no default values; user must call set_intervention_space
                self._intervention_space[var] = []

        # Reward type
        if reward_type is not None:
            self.reward_type = reward_type
        else:
            card = model.get_cpds(reward_variable).variable_card
            self.reward_type = "binary" if card == 2 else "categorical"

    # ------------------------------------------------------------------
    # Intervention space
    # ------------------------------------------------------------------
    def set_intervention_space(self, variable, values):
        """
        Restrict or define the intervention values for *variable*.

        Parameters
        ----------
        variable : str
            Must be one of ``intervenable_variables``.
        values : list
            For hard interventions: list of state-name strings.
            For soft interventions: list of ``TabularCPD`` objects.
        """
        if variable not in self.intervenable_variables:
            raise ValueError(f"'{variable}' is not an intervenable variable.")

        model_states = self.model.states
        if self.intervention_type == "hard":
            for v in values:
                if v not in model_states[variable]:
                    raise ValueError(
                        f"State '{v}' is not a valid state for variable '{variable}'. "
                        f"Valid states: {model_states[variable]}"
                    )
            self._intervention_space[variable] = list(values)
        else:
            for v in values:
                if not isinstance(v, TabularCPD):
                    raise ValueError(
                        "For soft interventions, values must be TabularCPD instances."
                    )
            self._intervention_space[variable] = list(values)

    def get_possible_interventions(self, include_observational=True):
        """
        Return the list of all possible interventions (arms).

        Parameters
        ----------
        include_observational : bool (default: ``True``)
            If ``True``, include the empty-dict observational arm ``{}``.

        Returns
        -------
        list of dict
            Each dict maps intervenable variable names to intervention values.
            For hard interventions the values are state-name strings; for soft
            interventions the values are ``TabularCPD`` objects.
        """
        if self.intervention_type == "hard":
            var_values = [
                [(var, v) for v in self._intervention_space[var]]
                for var in self.intervenable_variables
            ]
            actions = [
                {var: val for var, val in combo} for combo in product(*var_values)
            ]
        else:
            # Soft: each variable has a list of TabularCPD alternatives
            var_values = [
                [(var, cpd) for cpd in self._intervention_space[var]]
                for var in self.intervenable_variables
            ]
            if all(len(v) > 0 for v in var_values):
                actions = [
                    {var: val for var, val in combo} for combo in product(*var_values)
                ]
            else:
                actions = []

        if include_observational:
            actions = [{}] + actions
        return actions

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------
    def expected_reward(self, action):
        """
        Compute the expected reward distribution under *action*.

        Parameters
        ----------
        action : dict
            Intervention dict (hard: ``{var: state_str}``, or ``{}``
            for observational).

        Returns
        -------
        pgmpy.factors.discrete.DiscreteFactor
        """
        do = {k: v for k, v in action.items() if isinstance(v, str)}
        return self._ci.query(
            variables=[self.reward_variable], do=do if do else None, show_progress=False
        )

    def observe(self, action, n_samples=1, seed=None):
        """
        Draw reward samples by simulating from the model under *action*.

        Parameters
        ----------
        action : dict
            Intervention dict.
        n_samples : int
            Number of samples to draw.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with at least the reward-variable column.
        """
        if self.intervention_type == "hard" or not action:
            do = action if action else None
            samples = self.model.simulate(
                n_samples=n_samples,
                do=do,
                seed=seed,
                show_progress=False,
            )
        else:
            # Soft intervention: action values are TabularCPD objects
            virtual_intervention = list(action.values())
            samples = self.model.simulate(
                n_samples=n_samples,
                virtual_intervention=virtual_intervention,
                seed=seed,
                show_progress=False,
            )
        return samples

    def get_numeric_reward(self, observation):
        """
        Convert a sampled reward-variable value to a numeric scalar.

        Parameters
        ----------
        observation : pd.DataFrame
            Single-row (or multi-row) simulation output.

        Returns
        -------
        float
        """
        raw = observation[self.reward_variable].iloc[0]
        if self.reward_mapping is not None:
            return float(self.reward_mapping[raw])
        return float(raw)
