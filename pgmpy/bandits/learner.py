import numpy as np
import pandas as pd


class CausalBanditLearner:
    """
    Coordinates the interaction loop between a ``CausalBanditPolicy`` and a
    ``CausalBanditModel``.

    Parameters
    ----------
    model : pgmpy.bandits.CausalBanditModel
        The causal bandit environment.
    policy : pgmpy.bandits.CausalBanditPolicy
        The policy that selects arms.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.bandits import CausalBanditModel, CausalUCB, CausalBanditLearner
    >>> bn = get_example_model("asia")
    >>> cbm = CausalBanditModel(
    ...     bn, reward_variable="dysp", intervenable_variables=["smoke"]
    ... )
    >>> policy = CausalUCB(cbm)
    >>> learner = CausalBanditLearner(cbm, policy)
    >>> learner.run(n_rounds=100, seed=42)  # doctest: +SKIP
    >>> learner.get_history().head()  # doctest: +SKIP
    """

    def __init__(self, model, policy):
        self.model = model
        self.policy = policy
        self._history = []

    def run(self, n_rounds, seed=None):
        """
        Execute the bandit interaction loop for *n_rounds* steps.

        At each round the policy selects an action, the model returns a
        reward sample, and the policy is updated.

        Parameters
        ----------
        n_rounds : int
            Number of rounds to run.
        seed : int or None
            Base random seed.  Each round uses ``seed + round`` so that the
            full trajectory is reproducible.
        """
        if seed is not None:
            np.random.seed(seed)

        for t in range(n_rounds):
            action = self.policy.select_action()
            round_seed = (seed + t) if seed is not None else None
            observation = self.model.observe(action, n_samples=1, seed=round_seed)
            reward = self.model.get_numeric_reward(observation)
            self.policy.update(action, reward)

            record = {"round": t, "reward": reward}
            for var in self.model.intervenable_variables:
                record[var] = action.get(var, None)
            self._history.append(record)

    def get_history(self):
        """
        Return the interaction history as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``round``, one column per intervenable variable, ``reward``.
        """
        if not self._history:
            return pd.DataFrame(
                columns=["round"] + self.model.intervenable_variables + ["reward"]
            )
        return pd.DataFrame(self._history)
