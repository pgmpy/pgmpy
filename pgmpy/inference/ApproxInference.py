import itertools

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork


class ApproxInference(object):
    """
    Initializes the Approximate Inference class.

    Parameters
    ----------
    model: Instance of pgmpy.models.BayesianNetwork or pgmpy.models.DynamicBayesianNetwork

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model('alarm')
    >>> infer = ApproxInference(model)
    """

    def __init__(self, model):
        if not isinstance(model, (BayesianNetwork, DynamicBayesianNetwork)):
            raise ValueError(
                f"model should either be a Bayesian Network or Dynamic Bayesian Network. Got {type(model)}."
            )
        model.check_model()
        self.model = model

    @staticmethod
    def _get_factor_from_df(df, state_names):
        """
        Takes a groupby dataframe and converts it into a pgmpy.factors.discrete.DiscreteFactor object.
        """
        variables = list(df.index.names)
        if len(variables) == 1:
            df_index = state_names[variables[0]]
        else:
            df_index = itertools.product(*[state_names[var] for var in variables])
        # state_names = {var: list(df.index.unique(var)) for var in variables}
        cardinality = [len(state_names[var]) for var in variables]
        return DiscreteFactor(
            variables=variables,
            cardinality=cardinality,
            values=df.reindex(df_index).fillna(0).values,
            state_names=state_names,
        )

    def get_distribution(self, samples, variables, state_names=None, joint=True):
        """
        Computes distribution of `variables` from given data `samples`.

        Parameters
        ----------
        samples: pandas.DataFrame
            A dataframe of samples generated from the model.

        variables: list (array-like)
            A list of variables whose distribution needs to be computed.

        state_names: dict (default: None)
            A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
            If None, inferred from the data but is possible that the final distribution misses some states.

        joint: boolean
            If joint=True, computes the joint distribution over `variables`.
            Else, returns a dict with marginal distribution of each variable in
            `variables`.
        """
        if joint == True:
            return self._get_factor_from_df(
                samples.groupby(variables).size() / samples.shape[0], state_names
            )
        else:
            return {
                var: self._get_factor_from_df(
                    samples.groupby([var]).size() / samples.shape[0], state_names
                )
                for var in variables
            }

    def query(
        self,
        variables,
        n_samples=int(1e4),
        samples=None,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        state_names=None,
        show_progress=True,
        seed=None,
    ):
        """
        Method for doing approximate inference based on sampling in Bayesian
        Networks and Dynamic Bayesian Networks.

        Parameters
        ----------
        variables: list
            List of variables for which the probability distribution needs to be calculated.

        n_samples: int
            The number of samples to generate for computing the distributions. Higher `n_samples`
            results in more accurate results at the cost of more computation time.

        samples: pd.DataFrame (default: None)
            If provided, uses these samples to compute the distribution instead
            of generating samples. `samples` **must** conform with the provided
            `evidence` and `virtual_evidence`.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        state_names: dict (default: None)
            A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
            If None, inferred from the data but is possible that the final distribution misses some states.

        show_progress: boolean (default: True)
            If True, shows a progress bar when generating samples.

        seed: int (default: None)
            Sets the seed for the random generators.

        Returns
        -------
        Probability distribution: pgmpy.factors.discrete.TabularCPD
            The queried probability distribution.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.inference import ApproxInference
        >>> model = get_example_model("alarm")
        >>> infer = ApproxInference(model)
        >>> infer.query(variables=["HISTORY"])
        <DiscreteFactor representing phi(HISTORY:2) at 0x7f92d9f5b910>
        >>> infer.query(variables=["HISTORY", "CVP"], joint=True)
        <DiscreteFactor representing phi(HISTORY:2, CVP:3) at 0x7f92d9f77610>
        >>> infer.query(variables=["HISTORY", "CVP"], joint=False)
        {'HISTORY': <DiscreteFactor representing phi(HISTORY:2) at 0x7f92dc61eb50>,
         'CVP': <DiscreteFactor representing phi(CVP:3) at 0x7f92d915ec40>}
        """
        # Step 1: If samples are not provided, generate samples for the query
        if samples is None:
            if isinstance(self.model, BayesianNetwork):
                samples = self.model.simulate(
                    n_samples=n_samples,
                    evidence=evidence,
                    virtual_evidence=virtual_evidence,
                    seed=seed,
                    show_progress=show_progress,
                )
            elif isinstance(self.model, DynamicBayesianNetwork):
                if evidence is None:
                    evidence = dict()
                if virtual_evidence is None:
                    virtual_evidence = dict()

                max_time_slices = 0
                for var in variables:
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for var, state in evidence.items():
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for cpd in virtual_evidence:
                    if cpd.variable[1] > max_time_slices:
                        max_time_slices = cpd.variable[2]
                samples = self.model.simulate(
                    n_samples=n_samples,
                    n_time_slices=max_time_slices + 1,
                    evidence=evidence,
                    virtual_evidence=virtual_evidence,
                    show_progress=show_progress,
                    seed=seed,
                )

        # Step 2: If state_names is None, infer it from samples.
        if state_names is None:
            if isinstance(self.model, BayesianNetwork):
                state_names = {
                    var: list(samples.loc[:, var].unique()) for var in variables
                }
            elif isinstance(self.model, DynamicBayesianNetwork):
                state_names = {
                    var: list(samples.loc[:, [var]].iloc[:, 0].unique())
                    for var in variables
                }

        # Step 3: Compute the distributions and return it.
        return self.get_distribution(
            samples, variables=variables, state_names=state_names, joint=joint
        )
