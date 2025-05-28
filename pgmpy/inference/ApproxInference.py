import itertools

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.utils import compat_fns


class ApproxInference(object):
    """
    Initializes the Approximate Inference class.

    Parameters
    ----------
    model: Instance of pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.DynamicBayesianNetwork

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model('alarm')
    >>> infer = ApproxInference(model)
    """

    def __init__(self, model):
        if not isinstance(model, (DiscreteBayesianNetwork, DynamicBayesianNetwork)):
            raise ValueError(
                f"model should either be a Bayesian Network or Dynamic Bayesian Network. Got {type(model)}."
            )
        model.check_model()
        self.model = model

    @staticmethod
    def _get_factor_from_df(df, model_states):
        """
        Takes a groupby dataframe and converts it into a pgmpy.factors.discrete.DiscreteFactor object.

        Parameters
        ----------
        df: pandas.DataFrame
            A groupby dataframe containing the counts/probabilities.

        model_states: dict
            A dict of state names for each variable from the model in the form {variable_name: list of states}.
        """
        if df.empty:
            raise ValueError("Cannot create factor from empty dataframe")

        variables = list(df.index.names)
        if not variables or None in variables:
            raise ValueError("DataFrame must have valid index names")

        if len(variables) == 1:
            df_index = model_states[variables[0]]
        else:
            df_index = itertools.product(*[model_states[var] for var in variables])
        cardinality = [len(model_states[var]) for var in variables]
        return DiscreteFactor(
            variables=variables,
            cardinality=cardinality,
            values=df.reindex(df_index).fillna(0).values,
            state_names=model_states,
        )

    def get_distribution(self, samples, variables, joint=True):
        """
        Computes distribution of `variables` from given data `samples`.

        Parameters
        ----------
        samples: pandas.DataFrame
            A dataframe of samples generated from the model.

        variables: list (array-like)
            A list of variables whose distribution needs to be computed.

        joint: boolean
            If joint=True, computes the joint distribution over `variables`.
            Else, returns a dict with marginal distribution of each variable in
            `variables`.
        """
        if isinstance(variables, (set, tuple)):
            variables = list(variables)

        # Get state names from the model
        model_states = {var: self.model.states[var] for var in variables}

        if joint == True:
            return self._get_factor_from_df(
                samples.groupby(variables).size() / samples.shape[0], model_states
            )
        else:
            return {
                var: self._get_factor_from_df(
                    samples.groupby([var]).size() / samples.shape[0],
                    {var: model_states[var]},
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

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

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
            if isinstance(self.model, DiscreteBayesianNetwork):
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

                # Validate time slices in evidence
                for var, state in evidence.items():
                    if not isinstance(var, tuple) or len(var) != 2:
                        raise ValueError(
                            f"Invalid variable format in evidence: {var}. Expected (node, time_slice)."
                        )
                    if var[1] < 0:
                        raise ValueError(
                            f"Invalid time slice in evidence: {var[1]}. Time slice must be non-negative."
                        )
                    if var not in self.model.states:
                        raise KeyError(f"Variable {var} not found in model states.")
                # Validate virtual evidence normalization and time slices
                for cpd in virtual_evidence:
                    values = (
                        cpd.values.detach().cpu().numpy()
                        if hasattr(cpd.values, "detach")
                        else cpd.values
                    )
                    if not (abs(sum(values) - 1.0) < 1e-3):
                        raise ValueError(
                            f"Virtual evidence CPD for {cpd.variable} is not normalized."
                        )
                    if cpd.variable not in self.model.states:
                        raise ValueError(
                            f"Virtual evidence variable {cpd.variable} not found in model states."
                        )

                max_time_slices = 0
                for var in variables:
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for var, state in evidence.items():
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for cpd in virtual_evidence:
                    if cpd.variable[1] > max_time_slices:
                        max_time_slices = cpd.variable[1]
                samples = self.model.simulate(
                    n_samples=n_samples,
                    n_time_slices=max_time_slices + 1,
                    evidence=evidence,
                    virtual_evidence=virtual_evidence,
                    show_progress=show_progress,
                    seed=seed,
                )

        # Step 2: Compute the distributions and return it.
        return self.get_distribution(samples, variables=variables, joint=joint)

    def map_query(
        self,
        variables,
        n_samples=int(1e4),
        samples=None,
        evidence=None,
        virtual_evidence=None,
        show_progress=True,
        seed=None,
    ):
        """
        Finds the most probable state in the joint distribution of variables. Calculates the
        result by generating samples and calculating most probable states based on the probabilities.

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

        show_progress: boolean (default: True)
            If True, shows a progress bar when generating samples.

        seed: int (default: None)
            Sets the seed for the random generators.

        Returns
        -------
        MAP values: dict
            The most probable state of provided `variables` given the evidence.
        """
        final_distribution = self.query(
            variables,
            n_samples=n_samples,
            samples=samples,
            evidence=evidence,
            virtual_evidence=virtual_evidence,
            joint=True,
            show_progress=show_progress,
            seed=seed,
        )

        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results
