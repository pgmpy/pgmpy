import itertools
from warnings import warn

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.base import BaseInference
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.utils import compat_fns


class ApproxInference(BaseInference):
    """Approximate Inference via Sampling.

    Parameters
    ----------
    n_samples: int
        The number of samples to generate for computing the distributions. Higher `n_samples`
        results in more accurate results at the cost of more computation time.

    samples: pd.DataFrame (default: None)
        If provided, uses these samples to compute the distribution instead
        of generating samples. `samples` **must** conform with the provided
        `evidence` and `virtual_evidence`.

    state_names: dict (default: None)
        A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
        If None, inferred from the data but is possible that the final distribution misses some states.

    show_progress: boolean (default: True)
        If True, shows a progress bar when generating samples.

    seed: int (default: None)
        Sets the seed for the random generators.

    Examples
    --------
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("alarm")
    >>> infer = ApproxInference(model)
    """

    def __init__(
        self,
        *args,
        n_samples=int(1e4),
        samples=None,
        joint=True,
        state_names=None,
        show_progress=True,
        seed=None,
        **kwargs,
    ):
        self.n_samples = n_samples
        self.samples = samples
        self.joint = joint
        self.state_names = state_names
        self.show_progress = show_progress
        self.seed = seed

        msg = (
            "Passing model to __init__ of inference classes is deprecated, "
            "and will raise an exception in pgmpy 2.0. "
            "Please pass model an argument to query."
        )

        if len(args) > 0:
            self._model = args[0]
            warn(msg, FutureWarning)
        else:
            self._model = None

        if kwargs is not None:
            model = kwargs.get("model", None)
            if model is not None:
                warn(msg, FutureWarning)
                self._model = model

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
        if isinstance(variables, (set, tuple)):
            variables = list(variables)

        if joint == True:
            return self._get_factor_from_df(
                samples.groupby(variables, observed=False).size() / samples.shape[0],
                state_names,
            )
        else:
            return {
                var: self._get_factor_from_df(
                    samples.groupby([var], observed=False).size() / samples.shape[0],
                    state_names,
                )
                for var in variables
            }

    def _handle_deprec_args(self, args, kwargs):
        """Utility to handle deprecated args for query methods."""

        msg = (
            "Passing parameters to query method of inference algorithms"
            " is deprecated, and will raise an exception in pgmpy 2.0. "
            "Please pass parameters to __init__ of inference class.",
            FutureWarning,
        )
        defaults = {
            "n_samples": self.n_samples,
            "samples": self.samples,
            "state_names": self.state_names,
            "show_progress": self.show_progress,
            "seed": self.seed,
        }
        var_names = [
            "variables",
            "n_samples",
            "samples",
            "evidence",
            "virtual_evidence",
            "joint",
            "state_names",
            "show_progress",
            "seed",
        ]
        # handle deprecated args
        from pgmpy.utils._deprecation import _handle_deprec_args

        final_args = _handle_deprec_args(args, kwargs, var_names, defaults, msg)
        return final_args

    def query(
        self,
        *args,
        model=None,
        variables=None,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        **kwargs,
    ):
        """Query the probability distribution from model, for variables.

        Method for doing approximate inference based on sampling in Bayesian
        Networks and Dynamic Bayesian Networks.

        Parameters
        ----------
        model: Instance of models.DiscreteBayesianNetwork or DynamicBayesianNetwork
            The probabilistic graphical model to do inference on.

        variables: list, optional, default=all variables in the model
            List of variables to calculate the probability distribution for.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        joint: boolean, optional, default=True
            If joint=True, computes the joint distribution over `variables`.
            Else, returns a dict with marginal distribution of each variable in
            `variables`.

        Returns
        -------
        Probability distribution or list thereof, of type factors.discrete.TabularCPD
            The queried probability distribution.

            * if `joint=True`, returns a single TabularCPD representing
              the joint distribution
            * if `joint=False`, returns a dict of TabularCPDs, these represent
              marginal distributions of each variable in `variables`,
              in the same order.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.inference import ApproxInference
        >>> model = get_example_model("alarm")
        >>> infer = ApproxInference()
        >>> infer.query(model, variables=["HISTORY"])
        <DiscreteFactor representing phi(HISTORY:2) at 0x7f92d9f5b910>
        >>> infer.query(model, variables=["HISTORY", "CVP"], joint=True)
        <DiscreteFactor representing phi(HISTORY:2, CVP:3) at 0x7f92d9f77610>
        >>> infer.query(model, variables=["HISTORY", "CVP"], joint=False)
        {'HISTORY': <DiscreteFactor representing phi(HISTORY:2) at 0x7f92dc61eb50>,
         'CVP': <DiscreteFactor representing phi(CVP:3) at 0x7f92d915ec40>}
        """
        if model is None and self._model is not None:
            model = self._model

        if not isinstance(model, (DiscreteBayesianNetwork, DynamicBayesianNetwork)):
            raise ValueError(
                f"model should either be a DiscreteBayesianNetwork "
                f"or a DynamicBayesianNetwork. Got {type(model)}."
            )
        model.check_model()

        # handle defaults
        # this seems to fail. todo: investigate
        # if variables is None:
        #     variables = list(model.nodes)

        if evidence is None:
            evidence = dict()
        if virtual_evidence is None:
            virtual_evidence = dict()

        final_args = self._handle_deprec_args(args, kwargs)

        n_samples = final_args["n_samples"]
        samples = final_args["samples"]
        state_names = final_args["state_names"]
        show_progress = final_args["show_progress"]
        seed = final_args["seed"]

        # Step 1: If samples are not provided, generate samples for the query
        if samples is None:
            simulate_kwargs = {
                "n_samples": n_samples,
                "evidence": evidence,
                "virtual_evidence": virtual_evidence,
                "show_progress": show_progress,
                "seed": seed,
            }

            # default for time_slices in DBN
            if isinstance(model, DynamicBayesianNetwork):
                max_time_slices = 0
                for var in variables:
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for var, _ in evidence.items():
                    if var[1] > max_time_slices:
                        max_time_slices = var[1]
                for cpd in virtual_evidence:
                    if cpd.variable[1] > max_time_slices:
                        max_time_slices = cpd.variable[2]
                simulate_kwargs["n_time_slices"] = max_time_slices + 1

            samples = model.simulate(**simulate_kwargs)

        # Step 2: If state_names is None, infer it from samples.
        if state_names is None:
            if isinstance(model, DiscreteBayesianNetwork):
                state_names = {
                    var: list(samples.loc[:, var].unique()) for var in variables
                }
            elif isinstance(model, DynamicBayesianNetwork):
                state_names = {
                    var: list(samples.loc[:, [var]].iloc[:, 0].unique())
                    for var in variables
                }

        # Step 3: Compute the distributions and return it.
        return self.get_distribution(
            samples, variables=variables, state_names=state_names, joint=joint
        )

    def map_query(
        self,
        *args,
        model=None,
        variables=None,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        **kwargs,
    ):
        """Query most probable states from model, for variables.

        Finds the most probable state in the joint distribution of variables. Calculates the
        result by generating samples and calculating most probable states based on the probabilities.

        Parameters
        ----------
        model: Instance of models.DiscreteBayesianNetwork or DynamicBayesianNetwork
            The probabilistic graphical model to do inference on.

        variables: list, optional, default=all variables in the model
            List of variables to calculate the probability distribution for.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        joint: boolean, optional, default=True
            If joint=True, computes the joint distribution over `variables`.
            Else, returns a dict with marginal distribution of each variable in
            `variables`.

        Returns
        -------
        MAP values: dict
            The most probable state of provided `variables` given the evidence.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.inference import ApproxInference
        >>> from pgmpy.factors.discrete import State, TabularCPD
        >>> model = get_example_model("alarm")
        >>> infer = ApproxInference(model)
        >>> print(infer.map_query(variables=["HISTORY", "CVP"]))
        {'HISTORY': 'FALSE', 'CVP': 'NORMAL'}
        >>> virtual_evidence_history = TabularCPD(
        ...     variable="HISTORY",
        ...     variable_card=2,
        ...     values=[[0.99], [0.01]],
        ...     state_names={"HISTORY": ["TRUE", "FALSE"]},
        ... )
        >>> evidence = {"CVP": "NORMAL"}
        >>> print(
        ...     infer.map_query(
        ...         variables=["HISTORY"],
        ...         evidence=evidence,
        ...         virtual_evidence=[virtual_evidence_history],
        ...     )
        ... )
        {'HISTORY': 'TRUE'}
        """
        final_distribution = self.query(
            *args,
            model=model,
            joint=joint,
            variables=variables,
            evidence=evidence,
            virtual_evidence=virtual_evidence,
            **kwargs,
        )

        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results
