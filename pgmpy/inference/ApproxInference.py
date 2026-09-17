import itertools

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.utils import compat_fns


class ApproxInference:
    def __init__(self, model):
        if not isinstance(model, (DiscreteBayesianNetwork, DynamicBayesianNetwork)):
            raise ValueError(
                f"model should either be a Bayesian Network or Dynamic Bayesian Network. Got {type(model)}."
            )
        model.check_model()
        self.model = model

    @staticmethod
    def _get_factor_from_df(df, state_names):
        variables = list(df.index.names)

        if len(variables) == 1:
            full_index = state_names[variables[0]]
        else:
            full_index = list(
                itertools.product(*[state_names[var] for var in variables])
            )

        df = df.reindex(full_index, fill_value=0)

        cardinality = [len(state_names[var]) for var in variables]
        values = df.to_numpy().reshape(cardinality)

        return DiscreteFactor(
            variables=variables,
            cardinality=cardinality,
            values=values,
            state_names=state_names,
        )

    def get_distribution(self, samples, variables, state_names=None, joint=True):
        if isinstance(variables, (set, tuple)):
            variables = list(variables)

        if joint:
            counts = samples.groupby(variables, observed=False).size()
            probs = counts / counts.sum()
            return self._get_factor_from_df(probs, state_names)
        else:
            return {
                var: self._get_factor_from_df(
                    (lambda c: c / c.sum())(
                        samples.groupby([var], observed=False).size()
                    ),
                    state_names,
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
        # STEP 1: Generate samples ONLY if not provided
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

                max_time_slices = 0

                for var in variables:
                    max_time_slices = max(max_time_slices, var[1])

                for var in evidence:
                    max_time_slices = max(max_time_slices, var[1])

                for cpd in virtual_evidence:
                    max_time_slices = max(max_time_slices, cpd.variable[1])

                samples = self.model.simulate(
                    n_samples=n_samples,
                    n_time_slices=max_time_slices + 1,
                    evidence=evidence,
                    virtual_evidence=virtual_evidence,
                    show_progress=show_progress,
                    seed=seed,
                )

        # STEP 2: Avoid double conditioning (IMPORTANT)
        if samples is not None and evidence is not None:
            evidence = None

        # STEP 3: Get state names correctly
        state_names = {}

        for var in variables:
            if isinstance(self.model, DiscreteBayesianNetwork):
                state_names[var] = self.model.get_cpds(var).state_names[var]
            else:
                # DBN fix: use base node
                base_var = var[0]
                state_names[var] = self.model.get_cpds(base_var).state_names[base_var]

        # STEP 4: Compute distribution
        return self.get_distribution(
            samples,
            variables=variables,
            state_names=state_names,
            joint=joint,
        )

    def map_query(
        self,
        variables,
        n_samples=int(1e4),
        samples=None,
        evidence=None,
        virtual_evidence=None,
        state_names=None,
        show_progress=True,
        seed=None,
    ):
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