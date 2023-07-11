import itertools
from collections import namedtuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.models import BayesianNetwork, MarkovChain, MarkovNetwork
from pgmpy.sampling import BayesianModelInference, _return_samples
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps

State = namedtuple("State", ["var", "state"])


class BayesianModelSampling(BayesianModelInference):
    """
    Class for sampling methods specific to Bayesian Models

    Parameters
    ----------
    model: instance of BayesianNetwork
        model on which inference queries will be computed
    """

    def __init__(self, model):
        super(BayesianModelSampling, self).__init__(model)

    def forward_sample(
        self,
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        partial_samples=None,
        n_jobs=-1,
    ):
        """
        Generates sample(s) from joint distribution of the Bayesian Network.

        Parameters
        ----------
        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.

        n_jobs: int (default: -1)
            The number of CPU cores to use. Default uses all cores.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = BayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...                0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...                ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> inference.forward_sample(size=2)
        rec.array([(0, 0, 1), (1, 0, 2)], dtype=
                  [('diff', '<i8'), ('intel', '<i8'), ('grade', '<i8')])
        """
        sampled = pd.DataFrame(columns=list(self.model.nodes()))

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(self.topological_order)
        else:
            pbar = self.topological_order

        if seed is not None:
            np.random.seed(seed)

        for node in pbar:
            if show_progress and config.SHOW_PROGRESS:
                pbar.set_description(f"Generating for node: {node}")
            # If values specified in partial_samples, use them. Else generate the values.
            if (partial_samples is not None) and (node in partial_samples.columns):
                sampled[node] = partial_samples.loc[:, node].values
            else:
                cpd = self.model.get_cpds(node)
                states = range(self.cardinality[node])
                evidence = cpd.variables[1:]
                if evidence:
                    evidence_values = np.vstack([sampled[i] for i in evidence])
                    unique, inverse = np.unique(
                        evidence_values.T, axis=0, return_inverse=True
                    )
                    unique = [tuple(u) for u in unique]
                    state_to_index, index_to_weight = self.pre_compute_reduce_maps(
                        variable=node, evidence=evidence, state_combinations=unique
                    )
                    if config.get_backend() == "numpy":
                        weight_index = np.array([state_to_index[u] for u in unique])[
                            inverse
                        ]
                    else:
                        weight_index = torch.Tensor(
                            [state_to_index[u] for u in unique]
                        )[inverse]
                    sampled[node] = sample_discrete_maps(
                        states, weight_index, index_to_weight, size
                    )
                else:
                    weights = cpd.values
                    sampled[node] = sample_discrete(states, weights, size)

        samples_df = _return_samples(sampled, self.state_names_map)
        if not include_latents:
            samples_df.drop(self.model.latents, axis=1, inplace=True)
        return samples_df

    def rejection_sample(
        self,
        evidence=[],
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        partial_samples=None,
    ):
        """
        Generates sample(s) from joint distribution of the Bayesian Network,
        given the evidence.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence

        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import State
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = BayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...                0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...                ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = [State(var='diff', state=0)]
        >>> inference.rejection_sample(evidence=evidence, size=2, return_type='dataframe')
                intel       diff       grade
        0         0          0          1
        1         0          0          1
        """

        if seed is not None:
            np.random.seed(seed)

        # If no evidence is given, it is equivalent to forward sampling.
        if len(evidence) == 0:
            return self.forward_sample(size=size, include_latents=include_latents)

        # Setup array to be returned
        sampled = pd.DataFrame()
        prob = 1
        i = 0

        # Do the sampling by generating samples from forward sampling and rejecting the
        # samples which do not match our evidence. Keep doing until we have enough
        # samples.
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=size)

        while i < size:
            _size = int(((size - i) / prob) * 1.5)

            # If partial_samples is specified, can only generate < partial_samples.shape[0] number of samples
            # at a time. For simplicity, just generate the same size as partial_samples.shape[0].
            if partial_samples is not None:
                _size = partial_samples.shape[0]

            _sampled = self.forward_sample(
                size=_size,
                include_latents=True,
                show_progress=False,
                partial_samples=partial_samples,
            )

            for var, state in evidence:
                _sampled = _sampled[_sampled[var] == state]

            prob = max(len(_sampled) / _size, 0.01)
            sampled = pd.concat([sampled, _sampled], axis=0, join="outer").iloc[
                :size, :
            ]
            i += _sampled.shape[0]

            if show_progress and config.SHOW_PROGRESS:
                # Update at maximum to `size`
                comp = _sampled.shape[0] if i < size else size - (i - _sampled.shape[0])
                pbar.update(comp)

        if show_progress and config.SHOW_PROGRESS:
            pbar.close()

        sampled = sampled.reset_index(drop=True)
        if not include_latents:
            sampled.drop(self.model.latents, axis=1, inplace=True)
        return sampled

    def likelihood_weighted_sample(
        self,
        evidence=[],
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        n_jobs=-1,
    ):
        """
        Generates weighted sample(s) from joint distribution of the Bayesian
        Network, that comply with the given evidence.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Algorithm 12.2 pp 493.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence

        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        n_jobs: int (default: -1)
            The number of CPU cores to use. Default uses all cores.

        Returns
        -------
        sampled: A pandas.DataFrame
            The generated samples with corresponding weights

        Examples
        --------
        >>> from pgmpy.factors.discrete import State
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = BayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
        >>> cpd_d = TabularCPD('diff', 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD('intel', 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD('grade', 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25,
        ...         0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...         ['intel', 'diff'], [2, 2])
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = [State('diff', 0)]
        >>> inference.likelihood_weighted_sample(evidence=evidence, size=2, return_type='recarray')
        rec.array([(0, 0, 1, 0.6), (0, 0, 2, 0.6)], dtype=
                  [('diff', '<i8'), ('intel', '<i8'), ('grade', '<i8'), ('_weight', '<f8')])
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert evidence state names to number
        evidence = [
            (var, self.model.get_cpds(var).get_state_no(var, state))
            for var, state in evidence
        ]

        # Prepare the return dataframe
        sampled = pd.DataFrame(columns=list(self.model.nodes()))
        sampled["_weight"] = np.ones(size)
        evidence_dict = dict(evidence)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(self.topological_order)
        else:
            pbar = self.topological_order

        # Do the sampling
        for node in pbar:
            if show_progress and config.SHOW_PROGRESS:
                pbar.set_description(f"Generating for node: {node}")

            cpd = self.model.get_cpds(node)
            states = range(self.cardinality[node])
            evidence = cpd.get_evidence()

            if evidence:
                evidence_values = np.vstack([sampled[i] for i in evidence])

                unique, inverse = np.unique(
                    evidence_values.T, axis=0, return_inverse=True
                )
                unique = [tuple(u) for u in unique]
                state_to_index, index_to_weight = self.pre_compute_reduce_maps(
                    variable=node, evidence=evidence, state_combinations=unique
                )
                weight_index = np.array([state_to_index[tuple(u)] for u in unique])[
                    inverse
                ]

                if node in evidence_dict:
                    evidence_value = evidence_dict[node]
                    sampled[node] = evidence_value
                    sampled.loc[:, "_weight"] *= np.array(
                        list(
                            map(
                                lambda i: index_to_weight[weight_index[i]][
                                    evidence_value
                                ],
                                range(size),
                            )
                        )
                    )
                else:
                    sampled[node] = sample_discrete_maps(
                        states, weight_index, index_to_weight, size
                    )
            else:
                if node in evidence_dict:
                    sampled[node] = evidence_dict[node]
                    sampled.loc[:, "_weight"] *= np.array(
                        list(
                            map(lambda _: cpd.values[evidence_dict[node]], range(size))
                        )
                    )
                else:
                    sampled[node] = sample_discrete(states, cpd.values, size)

        # Postprocess the samples: Change state numbers to names, remove latents.
        samples_df = _return_samples(sampled, self.state_names_map)
        if not include_latents:
            samples_df.drop(self.model.latents, axis=1, inplace=True)
        return samples_df


class GibbsSampling(MarkovChain):
    """
    Class for performing Gibbs sampling.

    Parameters
    ----------
    model: BayesianNetwork or MarkovNetwork
        Model from which variables are inherited and transition probabilities computed.

    Examples
    --------
    Initialization from a BayesianNetwork object:

    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.models import BayesianNetwork
    >>> intel_cpd = TabularCPD('intel', 2, [[0.7], [0.3]])
    >>> sat_cpd = TabularCPD('sat', 2, [[0.95, 0.2], [0.05, 0.8]], evidence=['intel'], evidence_card=[2])
    >>> student = BayesianNetwork()
    >>> student.add_nodes_from(['intel', 'sat'])
    >>> student.add_edge('intel', 'sat')
    >>> student.add_cpds(intel_cpd, sat_cpd)
    >>> from pgmpy.sampling import GibbsSampling
    >>> gibbs_chain = GibbsSampling(student)
    >>> gibbs_chain.sample(size=3)
       intel  sat
    0      0    0
    1      0    0
    2      1    1
    """

    def __init__(self, model=None):
        super(GibbsSampling, self).__init__()
        if isinstance(model, BayesianNetwork):
            self._get_kernel_from_bayesian_model(model)
        elif isinstance(model, MarkovNetwork):
            self._get_kernel_from_markov_model(model)

    def _get_kernel_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: BayesianNetwork
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.latents = model.latents
        self.cardinalities = {
            var: model.get_cpds(var).variable_card for var in self.variables
        }

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            factors = [cpd.to_factor() for cpd in model.cpds if var in cpd.scope()]
            factor = factor_product(*factors)
            scope = set(factor.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [State(v, s) for v, s in zip(other_vars, tup) if v in scope]
                reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = reduced_factor.values / sum(reduced_factor.values)
            self.transition_models[var] = kernel

    def _get_kernel_from_markov_model(self, model):
        """
        Computes the Gibbs transition models from a Markov Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: MarkovNetwork
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.latents = model.latents
        factors_dict = {var: [] for var in self.variables}
        for factor in model.get_factors():
            for var in factor.scope():
                factors_dict[var].append(factor)

        # Take factor product
        factors_dict = {
            var: factor_product(*factors) if len(factors) > 1 else factors[0]
            for var, factors in factors_dict.items()
        }
        self.cardinalities = {
            var: factors_dict[var].get_cardinality([var])[var] for var in self.variables
        }

        for var in self.variables:
            other_vars = [v for v in self.variables if var != v]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            factor = factors_dict[var]
            scope = set(factor.scope())
            for tup in itertools.product(*[range(card) for card in other_cards]):
                states = [
                    State(first_var, s)
                    for first_var, s in zip(other_vars, tup)
                    if first_var in scope
                ]
                reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = reduced_factor.values / sum(reduced_factor.values)
            self.transition_models[var] = kernel

    def sample(self, start_state=None, size=1, seed=None, include_latents=False):
        """
        Sample from the Markov Chain.

        Parameters
        ----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.

        size: int
            Number of samples to be generated.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import MarkovNetwork
        >>> model = MarkovNetwork([('A', 'B'), ('C', 'B')])
        >>> factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(['C', 'B'], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gibbs.sample(size=4, return_tupe='dataframe')
           A  B  C
        0  0  1  1
        1  1  0  0
        2  1  1  0
        3  1  1  1
        """
        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        if seed is not None:
            np.random.seed(seed)

        types = [(str(var_name), "int") for var_name in self.variables]
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        sampled[0] = tuple(st for var, st in self.state)
        for i in tqdm(range(size - 1)):
            for j, (var, st) in enumerate(self.state):
                other_st = tuple(st for v, st in self.state if var != v)
                next_st = sample_discrete(
                    list(range(self.cardinalities[var])),
                    self.transition_models[var][other_st],
                )[0]
                self.state[j] = State(var, next_st)
            sampled[i + 1] = tuple(st for var, st in self.state)

        samples_df = _return_samples(sampled)
        if not include_latents:
            samples_df.drop(self.latents, axis=1, inplace=True)
        return samples_df

    def generate_sample(
        self, start_state=None, size=1, include_latents=False, seed=None
    ):
        """
        Generator version of self.sample

        Returns
        -------
        List of State namedtuples, representing the assignment to all variables of the model.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import MarkovNetwork
        >>> model = MarkovNetwork([('A', 'B'), ('C', 'B')])
        >>> factor_ab = DiscreteFactor(['A', 'B'], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(['C', 'B'], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gen = gibbs.generate_sample(size=2)
        >>> [sample for sample in gen]
        [[State(var='C', state=1), State(var='B', state=1), State(var='A', state=0)],
         [State(var='C', state=0), State(var='B', state=1), State(var='A', state=1)]]
        """
        if seed is not None:
            np.random.seed(seed)

        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        for i in range(size):
            for j, (var, st) in enumerate(self.state):
                other_st = tuple(st for v, st in self.state if var != v)
                next_st = sample_discrete(
                    list(range(self.cardinalities[var])),
                    self.transition_models[var][other_st],
                )[0]
                self.state[j] = State(var, next_st)
            if include_latents:
                yield self.state[:]
            else:
                yield [s for s in self.state if i not in self.latents]
