#!/usr/bin/env python3

from collections import defaultdict
from itertools import chain

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import (
    BayesianNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    JunctionTree,
    MarkovNetwork,
)
from pgmpy.utils import compat_fns


class Inference(object):
    """
    Base class for all inference algorithms.

    Converts BayesianNetwork and MarkovNetwork to a uniform representation so that inference
    algorithms can be applied. Also, it checks if all the associated CPDs / Factors are
    consistent with the model.

    Initialize inference for a model.

    Parameters
    ----------
    model: pgmpy.models.BayesianNetwork or pgmpy.models.MarkovNetwork or pgmpy.models.NoisyOrModel
        model for which to initialize the inference object.

    Examples
    --------
    >>> from pgmpy.inference import Inference
    >>> from pgmpy.models import BayesianNetwork
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> student = BayesianNetwork([('diff', 'grade'), ('intel', 'grade')])
    >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
    >>> intel_cpd = TabularCPD('intel', 2, [[0.3], [0.7]])
    >>> grade_cpd = TabularCPD('grade', 3, [[0.1, 0.1, 0.1, 0.1],
    ...                                     [0.1, 0.1, 0.1, 0.1],
    ...                                     [0.8, 0.8, 0.8, 0.8]],
    ...                        evidence=['diff', 'intel'], evidence_card=[2, 2])
    >>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    >>> model = Inference(student)

    >>> from pgmpy.models import MarkovNetwork
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> import numpy as np
    >>> student = MarkovNetwork([('Alice', 'Bob'), ('Bob', 'Charles'),
    ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
    >>> factor_a_b = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_b_c = DiscreteFactor(['Bob', 'Charles'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_c_d = DiscreteFactor(['Charles', 'Debbie'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> factor_d_a = DiscreteFactor(['Debbie', 'Alice'], cardinality=[2, 2],
    ...                             values=np.random.rand(4))
    >>> student.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
    >>> model = Inference(student)
    """

    def __init__(self, model):
        self.model = model
        model.check_model()

        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()

    def _initialize_structures(self):
        """
        Initializes all the data structures which will
        later be used by the inference algorithms.
        """
        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()

        self.cardinality = {}
        self.factors = defaultdict(list)

        if isinstance(self.model, BayesianNetwork):
            self.state_names_map = {}
            for node in self.model.nodes():
                cpd = self.model.get_cpds(node)
                if isinstance(cpd, TabularCPD):
                    self.cardinality[node] = cpd.variable_card
                    cpd = cpd.to_factor()
                for var in cpd.scope():
                    self.factors[var].append(cpd)
                self.state_names_map.update(cpd.no_to_name)

        elif isinstance(self.model, (MarkovNetwork, FactorGraph, JunctionTree)):
            self.cardinality = self.model.get_cardinality()

            for factor in self.model.get_factors():
                for var in factor.variables:
                    self.factors[var].append(factor)

        elif isinstance(self.model, DynamicBayesianNetwork):
            self.start_bayesian_model = BayesianNetwork(self.model.get_intra_edges(0))
            self.start_bayesian_model.add_cpds(*self.model.get_cpds(time_slice=0))
            cpd_inter = [
                self.model.get_cpds(node) for node in self.model.get_interface_nodes(1)
            ]
            self.interface_nodes = self.model.get_interface_nodes(0)
            self.one_and_half_model = BayesianNetwork(
                self.model.get_inter_edges() + self.model.get_intra_edges(1)
            )
            self.one_and_half_model.add_cpds(
                *(self.model.get_cpds(time_slice=1) + cpd_inter)
            )

    def _prune_bayesian_model(self, variables, evidence):
        """
        Prunes unnecessary nodes from the model to optimize the computation.

        Parameters
        ----------
        variables: list
            The variables on which the query is done i.e. the variables whose
            values we are interested in.

        evidence: dict (default: None)
            The variables whose values we know. The values can be specified as
            {variable: state}.

        Returns
        -------
        Pruned model: pgmpy.models.BayesianNetwork
            The pruned model.

        Examples
        --------
        >>>
        >>>

        References
        ----------
        [1] Baker, M., & Boult, T. E. (2013). Pruning Bayesian networks for efficient computation. arXiv preprint arXiv:1304.1112.
        """
        evidence = {} if evidence is None else evidence
        variables = list(self.model.nodes()) if len(variables) == 0 else list(variables)

        # Step 1: Remove all the variables that are d-separated from `variables` when conditioned
        #         on `evidence`
        d_connected = self.model.active_trail_nodes(
            variables=variables, observed=list(evidence.keys()), include_latents=True
        )
        d_connected = set.union(*d_connected.values()).union(evidence.keys())
        bn = self.model.subgraph(d_connected)
        evidence = {var: state for var, state in evidence.items() if var in d_connected}

        # Step 2: Reduce the model to ancestral graph of [`variables` + `evidence`]
        bn = bn.get_ancestral_graph(list(variables) + list(evidence.keys()))

        # Step 3: Since all the CPDs are lost, add them back. Also marginalize them if some
        #         of the variables in scope aren't in the network anymore.
        cpds = []
        for var in bn.nodes():
            cpd = self.model.get_cpds(var)
            scope_diff = set(cpd.scope()) - set(bn.nodes())
            if len(scope_diff) == 0:
                cpds.append(cpd)
            else:
                cpds.append(cpd.marginalize(scope_diff, inplace=False))

        bn.cpds = cpds

        return bn, evidence

    def _check_virtual_evidence(self, virtual_evidence):
        """
        Checks the virtual evidence's format is correct. Each evidence must:
        - Be a TabularCPD instance or a DiscreteFactor on a single variable.
        - Be targeted to a single variable
        - Be defined on a variable which is in the model
        - Have the same cardinality as its corresponding variable in the model

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.
        """
        for cpd in virtual_evidence:
            if not isinstance(cpd, (TabularCPD, DiscreteFactor)):
                raise ValueError(
                    f"Virtual evidence should be an instance of TabularCPD or DiscreteFactor. Got: {type(cpd)}"
                )
            if isinstance(cpd, DiscreteFactor):
                if len(cpd.variables) > 1:
                    raise ValueError(
                        f"If cpd is an instance of DiscreteFactor, it should be defined on a single variable. Got: {cpd}"
                    )
            var = cpd.variables[0]
            if var not in self.model.nodes():
                raise ValueError(
                    "Evidence provided for variable which is not in the model"
                )
            elif len(cpd.variables) > 1:
                raise ValueError(
                    "Virtual evidence should be defined on individual variables. Maybe you are looking for soft evidence."
                )

            elif self.model.get_cardinality(var) != cpd.get_cardinality([var])[var]:
                raise ValueError(
                    "The number of states/cardinality for the evidence should be same as the number of states/cardinality of the variable in the model"
                )

    def _virtual_evidence(self, virtual_evidence):
        """
        Modifies the model to incorporate virtual evidence. For each virtual evidence
        variable a binary variable is added as the child of the evidence variable to
        the model. The state 0 probabilities of the child is the evidence.

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.

        Returns
        -------
        None

        References
        ----------
        [1] Mrad, Ali Ben, et al. "Uncertain evidence in Bayesian networks: Presentation and comparison on a simple example." International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems. Springer, Berlin, Heidelberg, 2012.
        """
        self._check_virtual_evidence(virtual_evidence)

        bn = self.model.copy()
        for cpd in virtual_evidence:
            var = cpd.variables[0]
            new_var = "__" + var
            bn.add_edge(var, new_var)
            values = compat_fns.get_compute_backend().vstack(
                (cpd.values, 1 - cpd.values)
            )
            new_cpd = TabularCPD(
                variable=new_var,
                variable_card=2,
                values=values,
                evidence=[var],
                evidence_card=[self.model.get_cardinality(var)],
                state_names={new_var: [0, 1], var: cpd.state_names[var]},
            )
            bn.add_cpds(new_cpd)

        self.__init__(bn)

    @staticmethod
    def _get_virtual_evidence_var_list(virtual_evidence):
        """
        Returns the list of variables that have a virtual evidence.

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.
        """
        return [cpd.variables[0] for cpd in virtual_evidence]
