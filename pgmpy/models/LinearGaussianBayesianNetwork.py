from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression

from pgmpy import logger
from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD


class LinearGaussianBayesianNetwork(DAG):
    """
    Class to represent Linear Gaussian Bayesian Networks (LGBN).

    A LGBN is a graphical model that represents a set of continuous random variables and their conditional dependencies
    via a directed acyclic graph (DAG). In a LGBN, each variable is assumed to be conditionally normally distributed,
    and the conditional probability distribution (CPD) of each variable given its parents is modeled as a linear
    function of the parents' values plus Gaussian noise. This is equivalent to assumptions of a Linear Structural
    Equation Model (SEM) with Gaussian noise.

    Parameters
    ----------
    ebunch : input graph, optional
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes, default=None
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, default=set()
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, optional (default: None)
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. If None, an empty set is used.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    # Defining a Linear Gaussian Bayesian Network.

    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.factors.continuous import LinearGaussianCPD
    >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
    >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
    >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
    >>> model.add_cpds(cpd1, cpd2, cpd3)
    >>> for cpd in model.cpds:
    ...     print(cpd)
    ...
    P(x1) = N(1; 4)
    P(x2 | x1) = N(-5 + 0.5*x1; 4)
    P(x3 | x2) = N(4 + -1*x2; 3)

    # Simulating data from the model.

    >>> df = model.simulate(n_samples=100, seed=42)
    >>> print(df.columns)
    Index(['x1', 'x2', 'x3'], dtype='object')

    # Fitting the model to the simulated data.

    >>> model.fit(df)

    # Predicting missing variables.

    >>> df_missing = df.drop(columns=["x3"])
    >>> missing_vars, mu_cond, cov_cond = model.predict(df_missing)
    >>> print(missing_vars)
    ['x3']
    >>> print(mu_cond)
    [[ 0.13440001]
     [-0.39458728]
     [ 0.60606023]
     [ 0.0732233 ]
     [-0.07241039]
     [ 0.43420811]
     [ 0.23197845]
     [ 0.35382335]
     [ 0.11859155]
     [ 0.18397848]]
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
        self.cpds = []

    @classmethod
    def load(
        cls,
        filename: str | os.PathLike | io.IOBase,
    ) -> LinearGaussianBayesianNetwork:
        """
        Read the model from a JSON file or a file-like object of a JSON file.

        Parameters
        ----------
        filename: str or file-like object
            The path along with the filename where to read the file, or a
            file-like object containing the model data.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> model = LinearGaussianBayesianNetwork.load("ecoli70.json")
        >>> print(model)
        LinearGaussianBayesianNetwork with 46 nodes and 70 edges
        """

        if isinstance(filename, (str, os.PathLike)):
            with open(filename) as f:
                data = json.load(f)
        else:
            content = filename.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)

        nodes = data.get("nodes")
        edges = data.get("arcs")
        cpds_data = data.get("cpds")

        model = cls(edges)
        model.add_nodes_from(nodes)

        cpds = []
        for node, cpd_info in cpds_data.items():
            coefficients = cpd_info["coefficients"]
            var = cpd_info["variance"][0]
            parents = cpd_info["parents"]

            intercept = coefficients["(Intercept)"][0]
            parent_coeffs = [coefficients[parent][0] for parent in parents]

            cpd = LinearGaussianCPD(
                variable=node,
                beta=[intercept] + parent_coeffs,
                std=math.sqrt(var),
                evidence=parents,
            )
            cpds.append(cpd)

        model.add_cpds(*cpds)
        return model

    def save(self, filename: str) -> None:
        """
        Writes the model to a JSON file.

        Parameters
        ----------
        filename: str
            The path along with the filename where to write the file.

        Examples
        --------
        >>> from pgmpy.datasets import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> model.save("ecoli70.json")
        """

        model_data = {
            "nodes": list(self.nodes()),
            "arcs": list(self.edges()),
            "cpds": {},
        }

        for cpd in self.get_cpds():
            coeffs_dict = {"(Intercept)": [float(cpd.beta[0])]}
            for idx, parent in enumerate(cpd.evidence):
                coeffs_dict[parent] = [float(cpd.beta[idx + 1])]

            cpd_data = {
                "coefficients": coeffs_dict,
                "variance": [float(cpd.std**2)],
                "parents": list(cpd.evidence),
            }
            model_data["cpds"][cpd.variable] = cpd_data

        with open(filename, "w") as f:
            json.dump(model_data, f, indent=4)

    def add_cpds(self, *cpds: LinearGaussianCPD) -> None:
        """
        Add Linear Gaussian CPDs (Conditional Probability Distributions)
        to the Bayesian Network.

        Parameters
        ----------
        cpds : instances of LinearGaussianCPD
            LinearGaussianCPDs which will be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.cpds:
        ...     print(cpd)
        ...
        P(x1) = N(1; 4)
        P(x2 | x1) = N(-5 + 0.5*x1; 4)
        P(x3 | x2) = N(4 + -1*x2; 3)
        """
        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD):
                raise ValueError("Only LinearGaussianCPD can be added.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logger.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node: Hashable | None = None) -> LinearGaussianCPD | list[LinearGaussianCPD]:
        """
        Returns the CPD of the specified node. If node is not specified, returns all CPDs
        that have been added so far to the graph.

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        list[LinearGaussianCPD] or LinearGaussianCPD
            A CPD or list of Linear Gaussian CPDs.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        [P(x1) = N(1; 4), P(x2 | x1) = N(-5 + 0.5*x1; 4), P(x3 | x2) = N(4 + -1*x2; 3)]
        """
        if node is not None:
            if node not in self.nodes():
                raise ValueError("Node not present in the Directed Graph")
            else:
                for cpd in self.cpds:
                    if cpd.variable == node:
                        return cpd
        else:
            return self.cpds

    def remove_cpds(self, *cpds: LinearGaussianCPD) -> None:
        """
        Removes the CPDs provided in the arguments.

        Parameters
        ----------
        *cpds: LinearGaussianCPD
            LinearGaussianCPD objects (or their variable names) to remove.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        ...
        P(x1) = N(1; 4)
        P(x2 | x1) = N(-5 + 0.5*x1; 4)
        P(x3 | x2) = N(4 + -1*x2; 3)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        ...
        P(x1) = N(1; 4)


        """
        for cpd in cpds:
            if isinstance(cpd, (str, int)):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def get_random_cpds(
        self,
        loc: float = 0,
        scale: float = 1,
        inplace: bool = False,
        seed: int | None = None,
    ) -> None | list[LinearGaussianCPD]:
        """
        Generates random Linear Gaussian CPDs for the model. The coefficients
        are sampled from a normal distribution with mean `loc` and standard
        deviation `scale`.

        Parameters
        ----------
        loc: float
            Mean of the normal from which coefficients are sampled.
        scale: float
            Std dev of the normal from which coefficients are sampled.
        inplace: bool (default: False)
            If True, adds the generated LinearGaussianCPDs to the model;
            otherwise returns them.
        seed: int (optional)
            Seed for the random number generator.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> model.get_random_cpds(loc=0, scale=1, seed=42)
        """
        # We want a different seed for each CPD; increment an integer seed in the loop.
        # We want to provide a different seed for each cpd, therefore we force it to be integer and increment in a loop.
        seed = seed if seed else 42

        cpds = []
        for i, var in enumerate(self.nodes()):
            parents = self.get_parents(var)
            cpds.append(
                LinearGaussianCPD.get_random(
                    variable=var,
                    evidence=parents,
                    loc=loc,
                    scale=scale,
                    seed=(seed + i),
                )
            )
        if inplace:
            self.add_cpds(*cpds)
        else:
            return cpds

    def to_joint_gaussian(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Represents the Linear Gaussian Bayesian Network as a joint
        Linear Gaussian Bayesian Networks can be represented using a joint
        Gaussian distribution over all the variables. This method gives
        the mean and covariance of this equivalent joint gaussian distribution.
        Returns
        -------
        mean, cov: np.ndarray, np.ndarray
            Mean vector and covariance matrix of the joint Gaussian.
            The mean and the covariance matrix of the joint gaussian distribution.
        Examples
        --------
        >>> mean, cov = model.to_joint_gaussian()
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> mean
        array([ 1. , -4.5,  8.5])
        >>> cov
        array([[ 16.,   8.,  -8.],
               [  8.,  20., -20.],
               [ -8., -20.,  29.]])
        """
        variables = list(nx.topological_sort(self))
        var_to_index = {var: i for i, var in enumerate(variables)}
        n_nodes = len(self.nodes())

        # Step 1: Compute the mean for each variable.
        mean = {}
        for var in variables:
            cpd = self.get_cpds(node=var)
            mean[var] = (cpd.beta * (np.array([1] + [mean[u] for u in cpd.evidence]))).sum()
        mean = np.array([mean[u] for u in variables])

        # Step 2: Populate the adjacency matrix, and variance matrix
        B = np.zeros((n_nodes, n_nodes))
        omega = np.zeros((n_nodes, n_nodes))
        for var in variables:
            cpd = self.get_cpds(node=var)
            for i, evidence_var in enumerate(cpd.evidence):
                B[var_to_index[evidence_var], var_to_index[var]] = cpd.beta[i + 1]
            omega[var_to_index[var], var_to_index[var]] = (cpd.std) ** 2

        # Step 3: Compute the implied covariance matrix
        identity_matrix = np.eye(n_nodes)
        inv = np.linalg.inv(identity_matrix - B)
        implied_cov = inv.T @ omega @ inv

        # Round because numerical errors can lead to non-symmetric cov matrix.
        return mean.round(decimals=8), implied_cov.round(decimals=8)

    def log_likelihood(self, data: pd.DataFrame) -> float:
        """
        Computes the log-likelihood of the given dataset under the current
        Linear Gaussian Bayesian Network.

        Parameters
        ----------
        data : pandas.DataFrame
            Observations for all variables (columns must match model variables).

        Returns
        -------
        float
            Total log-likelihood of the data under the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> df = pd.DataFrame(
        ...     np.random.normal(0, 1, size=(100, 3)), columns=["x1", "x2", "x3"]
        ... )
        >>> model.log_likelihood(df)
        -1128.66
        """
        ordering = list(nx.topological_sort(self))
        missing = set(ordering) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")
        data = data[ordering].values
        mean, cov = self.to_joint_gaussian()
        return np.sum(multivariate_normal.logpdf(data, mean=mean, cov=cov))

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        Model's copy: pgmpy.models.LinearGaussianBayesianNetwork
            Copy of the model on which the method was called.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("A", "B"), ("B", "C")])
        >>> cpd_a = LinearGaussianCPD(variable="A", beta=[1], std=4)
        >>> cpd_b = LinearGaussianCPD(
        ...     variable="B", beta=[-5, 0.5], std=4, evidence=["A"]
        ... )
        >>> cpd_c = LinearGaussianCPD(variable="C", beta=[4, -1], std=3, evidence=["B"])
        >>> model.add_cpds(cpd_a, cpd_b, cpd_c)
        >>> copy_model = model.copy()
        >>> copy_model.nodes()
        NodeView(('A', 'B', 'C'))
        >>> copy_model.edges()
        OutEdgeView([('A', 'B'), ('B', 'C')])
        >>> len(copy_model.get_cpds())
        3
        """
        model_copy = LinearGaussianBayesianNetwork()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def simulate(
        self,
        n_samples: int = 1000,
        do: dict[str, float] | None = None,
        evidence: dict[str, float] | None = None,
        virtual_intervention: list[LinearGaussianCPD] | None = None,
        include_latents: bool = False,
        seed: int | None = None,
        missing_prob=None,
    ) -> pd.DataFrame:
        """
        Simulates data from the model.

        Parameters
        ----------
        n_samples: int
            Number of samples to draw.
            The number of samples to draw from the model.

        do: dict (default: None)
            The interventions to apply to the model. dict should be of the form
            {variable_name: value}

        evidence: dict (default: None)
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: value}

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.LinearGaussianCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            Seed for the random number generator.

        missing_prob: dict (default: None)
            A dictionary specifying the probability of missingness for each variable.
            Keys must be valid variable names in the model, and values must be floats
            between 0 and 1. Each sampled value is independently replaced with NaN
            with the specified probability (MCAR assumption). A ValueError is raised
            if a variable is not present in the sampled data or if the probability
            is outside the range [0, 1].

        Returns
        -------
        pandas.DataFrame: A pandas data frame with the generated samples.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        Simple forward sampling
        >>> model.simulate(n_samples=3, seed=42)

        Sampling with intervention (do)
        >>> model.simulate(n_samples=3, seed=42, do={"x2": 0.0})

        Sampling with evidence
        >>> model.simulate(n_samples=3, seed=42, evidence={"x1": 2.0})

        Sampling with both intervention and evidence
        >>> model.simulate(n_samples=3, seed=42, do={"x2": 1.0}, evidence={"x1": 0.0})

        """
        # Step 1: Check if all arguments are specified and valid
        evidence = {} if evidence is None else evidence

        do = {} if do is None else do

        virtual_intervention = [] if virtual_intervention is None else virtual_intervention

        do_nodes = list(do.keys())
        evidence_nodes = list(evidence.keys())
        rng = np.random.default_rng(seed=seed)

        invalid_nodes = set(do_nodes) - set(self.nodes())
        if not set(do_nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"The following do-nodes are not present in the model: {invalid_nodes}. "
                f"do argument contains: {do_nodes}"
            )

        invalid_nodes = set(evidence_nodes) - set(self.nodes())
        if not set(evidence_nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"The following evidence-nodes are not present in the model: {invalid_nodes}. "
                f"evidence argument contains: {evidence_nodes}"
            )

        self.check_model()
        model = self.copy()

        if common_vars := set(do.keys()) & set(evidence.keys()):
            raise ValueError(f"Variable(s) can't be in both do and evidence: {', '.join(common_vars)}")

        if virtual_intervention != []:
            for cpd in virtual_intervention:
                var = cpd.variable
                if var not in self.nodes():
                    raise ValueError(
                        f"Virtual intervention provided for variable which is not in the model: {var}"
                        f"The following nodes are present in the model: {self.nodes()}"
                    )

        # Step 2: If do is specified, modify the network structure.
        if do != {}:
            for var, val in do.items():
                # Step 2.1: Remove incoming edges to the intervened
                #  node as well as remove the CPD's of the intervened nodes.
                for parent in list(model.get_parents(var)):
                    model.remove_edge(parent, var)

                model.remove_cpds(model.get_cpds(var))

                # Step 2.2 : For each child of an intervened node, change its CPD to remove
                #  the parent (intervened node) from the evidence and update its intercept accordingly
                for child in model.get_children(var):
                    child_cpd = model.get_cpds(child)

                    new_evidence = list(child_cpd.evidence)
                    new_beta = list(child_cpd.beta)

                    parent_idx = child_cpd.evidence.index(var)
                    new_beta[0] += new_beta[parent_idx + 1] * val

                    del new_evidence[parent_idx]
                    del new_beta[parent_idx + 1]

                    new_cpd = LinearGaussianCPD(
                        variable=child_cpd.variable,
                        beta=new_beta,
                        std=child_cpd.std,
                        evidence=new_evidence,
                    )

                    model.remove_cpds(child_cpd)
                    model.add_cpds(new_cpd)

                model.remove_node(var)

        # Step 3: If virtual_interventions are specified, change the CPD's of intervened variables
        # to specified ones and remove the incoming nodes
        for cpd in virtual_intervention:
            var = cpd.variable
            old_cpd = model.get_cpds(var)
            model.remove_cpds(old_cpd)
            model.add_cpds(cpd)

            for parent in list(model.get_parents(var)):
                model.remove_edge(parent, var)

        mean, cov = model.to_joint_gaussian()
        variables = list(nx.topological_sort(model))

        # Step 4: Sample according to evidence
        if len(evidence) == 0:
            df = pd.DataFrame(
                rng.multivariate_normal(mean=mean, cov=cov, size=n_samples),
                columns=variables,
            )

        else:
            df_evidence = pd.DataFrame([evidence])
            missing_vars, mean_cond, cov_cond = model.predict_probability(data=df_evidence)

            sorted_indices = np.argsort(missing_vars)
            missing_vars = [missing_vars[i] for i in sorted_indices]
            mean_cond = mean_cond[:, sorted_indices]
            cov_cond = cov_cond[sorted_indices][:, sorted_indices]

            samples_missing = rng.multivariate_normal(mean=mean_cond[0], cov=cov_cond, size=n_samples)
            df_missing = pd.DataFrame(samples_missing, columns=missing_vars)

            df = pd.DataFrame(index=range(n_samples), columns=variables)

            for ev_var, ev_val in evidence.items():
                df[ev_var] = ev_val

            for mv in missing_vars:
                df[mv] = df_missing[mv].values

            df = df[variables]

        # Step 5: Add do variables to the final dataFrame
        for do_var, do_val in do.items():
            df[do_var] = do_val

        # Step 6: Remove latent variables if specified
        if not include_latents:
            df = df.drop(columns=self.latents)

        # Step 7: Handle missing_prob argument
        if missing_prob is not None:
            if not isinstance(missing_prob, dict):
                raise ValueError(f"missing_prob should be dict[str, float]. Got {type(missing_prob)}")

            for node, prob in missing_prob.items():
                if node not in df.columns:
                    raise ValueError(f"{node} not present in sampled data")

                if not isinstance(prob, (int, float)):
                    raise ValueError(f"Missing probability for {node} must be numeric")

                if not (0 <= prob <= 1):
                    raise ValueError(f"Missing probability for {node} must be between 0 and 1")

            # Apply masking (post-processing stage)
            for node, prob in missing_prob.items():
                mask = rng.random(len(df)) < prob
                df.loc[mask, node] = np.nan

        return df

    def check_model(self) -> bool:
        """
        Checks the model for structural/parameter consistency.

        Currently checks:
        * Each CPD's listed parents match the graph's parents.

        Returns
        -------
        bool
            True if all checks pass; raises ValueError otherwise.
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if isinstance(cpd, LinearGaussianCPD):
                if set(cpd.evidence) != set(self.get_parents(node)):
                    raise ValueError("CPD associated with %s doesn't have proper parents associated with it." % node)
        return True

    def get_cardinality(self, node: Any) -> None:
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "mle",
        std_estimator: str = "unbiased",
    ) -> LinearGaussianBayesianNetwork:
        """
        Estimates (fits) the Linear Gaussian CPDs from data.

        Parameters
        ----------
        data : pd.DataFrame
                Continuous-valued data containing all model variables.
                A pandas DataFrame with the data to which to fit the model
                structure. All variables must be continuously valued.

        estimator : str, optional (default 'mle')
                The estimator to use for mean estimation.
                 - 'mle': Maximum Likelihood Estimation via OLS.
                Currently, MLE via OLS is the only supported method for mean estimation.

        std_estimator : str, optional (default 'unbiased')
                The estimator to use for standard deviation estimation.
                Must be one of:
                    - 'mle': Maximum Likelihood Estimation. Uses ddof=0.
                    - 'unbiased': Unbiased estimation. For root nodes, uses
                    ddof=1. For non-root nodes, uses ddof = 1 + number of parents.

        Returns
        -------
        self
        None: The estimated LinearGaussianCPDs are added to the model. They can
            be accessed using `model.cpds`.
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> df = pd.DataFrame(
        ...     np.random.normal(0, 1, (100, 3)), columns=["x1", "x2", "x3"]
        ... )
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> model.fit(df, estimator="mle", std_estimator="unbiased")
        >>> model.cpds
        [<LinearGaussianCPD: P(x1) = N(0.092; 0.825) at 0x78474cbdb350,
        <LinearGaussianCPD: P(x2 | x1) = N(-0.058*x1 + -0.178; 0.983) at 0x78474be12ba0,
        <LinearGaussianCPD: P(x3 | x2) = N(-0.141*x2 + 0.049; 1.11) at 0x78474be12750]
        """
        # Step 1: Check the input
        if len(missing_vars := (set(self.nodes()) - set(data.columns))) > 0:
            raise ValueError(f"Following variables are missing in the data: {missing_vars}")

        if estimator not in {
            "mle",
        }:
            raise ValueError("estimator must be {'mle'}")
        if std_estimator not in {"mle", "unbiased"}:
            raise ValueError("std_estimator must be one of {'mle', 'unbiased'}")

        # Step 2: Estimate the LinearGaussianCPDs
        cpds = []
        for node in self.nodes():
            parents = self.get_parents(node)
            # Step 2.1: If node doesn't have any parents (i.e. root node),
            #  simply take the mean and variance.

            if len(parents) == 0:
                ddof = 0 if std_estimator == "mle" else 1
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=[data.loc[:, node].mean()],
                        std=data.loc[:, node].std(ddof=ddof),
                    )
                )
            # Step 2.2: Else, fit a linear regression model and take the coefficients and intercept.
            # Compute error variance using predicted values.

            else:
                lm = LinearRegression().fit(data.loc[:, parents], data.loc[:, node])
                residuals = data.loc[:, node] - lm.predict(data.loc[:, parents])
                p = 1 + len(parents)  # intercept + coefficients
                ddof = 0 if std_estimator == "mle" else p
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=np.append([lm.intercept_], lm.coef_),
                        std=residuals.std(ddof=ddof),
                        evidence=parents,
                    )
                )

        # Step 3: Add the estimated CPDs to the model
        self.add_cpds(*cpds)
        return self

    def predict_probability(self, data: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Predicts the conditional distribution of missing variables

        Returns the posterior mean and covariance of the missing variables
        given the observed variables in each row of data.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with a subset of model variables observed.

        Returns
        -------
        variables: list
            Missing variables (order matches returned distribution).

        mu: np.array
            Posterior mean for each row of data.

        cov: np.array
            Posterior covariance (same for all rows, depends only on structure).

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> df = model.simulate(n_samples=5)
        >>> df = df.drop(columns=["folK"], axis=1)
        >>> model.predict(df)
        (['folK'], array([[0.13440001]]), array([[0.13440001]]))
        """
        # Step 0: Check the inputs
        missing_vars = list(set(self.nodes()) - set(data.columns))

        if len(missing_vars) == 0:
            raise ValueError("No missing variables in the data")

        # Step 1: Create separate mean and cov matrices for missing and known variables.
        mu, cov = self.to_joint_gaussian()
        variable_order = list(nx.topological_sort(self))

        missing_vars = [var for var in variable_order if var in missing_vars]
        observed_vars = [var for var in variable_order if var not in missing_vars]
        missing_indexes = [variable_order.index(var) for var in missing_vars]
        observed_indexes = [variable_order.index(var) for var in observed_vars]

        mu_a = mu[missing_indexes]
        mu_b = mu[observed_indexes]

        cov_aa = cov[np.ix_(missing_indexes, missing_indexes)]  # Full |a|×|a| submatrix
        cov_bb = cov[np.ix_(observed_indexes, observed_indexes)]  # Full |b|×|b| submatrix
        cov_ab = cov[np.ix_(missing_indexes, observed_indexes)]  # Full |a|×|b| submatrix

        # Step 2: Compute the conditional distributions
        X_b = data.loc[:, observed_vars].values  # shape: (n_samples, |observed|)
        centered_b = X_b - np.atleast_1d(mu_b)  # shape: (n_samples, |observed|).
        mu_cond = np.atleast_2d(mu_a) + (cov_ab @ np.linalg.solve(cov_bb, centered_b.T)).T
        cov_cond = cov_aa - cov_ab @ np.linalg.solve(cov_bb, cov_ab.T)

        # Step 3: Return values
        return (missing_vars, mu_cond, cov_cond)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the MAP estimates (posterior mean) of missing variables.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with a subset of model variables observed.

        Returns
        -------
        predictions: pandas.DataFrame
            DataFrame with missing variables columns containing the posterior mean
            (MAP estimate) for each row of data.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> df = model.simulate(n_samples=5)
        >>> df = df.drop(columns=["folK"], axis=1)
        >>> model.predict(df)
           folK
        0  0.134
        1  0.217
        ...
        """
        missing_vars, mu_cond, _ = self.predict_probability(data)
        return pd.DataFrame(mu_cond, columns=missing_vars, index=data.index)

    def to_markov_model(self) -> None:
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.")

    def is_imap(self, JPD: Any) -> None:
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError("is_imap method has not been implemented for LinearGaussianBayesianNetwork.")

    @staticmethod
    def get_random(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: list | None = None,
        latents: bool = False,
        loc: float = 0,
        scale: float = 1,
        seed: int | None = None,
    ) -> LinearGaussianBayesianNetwork:
        """
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes`
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.
        Parameters
        ----------
        n_nodes: int
            Number of nodes.
            The number of nodes in the randomly generated DAG.

            Probability of an edge (consistent with a topological order).
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
        loc: float

            Mean of normal for coefficients.
            The mean of the normal distribution from which the coefficients are
            sampled.

            Std dev of normal for coefficients.
            The standard deviation of the normal distribution from which the
            coefficients are sampled.

        seed: int
            The seed for the random number generator.
        Returns
        -------
        LinearGaussianBayesianNetwork
            The randomly generated model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> model = LinearGaussianBayesianNetwork.get_random(n_nodes=5)
        >>> model.nodes()
        NodeView((0, 3, 1, 2, 4))
        >>> model.edges()
        OutEdgeView([(0, 3), (3, 4), (1, 3), (2, 4)])
        >>> model.cpds
        [<LinearGaussianCPD: P(0) = N(1.764; 1.613) at 0x2732f41aae0,
        <LinearGaussianCPD: P(3 | 0, 1) = N(-0.721*0 + -0.079*1 + 0.943; 0.12) at 0x2732f16db20,
        <LinearGaussianCPD: P(1) = N(-0.534; 0.208) at 0x2732f320b30,
        <LinearGaussianCPD: P(2) = N(-0.023; 0.166) at 0x2732d8d5f40,
        <LinearGaussianCPD: P(4 | 2, 3) = N(-0.24*2 + -0.907*3 + 0.625; 0.48) at 0x2737fecdaf0]
        """
        dag = DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, node_names=node_names, latents=latents)
        lgbn_model = LinearGaussianBayesianNetwork(dag.edges(), latents=dag.latents)
        lgbn_model.add_nodes_from(dag.nodes())

        cpds = lgbn_model.get_random_cpds(loc=loc, scale=scale, seed=seed)

        lgbn_model.add_cpds(*cpds)
        return lgbn_model

    def __eq__(self, other):
        """
        Checks equality of two LinearGaussianBayesianNetwork objects. Two models are equal if they have the same
        structure and the same CPDs.

        Parameters
        ----------
        other: LinearGaussianBayesianNetwork instance
            The model to compare with.

        Returns
        -------
        bool
            True if the two LinearGaussianCPD objects are equal, False otherwise.
        """
        if not isinstance(other, LinearGaussianBayesianNetwork):
            return False

        # Test for structure equality using the DAG's __eq__ method.
        super().__eq__(other)

        # Test for LinearGaussianCPD equality.
        self_cpds = {cpd.variable: cpd for cpd in self.cpds}
        other_cpds = {cpd.variable: cpd for cpd in other.cpds}

        for var in self_cpds:
            if self_cpds[var] != other_cpds[var]:
                return False

        return True
