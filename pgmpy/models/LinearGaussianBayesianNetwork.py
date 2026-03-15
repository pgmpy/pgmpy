import io
import json
import math
import os
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.global_vars import logger


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
    P(x2 | x1) = N(0.5*x1 + -5.0; 4)
    P(x3 | x2) = N(-1*x2 + 4; 3)

    # Simulating data from the model.

    >>> df = model.simulate(n_samples=100, seed=42)
    >>> print(df.columns)
    Index(['x1', 'x2', 'x3'], dtype='object')

    # Fitting the model to the simulated data.

    >>> fitted_model = model.fit(df)

    # Predicting MAP estimates of missing variables (returns a DataFrame).

    >>> df_missing = df.drop(columns=["x3"])
    >>> predicted = fitted_model.predict(df_missing)
    >>> list(predicted.columns)
    ['x3']

    # Predicting the full posterior distribution of missing variables.

    >>> missing_vars, mu_cond, cov_cond = fitted_model.predict_probability(df_missing)
    >>> print(missing_vars)
    ['x3']
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[Tuple[Hashable, Hashable]]] = None,
        latents: Optional[Set[Hashable]] = None,
        exposures: Optional[Set[Hashable]] = None,
        outcomes: Optional[Set[Hashable]] = None,
        roles: Optional[Dict[str, Iterable]] = None,
    ) -> None:
        super(LinearGaussianBayesianNetwork, self).__init__(
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
        filename: Union[str, os.PathLike, io.IOBase],
    ) -> "LinearGaussianBayesianNetwork":
        """
        Read the model from a JSON file or a file-like object of a JSON file.

        Parameters
        ----------
        filename: str or file-like object
            The path along with the filename where to read the file, or a
            file-like object containing the model data.

        Examples
        --------
        >>> import json, io
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> data = {
        ...     "nodes": ["x1", "x2"],
        ...     "arcs": [["x1", "x2"]],
        ...     "cpds": {
        ...         "x1": {
        ...             "coefficients": {"(Intercept)": [1.0]},
        ...             "variance": [16.0],
        ...             "parents": [],
        ...         },
        ...         "x2": {
        ...             "coefficients": {"(Intercept)": [-5.0], "x1": [0.5]},
        ...             "variance": [16.0],
        ...             "parents": ["x1"],
        ...         },
        ...     },
        ... }
        >>> f = io.StringIO(json.dumps(data))
        >>> model = LinearGaussianBayesianNetwork.load(f)
        >>> sorted(model.nodes())
        ['x1', 'x2']
        """

        if isinstance(filename, (str, os.PathLike)):
            with open(filename, "r") as f:
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
        >>> import tempfile, os
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2")])
        >>> model.add_cpds(
        ...     LinearGaussianCPD("x1", [1], 4),
        ...     LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"]),
        ... )
        >>> tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        >>> tmp.close()
        >>> model.save(tmp.name)
        >>> os.path.exists(tmp.name)
        True
        >>> os.unlink(tmp.name)
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
        P(x2 | x1) = N(0.5*x1 + -5.0; 4)
        P(x3 | x2) = N(-1*x2 + 4; 3)
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

    def get_cpds(
        self, node: Optional[Hashable] = None
    ) -> Union[LinearGaussianCPD, List[LinearGaussianCPD]]:
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
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        ...
        P(x1) = N(1; 4)
        P(x2 | x1) = N(0.5*x1 + -5.0; 4)
        P(x3 | x2) = N(-1*x2 + 4; 3)
        >>> print(model.get_cpds("x1"))
        P(x1) = N(1; 4)
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
        seed: Optional[int] = None,
    ) -> Union[None, List[LinearGaussianCPD]]:
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
        >>> cpds = model.get_random_cpds(loc=0, scale=1, seed=42)
        >>> len(cpds)
        3
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

    def to_joint_gaussian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Represents the Linear Gaussian Bayesian Network as a joint Gaussian
        distribution over all variables. Returns the mean vector and covariance
        matrix of this equivalent joint Gaussian distribution.

        Returns
        -------
        mean, cov: np.ndarray, np.ndarray
            Mean vector and covariance matrix of the joint Gaussian distribution.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> mean, cov = model.to_joint_gaussian()
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
            mean[var] = (
                cpd.beta * (np.array([1] + [mean[u] for u in cpd.evidence]))
            ).sum()
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
        inv = np.linalg.inv((identity_matrix - B))
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
        ...     np.random.default_rng(42).normal(0, 1, size=(100, 3)),
        ...     columns=["x1", "x2", "x3"],
        ... )
        >>> ll = model.log_likelihood(df)
        >>> isinstance(ll, float)
        True
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
        do: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, float]] = None,
        virtual_intervention: Optional[List[LinearGaussianCPD]] = None,
        include_latents: bool = False,
        seed: Optional[int] = None,
        missing_prob=None,
    ) -> pd.DataFrame:
        """
        Simulates data from the model.

        Parameters
        ----------
        n_samples: int
            The number of samples to draw from the model.

        do: dict (default: None)
            The interventions to apply to the model. dict should be of the form
            {variable_name: value}

        evidence: dict (default: None)
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: value}

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `LinearGaussianCPD` objects specifying the virtual/soft
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

        Simple forward sampling:

        >>> df = model.simulate(n_samples=3, seed=42)
        >>> list(df.columns)
        ['x1', 'x2', 'x3']

        Sampling with intervention (do):

        >>> df_do = model.simulate(n_samples=3, seed=42, do={"x2": 0.0})
        >>> sorted(df_do.columns)
        ['x1', 'x2', 'x3']

        Sampling with evidence:

        >>> df_ev = model.simulate(n_samples=3, seed=42, evidence={"x1": 2.0})
        >>> list(df_ev.columns)
        ['x1', 'x2', 'x3']

        Sampling with both intervention and evidence:

        >>> df_both = model.simulate(
        ...     n_samples=3, seed=42, do={"x2": 1.0}, evidence={"x1": 0.0}
        ... )
        >>> sorted(df_both.columns)
        ['x1', 'x2', 'x3']
        """
        # Step 1: Check if all arguments are specified and valid
        evidence = {} if evidence is None else evidence

        do = {} if do is None else do

        virtual_intervention = (
            [] if virtual_intervention is None else virtual_intervention
        )

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
            raise ValueError(
                f"Variable(s) can't be in both do and evidence: {', '.join(common_vars)}"
            )

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
            # Use predict_probability (not predict) to get the raw posterior
            # parameters (missing_vars, mu_cond, cov_cond) needed for sampling.
            missing_vars, mean_cond, cov_cond = model.predict_probability(
                data=df_evidence
            )

            sorted_indices = np.argsort(missing_vars)
            missing_vars = [missing_vars[i] for i in sorted_indices]
            mean_cond = mean_cond[:, sorted_indices]
            cov_cond = cov_cond[sorted_indices][:, sorted_indices]

            samples_missing = rng.multivariate_normal(
                mean=mean_cond[0], cov=cov_cond, size=n_samples
            )
            df_missing = pd.DataFrame(samples_missing, columns=missing_vars)

            df = pd.DataFrame(index=range(n_samples), columns=variables)

            for ev_var, ev_val in evidence.items():
                df[ev_var] = ev_val

            for mv in missing_vars:
                df[mv] = df_missing[mv].values

            df = df[variables]

        # Step 5: Add do variables to the final dataframe
        for do_var, do_val in do.items():
            df[do_var] = do_val

        # Step 6: Remove latent variables if specified
        if not include_latents:
            df = df.drop(columns=self.latents)

        # Step 7: Handle missing_prob argument
        if missing_prob is not None:
            if not isinstance(missing_prob, dict):
                raise ValueError(
                    f"missing_prob should be dict[str, float]. Got {type(missing_prob)}"
                )

            for node, prob in missing_prob.items():
                if node not in df.columns:
                    raise ValueError(f"{node} not present in sampled data")

                if not isinstance(prob, (int, float)):
                    raise ValueError(f"Missing probability for {node} must be numeric")

                if not (0 <= prob <= 1):
                    raise ValueError(
                        f"Missing probability for {node} must be between 0 and 1"
                    )

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
                    raise ValueError(
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
                    )
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
    ) -> "LinearGaussianBayesianNetwork":
        """
        Estimates (fits) the Linear Gaussian CPDs from data.

        Parameters
        ----------
        data : pd.DataFrame
                Continuous-valued data containing all model variables.
                All variables must be continuously valued.

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
        Fitted model: LinearGaussianBayesianNetwork
            The same model object with learned CPDs added. The estimated CPDs
            can be accessed via ``model.cpds``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> df = pd.DataFrame(
        ...     np.random.default_rng(42).normal(0, 1, (100, 3)),
        ...     columns=["x1", "x2", "x3"],
        ... )
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> fitted = model.fit(df, estimator="mle", std_estimator="unbiased")
        >>> len(fitted.get_cpds())
        3
        """
        # Step 1: Check the input
        if len(missing_vars := (set(self.nodes()) - set(data.columns))) > 0:
            raise ValueError(
                f"Following variables are missing in the data: {missing_vars}"
            )

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

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the MAP (Maximum A Posteriori) estimates of the missing variables.

        For each row of the observed data, returns the conditional mean of the
        missing variables given the observed ones. For Gaussian distributions the
        MAP estimate equals the conditional mean, making this the continuous
        analogue of DiscreteBayesianNetwork.predict(), which returns the most
        probable state (MAP) per missing variable per row.

        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame with column names corresponding to a subset of variables
            in the model. Variables present in the model but absent from `data`
            are treated as missing and will be predicted.

        Returns
        -------
        predictions: pandas.DataFrame
            A DataFrame with the same index as `data` and one column per missing
            variable, containing its MAP (conditional mean) estimate.

        Raises
        ------
        ValueError
            If no variables are missing in the data (nothing to predict).
        ValueError
            If the data contains variables not present in the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> import pandas as pd
        >>> df = model.simulate(n_samples=5, seed=42)
        >>> df_obs = df.drop(columns=["x3"])
        >>> result = model.predict(df_obs)
        >>> list(result.columns)
        ['x3']
        >>> result.shape
        (5, 1)
        """
        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict.")

        if set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model.")

        # Delegate to predict_probability and return only the conditional means
        # as a DataFrame, matching DiscreteBayesianNetwork.predict()'s interface.
        missing_vars, mu_cond, _ = self.predict_probability(data)
        return pd.DataFrame(mu_cond, index=data.index, columns=missing_vars)

    def predict_probability(
        self, data: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Returns the full posterior (conditional) distribution of the missing
        variables given the observed variables.

        For a Linear Gaussian Bayesian Network the posterior is a multivariate
        Gaussian. This method returns its parameters: the per-row conditional
        mean vector, and the conditional covariance matrix (constant across all
        rows, since Gaussian covariance does not depend on the observed values).

        This is the continuous analogue of DiscreteBayesianNetwork.predict_probability(),
        which returns the full probability distribution over missing variable states.

        Parameters
        ----------
        data: pandas.DataFrame
            A DataFrame with column names corresponding to a subset of variables
            in the model. Variables present in the model but absent from `data`
            are treated as missing.

        Returns
        -------
        missing_vars: list of str
            Names of the missing variables in topological order. This ordering
            matches the columns of `mu_cond` and the axes of `cov_cond`.

        mu_cond: numpy.ndarray, shape (n_rows, n_missing)
            Conditional mean of the missing variables for each row of `data`.

        cov_cond: numpy.ndarray, shape (n_missing, n_missing)
            Conditional covariance matrix of the missing variables. Identical
            for every row of `data`.

        Raises
        ------
        ValueError
            If no variables are missing in the data (nothing to predict).
        ValueError
            If the data contains variables not present in the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> import pandas as pd
        >>> df = model.simulate(n_samples=5, seed=42)
        >>> df_obs = df.drop(columns=["x3"])
        >>> missing_vars, mu_cond, cov_cond = model.predict_probability(df_obs)
        >>> print(missing_vars)
        ['x3']
        >>> mu_cond.shape
        (5, 1)
        >>> cov_cond.shape
        (1, 1)
        """
        # Step 0: Validate inputs.
        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict.")

        if set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model.")

        # Step 1: Compute joint Gaussian parameters and establish variable ordering.
        mu, cov = self.to_joint_gaussian()
        variable_order = list(nx.topological_sort(self))

        missing_vars = [var for var in variable_order if var not in data.columns]
        observed_vars = [var for var in variable_order if var in data.columns]

        missing_indexes = [variable_order.index(var) for var in missing_vars]
        observed_indexes = [variable_order.index(var) for var in observed_vars]

        # Step 2: Partition joint mean and covariance into missing (a) and
        # observed (b) blocks.
        mu_a = mu[missing_indexes]  # shape: (n_missing,)
        mu_b = mu[observed_indexes]  # shape: (n_observed,)

        cov_aa = cov[
            np.ix_(missing_indexes, missing_indexes)
        ]  # (n_missing,  n_missing)
        cov_bb = cov[
            np.ix_(observed_indexes, observed_indexes)
        ]  # (n_observed, n_observed)
        cov_ab = cov[
            np.ix_(missing_indexes, observed_indexes)
        ]  # (n_missing,  n_observed)

        # Step 3: Apply the standard Gaussian conditioning formulae:
        #   mu_cond(row) = mu_a + Cov_ab @ Cov_bb^{-1} @ (x_b(row) - mu_b)
        #   cov_cond     = Cov_aa - Cov_ab @ Cov_bb^{-1} @ Cov_ab^T
        X_b = data.loc[:, observed_vars].values  # shape: (n_rows, n_observed)
        centered_b = X_b - np.atleast_1d(mu_b)  # shape: (n_rows, n_observed)
        mu_cond = (
            np.atleast_2d(mu_a) + (cov_ab @ np.linalg.solve(cov_bb, centered_b.T)).T
        )  # shape: (n_rows, n_missing)
        cov_cond = cov_aa - cov_ab @ np.linalg.solve(cov_bb, cov_ab.T)
        # shape: (n_missing, n_missing)

        return (missing_vars, mu_cond, cov_cond)

    def to_markov_model(self) -> None:
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD: Any) -> None:
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )

    @staticmethod
    def get_random(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: Optional[List] = None,
        latents: bool = False,
        loc: float = 0,
        scale: float = 1,
        seed: Optional[int] = None,
    ) -> "LinearGaussianBayesianNetwork":
        """
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes`
        variables with edge probability of `edge_prob` between variables.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
            If True, also creates latent variables.

        loc: float
            The mean of the normal distribution from which the coefficients are
            sampled.

        scale: float
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
        >>> model = LinearGaussianBayesianNetwork.get_random(
        ...     n_nodes=3, edge_prob=1.0, seed=42
        ... )
        >>> len(model.nodes())
        3
        >>> len(model.cpds)
        3
        """
        dag = DAG.get_random(
            n_nodes=n_nodes, edge_prob=edge_prob, node_names=node_names, latents=latents
        )
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
