import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.global_vars import logger


class LinearGaussianBayesianNetwork(DAG):
    """
    A linear Gaussian Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.

    """

    def __init__(self, ebunch=None, latents=set(), lavaan_str=None, dagitty_str=None):
        super(LinearGaussianBayesianNetwork, self).__init__(
            ebunch=ebunch,
            latents=latents,
        )
        self.cpds = []

    def add_cpds(self, *cpds):
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Network.

        Parameters
        ----------
        cpds  :  instances of LinearGaussianCPD
            List of LinearGaussianCPDs which will be associated with the model

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
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

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

    def get_cpds(self, node=None):
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of linear Gaussian CPDs.

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

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: LinearGaussianCPD object
            A LinearGaussianCPD object on any subset of the variables
            of the model which is to be associated with the model.

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
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

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

    def get_random_cpds(self, loc=0, scale=1, inplace=False, seed=None):
        """
        Generates random Linear Gaussian CPDs for the model. The coefficients
        are sampled from a normal distribution with mean `loc` and standard
        deviation `scale`.

        Parameters
        ----------
        loc: float
            The mean of the normal distribution from which the coefficients are
            sampled.

        scale: float
            The standard deviation of the normal distribution from which the
            coefficients are sampled.

        inplace: bool (default: False)
            If inplace=True, adds the generated LinearGaussianCPDs to `model` itself,
            else creates a copy of the model.

        seed: int
            The seed for the random number generator.
        """
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

    def to_joint_gaussian(self):
        """
        Linear Gaussian Bayesian Networks can be represented using a joint
        Gaussian distribution over all the variables. This method gives
        the mean and covariance of this equivalent joint gaussian distribution.

        Returns
        -------
        mean, cov: np.ndarray, np.ndarray
            The mean and the covariance matrix of the joint gaussian distribution.

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
        array([ 1. ], [-4.5], [ 8.5])
        >>> cov
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

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
            omega[var_to_index[var], var_to_index[var]] = cpd.std

        # Step 3: Compute the implied covariance matrix
        identity_matrix = np.eye(n_nodes)
        inv = np.linalg.inv((identity_matrix - B))
        implied_cov = inv.T @ omega @ inv

        # Round because numerical errors can lead to non-symmetric cov matrix.
        return mean.round(decimals=8), implied_cov.round(decimals=8)

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
        >>> cpd_c = LinearGaussianCPD(
        ...     variable="C", beta=[4, -1], std=3, evidence=["x2"]
        ... )
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
        n_samples=1000,
        do=None,
        evidence=None,
        virtual_intervention=None,
        include_latents=False,
        seed=None,
    ):
        """
        Simulates data from the given model.

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
            of `pgmpy.factors.discrete.LinearGaussianCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            Seed for the random number generator.

        Returns
        -------
        pandas.DataFrame: generated samples
            A pandas data frame with the generated samples.

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

                # Step 2.2 : For each children of an intervened node, change its CPD to remove
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
            missing_vars, mean_cond, cov_cond = model.predict(data=df_evidence)

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

        return df

    def check_model(self):
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

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

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(self, data, method="mle"):
        """
        Estimates the parameters of the model using the given `data`.

        Parameters
        ----------
        data: pd.DataFrame
            A pandas DataFrame with the data to which to fit the model
            structure. All variables must be continuous valued.

        Returns
        -------
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
        >>> model.fit(df)
        >>> model.cpds
        [<LinearGaussianCPD: P(x1) = N(-0.114; 0.911) at 0x7eb77d30cec0,
         <LinearGaussianCPD: P(x2 | x1) = N(0.07*x1 + -0.075; 1.172) at 0x7eb77171fb60,
         <LinearGaussianCPD: P(x3 | x2) = N(0.006*x2 + -0.1; 0.922) at 0x7eb6abbdba10]
        """
        # Step 1: Check the input
        if len(missing_vars := (set(self.nodes()) - set(data.columns))) > 0:
            raise ValueError(
                f"Following variables are missing in the data: {missing_vars}"
            )

        # Step 2: Estimate the LinearGaussianCPDs
        cpds = []
        for node in self.nodes():
            parents = self.get_parents(node)

            # Step 2.1: If node doesn't have any parents (i.e. root node),
            #           simply take the mean and variance.
            if len(parents) == 0:
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=[data.loc[:, node].mean()],
                        std=data.loc[:, node].var(),
                    )
                )

            # Step 2.2: Else, fit a linear regression model and take the coefficients and intercept.
            #           Compute error variance using predicted values.
            else:
                lm = LinearRegression().fit(data.loc[:, parents], data.loc[:, node])
                error_var = (data.loc[:, node] - lm.predict(data.loc[:, parents])).var()
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=np.append([lm.intercept_], lm.coef_),
                        std=error_var,
                        evidence=parents,
                    )
                )

        # Step 3: Add the estimated CPDs to the model
        self.add_cpds(*cpds)

        return self

    def predict(self, data, distribution="joint"):
        """
        Predicts the distribution of the missing variable (i.e. missing columns) in the given dataset.

        Parameters
        ----------
        data: pandas.DataFrame
            The dataframe with missing variable which to predict.

        Returns
        -------
        variables: list
            The list of variables on which the returned conditional distribution is defined on.

        mu: np.array
            The mean array of the conditional joint distribution over
              the missing variables corresponding to each row of data.

        cov: np.array
            The covariance of the conditional joint distribution over the missing variables.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model("ecoli70")
        >>> df = model.simulate(n_samples=5)
        >>> # Drop a column that we want to predict.
        >>> df = df.drop(columns=["folK"], axis=1, inplace=True)
        >>> model.predict(df)
        (['folK'], array([[0.38194262], [3.06014724], [1.36829103], [0.89197438], [2.98887488]]),
                   array([[0.13440001]]))
        """
        # Step 0: Check the inputs
        missing_vars = list(set(self.nodes()) - set(data.columns))

        if len(missing_vars) == 0:
            raise ValueError("No missing variables in the data")

        # Step 1: Create separate mean and cov matrices for missing and known variables.
        mu, cov = self.to_joint_gaussian()
        variable_order = list(nx.topological_sort(self))
        missing_indexes = [variable_order.index(var) for var in missing_vars]
        remain_vars = [var for var in variable_order if var not in missing_vars]

        mu_a = mu[missing_indexes]
        mu_b = np.delete(mu, missing_indexes)

        cov_aa = cov[missing_indexes, missing_indexes]
        cov_bb = np.delete(
            np.delete(cov, missing_indexes, axis=0), missing_indexes, axis=1
        )
        cov_ab = np.delete(cov[missing_indexes, :], missing_indexes, axis=1)

        # Step 2: Compute the conditional distributions
        cov_bb_inv = np.linalg.inv(cov_bb)
        mu_cond = (
            np.atleast_2d(mu_a)
            + (
                cov_ab
                @ cov_bb_inv
                @ (data.loc[:, remain_vars].values - np.atleast_2d(mu_b)).T
            ).T
        )
        cov_cond = cov_aa - cov_ab @ cov_bb_inv @ cov_ab.T

        # Step 3: Return values
        return ([variable_order[i] for i in missing_indexes], mu_cond, cov_cond)

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )

    @staticmethod
    def get_random(
        n_nodes=5,
        edge_prob=0.5,
        node_names=None,
        latents=False,
        loc=0,
        scale=1,
        seed=None,
    ):
        """
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.

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
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

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
        dag = DAG.get_random(
            n_nodes=n_nodes, edge_prob=edge_prob, node_names=node_names, latents=latents
        )
        lgbn_model = LinearGaussianBayesianNetwork(dag.edges(), latents=dag.latents)
        lgbn_model.add_nodes_from(dag.nodes())

        cpds = lgbn_model.get_random_cpds(loc=loc, scale=scale, seed=seed)

        lgbn_model.add_cpds(*cpds)
        return lgbn_model
