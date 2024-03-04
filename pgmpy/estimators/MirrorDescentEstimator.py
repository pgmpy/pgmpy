#!/usr/bin/env python3
import numpy as np
from tqdm.auto import tqdm
from scipy.special import logsumexp

from pgmpy.estimators.base import MarginalEstimator
from pgmpy.factors import FactorDict


class MirrorDescentEstimator(MarginalEstimator):
    def _calibrate(self, theta, n):
        """
        Wrapper for JunctionTree.calibrate that handles:
            1) getting and setting clique_beliefs
            2) normalizing cliques in log-space
            3) returning marginal values in the original space

        Parameters
        ----------
        theta: FactorDict
            Mapping of clique to factors in a JunctionTree.

        n: int
            Total number of observations from a dataset.

        Returns
        -------
        mu: FactorDict
            Mapping of clique to factors representing marginal beliefs.
        """
        # Assign a new value for theta.
        self.belief_propagation.junction_tree.clique_beliefs = theta

        # TODO: Currently, belief propogation operates in the original space.
        # To be compatible with this function and for better numerical conditioning,
        # allow calibration to happen in log-space.
        self.belief_propagation.calibrate()
        mu = self.belief_propagation.junction_tree.clique_beliefs
        cliques = list(mu.keys())
        clique = cliques[0]

        # Normalize each clique (in log-space) for numerical stability
        # and then convert the marginals back to probability space so
        # they are comparable with the observed marginals.
        log_z = logsumexp(mu[clique].values)
        for clique in cliques:
            mu[clique] += np.log(n) - log_z
            np.exp(mu[clique].values, out=mu[clique].values)
        return mu

    def estimate(
        self,
        marginals,
        metric="L2",
        iterations=100,
        stepsize=None,
        show_progress=True,
    ):
        """
        Method to estimate the marginals for a given dataset.

        Parameters
        ----------
        marginals: List[Tuple[str, ...]]
            The names of the marginals to be estimated. These marginals must be present
            in the data passed to the `__init__()` method.

        metric: str
            One of either 'L1' or 'L2'.

        iterations: int
            The number of iterations to run mirror descent optimization.

        stepsize: Optional[float]
            The step size of each mirror descent gradient.
            If None, stepsize is defaulted as: `alpha = 2.0 / len(self.data) ** 2`
            and a line search is conducted each iteration.

        show_progress: bool
            Whether to show a tqdm progress bar during during optimization.

        Notes
        -------
        Estimation occurs in log-space.


        Returns
        -------
        Estimated Junction Tree: pgmpy.models.JunctionTree.JunctionTree
            Estimated Junction Tree with potentials optimized to faithfully
            represent `marginals` from a dataset.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.estimators import MirrorDescentEstimator
        >>> data = pd.DataFrame(data={"a": [0, 0, 1, 1, 1], "b": [0, 1, 0, 1, 1]})
        >>> model = FactorGraph()
        >>> model.add_nodes_from(["a", "b"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.zeros(4))
        >>> model.add_factors(phi1)
        >>> model.add_edges_from([("a", phi1), ("b", phi1)])
        >>> tree1 = MirrorDescentEstimator(model=model, data=data).estimate(marginals=[("a", "b")])
        >>> print(tree1.factors[0])
        +------+------+------------+
        | a    | b    |   phi(a,b) |
        +======+======+============+
        | a(0) | b(0) |     1.0000 |
        +------+------+------------+
        | a(0) | b(1) |     1.0000 |
        +------+------+------------+
        | a(1) | b(0) |     1.0000 |
        +------+------+------------+
        | a(1) | b(1) |     2.0000 |
        +------+------+------------+
        >>> tree2 = MirrorDescentEstimator(model=model, data=data).estimate(marginals=[("a",)])
        >>> print(tree2.factors[0])
        +------+------+------------+
        | a    | b    |   phi(a,b) |
        +======+======+============+
        | a(0) | b(0) |     1.0000 |
        +------+------+------------+
        | a(0) | b(1) |     1.0000 |
        +------+------+------------+
        | a(1) | b(0) |     1.5000 |
        +------+------+------------+
        | a(1) | b(1) |     1.5000 |
        +------+------+------------+
        """
        # Step 1: Setup variables such as data, step size, and clique to marginal mapping.
        if self.data is None:
            raise ValueError(f"No data was found to fit to the marginals {marginals}")

        n = len(self.data)

        _no_line_search = stepsize is not None
        alpha = stepsize if isinstance(stepsize, float) else 1.0 / n**2

        clique_to_marginal = self._clique_to_marginal(
            marginals=FactorDict.from_dataframe(df=self.data, marginals=marginals),
            clique_nodes=self.belief_propagation.junction_tree.nodes(),
        )

        # Step 2: Perform calibration to initialize variables.
        theta = (
            self.theta
            if self.theta
            else self.belief_propagation.junction_tree.clique_beliefs
        )
        mu = self._calibrate(theta=theta, n=n)
        answer = self._marginal_loss(
            marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
        )

        # Step 3: Optimize the potentials based off the observed marginals.
        pbar = tqdm(range(iterations)) if show_progress else range(iterations)
        for _ in pbar:
            omega, nu = theta, mu
            curr_loss, dL = answer
            if not _no_line_search:
                alpha *= 2

            if isinstance(pbar, tqdm):
                pbar.set_description_str(
                    ",\t".join(
                        [
                            "Loss: {:e}".format(curr_loss),
                            "Grad Norm: {:e}".format(np.sqrt(dL.dot(dL))),
                            "alpha: {:e}".format(alpha),
                        ]
                    )
                )

            for __ in range(25):
                # Take gradient step.
                theta = omega - alpha * dL

                # Calibrate to propogate gradients through the graph.
                mu = self._calibrate(theta=theta, n=n)

                # Compute the new loss with respect to the updated beliefs.
                answer = self._marginal_loss(
                    marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
                )
                # If we haven't appreciably improved, try reducing the step size.
                # Otherwise, we break to the next iteration.
                _step = 0.5 * alpha * dL.dot(nu - mu)
                if _no_line_search or curr_loss - answer[0] >= _step:
                    break
                alpha *= 0.5

        self.theta = theta
        self.belief_propagation.junction_tree.clique_beliefs = mu
        return self.belief_propagation.junction_tree
