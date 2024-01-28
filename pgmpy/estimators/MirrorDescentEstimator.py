#!/usr/bin/env python3
from numbers import Number
import numpy as np
from typing import Optional, Tuple, List
from tqdm.auto import tqdm

from pgmpy.estimators.base import MarginalEstimator
from pgmpy.factors import FactorDict
from pgmpy.models import JunctionTree


class MirrorDescentEstimator(MarginalEstimator):
    def estimate(
        self,
        marginals: List[Tuple[str, ...]],
        metric: str = "L2",
        iterations: int = 100,
        stepsize: Optional[float] = None,
        show_progress: bool = True,
        min_belief: Optional[float] = None,
        max_belief: Optional[float] = None,
    ) -> JunctionTree:
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

        min_belief: Optional[float]
            An additional constraint that ensures no belief's value
            goes below `min_belief`.

        max_belief: Optional[float]
            An additional constraint that ensures no belief's value
            goes above `max_belief`.

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
        | a(0) | b(0) |     0.9998 |
        +------+------+------------+
        | a(0) | b(1) |     0.9998 |
        +------+------+------------+
        | a(1) | b(0) |     0.9998 |
        +------+------+------------+
        | a(1) | b(1) |     1.9995 |
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
        if stepsize is None:
            alpha = 1.0 / n**2

            def _step_size_fn() -> float:
                return 2.0 * alpha

        elif isinstance(stepsize, Number):
            alpha = stepsize

            def _step_size_fn() -> float:
                return alpha

        else:
            raise ValueError(
                f"stepsize must be either float or None. Found {type(stepsize)}"
            )

        clique_to_marginal = self._clique_to_marginal(
            marginals=FactorDict.from_dataframe(df=self.data, marginals=marginals),
            clique_nodes=self.belief_propagation.junction_tree.nodes(),
        )

        # Step 2: Perform calibration to initialize variables.
        theta = self.belief_propagation.junction_tree.clique_beliefs
        self.belief_propagation.calibrate()
        mu = self.belief_propagation.junction_tree.clique_beliefs
        answer = self._marginal_loss(
            marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
        )

        if answer[0] == 0:
            return self.belief_propagation.model

        # Step 3: Optimize the potentials based off the observed marginals.
        pbar = tqdm(range(iterations)) if show_progress else range(iterations)
        for i in pbar:
            omega, nu = theta, mu
            curr_loss, dL = answer
            alpha = _step_size_fn()

            if isinstance(pbar, tqdm):
                pbar.set_description_str(
                    ",\t".join(
                        [
                            "Loss: {:e}".format(curr_loss),
                            "Grad Norm: {:e}".format(np.sqrt(dL.dot(dL))),
                        ]
                    )
                )

            for __ in range(25):
                # Take gradient step.
                theta = omega - alpha * dL

                if min_belief:
                    theta.max(min_belief, inplace=True)

                if max_belief:
                    theta.min(max_belief, inplace=True)

                # Assign gradient step to the junction tree.
                self.belief_propagation.junction_tree.clique_beliefs = theta

                # Calibrate to propogate gradients through the graph.
                self.belief_propagation.calibrate()
                mu = self.belief_propagation.junction_tree.clique_beliefs

                # Compute the new loss with respect to the updated beliefs.
                answer = self._marginal_loss(
                    marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
                )
                # If we haven't appreciably improved, try reducing the step size.
                # Otherwise, we break to the next iteration.
                _line_search = 0.5 * alpha * dL.dot(nu - mu)
                if _no_line_search or curr_loss - answer[0] >= _line_search:
                    break
                alpha *= 0.5

        self.factors = theta
        return self.belief_propagation.junction_tree
