#!/usr/bin/env python3
from typing import Optional, Tuple, List
from tqdm.auto import tqdm

from pgmpy.estimators.base import MarginalEstimator
from pgmpy.factors import FactorDict
from pgmpy.models import JunctionTree


class MirrorDescentEstimator(MarginalEstimator):
    def estimate(
        self,
        marginals: List[Tuple[str, ...]],
        metric: str,
        iterations: int,
        alpha: Optional[float] = None,
        show_progress: bool = True,
        potential_min: Optional[float] = None,
        potential_max: Optional[float] = None,
    ) -> JunctionTree:
        # Step 1: Map each marginal to the first clique that contains it.
        if self.data is None:
            raise ValueError(f"No data was found to fit to the marginals {marginals}")

        if not alpha:
            alpha = 2.0 / len(self.data) ** 2

        clique_to_marginal = self._clique_to_marginal(
            marginals=FactorDict.from_dataframe(df=self.data, marginals=marginals),
            clique_nodes=self.belief_propagation.junction_tree.nodes(),
        )

        # Step 2: Perform one gradient update to initialize variables.
        theta = self.belief_propagation.junction_tree.clique_beliefs
        self.belief_propagation.calibrate()
        mu = self.belief_propagation.junction_tree.clique_beliefs
        loss, dL = self._marginal_loss(
            marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
        )

        if loss == 0:
            return self.belief_propagation.model

        # Step 3: Optimize the potentials based off the observed marginals.
        pbar = tqdm(range(iterations)) if show_progress else range(iterations)
        for _ in pbar:
            omega, nu = theta, mu
            curr_loss = loss

            if isinstance(pbar, tqdm):
                pbar.set_description(f"Loss: {round(loss, 2)}")

            for __ in range(25):
                # Take gradient step.
                theta = omega - alpha * dL
                
                if potential_min:
                    theta.max(potential_min, inplace=True)

                if potential_max:
                    theta.min(potential_max, inplace=True)

                # Assign gradient step to the junction tree.
                self.belief_propagation.junction_tree.clique_beliefs = theta
                self.belief_propagation.calibrate()
                mu = self.belief_propagation.junction_tree.clique_beliefs
                loss, dL = self._marginal_loss(
                    marginals=mu, clique_to_marginal=clique_to_marginal, metric=metric
                )
                # If we haven't appreciably improved, try reducing the step size.
                # Otherwise, we break to the next iteration.
                if curr_loss - loss >= 0.5 * alpha * dL.dot(nu - mu):
                    break
                alpha *= 0.5

        self.factors = theta
        return self.belief_propagation.junction_tree
