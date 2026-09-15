import pgmpy.estimators.CITests as citests
from pgmpy.base import PAG
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator


class FCI(BaseConstraintEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        super().__init__(data=data, independencies=independencies, **kwargs)

    @staticmethod
    def orient_colliders(
        skeleton,
        separating_sets,
        temporal_ordering: dict | None = None,
    ):
        """
        Orient colliders in the skeleton using the separating sets and temporal ordering.

        Parameters
        ----------
        skeleton: UndirectedGraph
            The undirected graph skeleton of the BN underlying the data.
        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set of variables that makes them conditionally
            independent.
        temporal_ordering: dict, optional
            A dict containing for each node its temporal order.

        Returns
        -------
        pgmpy.base.PDAG
            The partially directed graph with v-structure orientations applied.
        """
        from pgmpy.causal_discovery import PC as _PC

        est = _PC()
        est.skeleton_ = skeleton
        est.separating_sets_ = separating_sets
        return est._orient_colliders(temporal_ordering=temporal_ordering or {})

    def estimate(
        self,
        ci_test="chi_square",
        significance_level=0.01,
        max_cond_vars=5,
        variant="stable",
        **kwargs,
    ) -> PAG | None:

        ci_test = citests.ci_registry.get_test(ci_test, data=self.data)

        # 1. Skeleton discovery
        skeleton, separating_sets = self.build_skeleton(
            variant=variant,
            ci_test=ci_test,
            significance_level=significance_level,
            max_cond_vars=max_cond_vars,
            **kwargs,
        )

        # 2. Initialize self with o–o edges
        pag = PAG()
        for u, v in skeleton.edges():
            pag.add_edge(u, v, "oo")

        # 2. Orient colliders
        pdag = self.orient_colliders(skeleton, separating_sets, temporal_ordering={})

        for u, v, edge_type in pdag.get_edges(data=True):
            if edge_type == "->":
                pag.modify_edge(u, v, "-", ">")
            elif edge_type == "<-":
                pag.modify_edge(u, v, ">", "-")
            elif edge_type == "--":
                # leave undirected edges as circle-circle in the PAG representation
                pass
        # 3. Apply orientation rules iteratively
        while True:
            pag_new = pag.apply_orientation_rules(inplace=False, separating_sets=separating_sets)
            if pag_new is None or pag_new == pag:
                return pag
            pag = pag_new

        return pag
