import pgmpy.estimators.CITests as citests
from pgmpy.base import PAG
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator


class FCI(BaseConstraintEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        super().__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        ci_test="chi_square",
        significance_level=0.01,
        max_cond_vars=5,
        variant="stable",
        **kwargs,
    ):

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
            pag.add_edge(u, v, "o", "o")

        # 2. Orient colliders
        directed_edges, _ = self.orient_colliders(
            skeleton, separating_sets, temporal_ordering={}
        )
        directed_edges = set(directed_edges)

        for u, v in skeleton.edges():
            uv = (u, v) in directed_edges
            vu = (v, u) in directed_edges

            if uv and not vu:
                pag.modify_edge(u, v, "-", ">")
            elif vu and not uv:
                pag.modify_edge(u, v, ">", "-")
            elif uv and vu:
                pag.modify_edge(u, v, "o", "o")
        # 3. Apply orientation rules iteratively
        while True:
            pag_new = pag.apply_orientation_rules(pag, False, separating_sets)

            if pag_new == pag:
                break
            pag = pag_new

        return pag_new
