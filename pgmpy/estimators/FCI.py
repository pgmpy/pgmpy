from pgmpy.base import PAG
from pgmpy.estimators import BaseConstraintEstimator
from pgmpy.estimators.CITests import get_callable_ci_test


class FCI(BaseConstraintEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        super().__init__(self, data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        ci_test="chi_square",
        significance_level=0.01,
        max_cond_vars=5,
        variant="stable",
        **kwargs,
    ):

        ci_test = get_callable_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

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

        # 3. Apply orientation rules iteratively
        changed = True
        while changed:
            changed = False
            pag_new = pag.apply_orientation_rules(pag, inplace=False, sepsets=None)
            if pag_new == pag:
                break

        return pag_new
