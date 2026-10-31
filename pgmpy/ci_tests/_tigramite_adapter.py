from ._base import BaseCITest, _CITestResult


class TigramiteAdapter(BaseCITest):
    def __init__(self, tigramite_ci, tau_max=0, cut_off="2xtau_max", alpha_or_thres=None):
        self.tigramite_ci = tigramite_ci
        self.tau_max = tau_max
        self.cut_off = cut_off
        self.alpha_or_thres = alpha_or_thres

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):

        # formatting inputs to be present standard size
        X_test = [(int(X)), 0]
        Y_test = [(int(Y)), 0]
        Z_test = [(int(z), 0) for z in Z] if Z else None

        val, pval = self.tigramite_ci.run_test(
            X=X_test, Y=Y_test, Z=Z_test, tau_max=self.tau_max, cut_off=self.cut_off, alpha_or_thres=self.alpha_or_thres
        )

        return _CITestResult(statistic=val, p_value=pval, effect_size=None, attributes=None)  # type: ignore
