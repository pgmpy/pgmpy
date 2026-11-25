from ._base import BaseCITest, _CITestResult


class TigramiteAdapter(BaseCITest):
    """Expose a Tigramite CI test through pgmpy's CI-test interface."""

    def __init__(
        self,
        tigramite_ci,
        variable_names,
        tau_max=0,
        cut_off="2xtau_max",
        alpha_or_thres=None,
    ):
        super().__init__()

        if not callable(getattr(tigramite_ci, "run_test", None)):
            raise TypeError("tigramite_ci must expose a callable run_test method.")

        self.tigramite_ci = tigramite_ci
        self.variable_names = variable_names
        self.variable_to_index = {name: index for index, name in enumerate(variable_names)}
        self.tau_max = tau_max
        self.cut_off = cut_off
        self.alpha_or_thres = alpha_or_thres

    def _compute_result(self, X: str, Y: str, Z: list):
        X_test = [(self.variable_to_index[X], 0)]
        Y_test = [(self.variable_to_index[Y], 0)]
        Z_test = [(self.variable_to_index[z], 0) for z in Z] if Z else None

        val, pval = self.tigramite_ci.run_test(
            X=X_test,
            Y=Y_test,
            Z=Z_test,
            tau_max=self.tau_max,
            cut_off=self.cut_off,
            alpha_or_thres=self.alpha_or_thres,
        )

        return _CITestResult(
            statistic=val,
            p_value=pval,
            effect_size=None,
        )
