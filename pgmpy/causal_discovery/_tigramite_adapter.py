class PgmpyTigramiteCITest:
    """Adapt a pgmpy conditional independence test for Tigramite."""

    def __init__(self, cond_ind_test, variable_names):
        self.cond_ind_test = cond_ind_test
        self.variable_names = list(variable_names)

    def run_test(self, X, Y, Z=None, tau_max=0, **kwargs):
        """Translate Tigramite's lagged-variable representation
        into the input expected by a pgmpy CI test.
        """
        raise NotImplementedError("Implement the lag alignment and pgmpy CI-test invocation.")
