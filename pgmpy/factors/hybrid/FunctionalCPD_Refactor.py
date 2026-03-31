from pgmpy.factors.hybrid.Adapters import LinearGaussianAdapter, TabularAdapter
from pgmpy.factors.base import BaseFactor
from pgmpy.factors.hybrid.SkproAdapter import SkproAdapter


class FunctionalCPD(BaseFactor):
    def __init__(self, variable, tag, estimator=None):
        self.variable = variable
        self.tag = tag
        self.estimator = estimator

    def fit(self, data, target=None, parents=None):
        self.data_ = data
        self.parents_ = parents if parents is not None else []
        if target is not None:
            self.variable = target
        self.tag_name_ = self.tag[0] if isinstance(self.tag, list) else self.tag

        if self.tag_name_ == "tabular":
            self._fit_tabular()
        elif self.tag_name_ == "linear":
            self._fit_linear()
        elif self.tag_name_ == "functional":
            self._fit_functional()
        elif self.tag_name_ == "skpro" or self.tag_name_.startswith("skpro."):
            self._fit_external_ml()

        self.is_fitted_ = True

        return self

    def _fit_tabular(self):
        self.adapter_ = TabularAdapter(variable=self.variable, estimator=self.estimator, parents=self.parents_)
        self.fitted_cpd_ = self.adapter_.fit(self.data_).fitted_cpd_

    def _fit_linear(self):
        self.adapter_ = LinearGaussianAdapter(variable=self.variable, estimator=self.estimator, parents=self.parents_)
        self.fitted_cpd_ = self.adapter_.fit(self.data_).fitted_cpd_

    def _fit_external_ml(self):
        if self.estimator is None:
            raise ValueError("For skpro tag, `estimator` must be provided.")

        self.adapter_ = SkproAdapter(variable=self.variable, model=self.estimator, parents=self.parents_).fit(self.data_)
        self.fitted_cpd_ = self.adapter_

    def __repr__(self):
        if not getattr(self, "is_fitted_", False):
            tag_display = self.tag[0] if isinstance(self.tag, list) else self.tag
            return (
                f"<FunctionalCPD(variable='{self.variable}', "
                f"tag='{tag_display}', status='unfitted') at {hex(id(self))}>"
            )

        if self.tag_name_ in {"tabular", "linear"} and hasattr(self, "adapter_"):
            return repr(self.adapter_)

        return (
            f"<FunctionalCPD(variable='{self.variable}', tag='{self.tag_name_}', status='fitted') at {hex(id(self))}>"
        )
