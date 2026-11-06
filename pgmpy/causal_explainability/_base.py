import pandas as pd


class _BaseAttribution:
    """Base class for all causal attribution methods."""

    def attribute(self, model, data, target, **kwargs):
        self._validate(model, data, target)
        return self._attribute(model, data, target, **kwargs)

    def _attribute(self, model, data, target, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} must implement _attribute.")

    def _validate(self, model, data, target):
        if target is not None and target not in model.nodes():
            raise ValueError(f"Target '{target}' is not a node in the model. Model nodes: {list(model.nodes())}")
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
            missing = set(model.nodes()) - set(data.columns)
            latents = set()
            if hasattr(model, "latents"):
                latents = set(model.latents)
            missing = missing - latents
            if missing:
                raise ValueError(f"data is missing columns for model nodes: {missing}")
