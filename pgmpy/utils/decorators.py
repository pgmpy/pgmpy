def convert_args_tuple(func):
    def _convert_param_to_tuples(
        obj, variable, parents=tuple(), complete_samples_only=None
    ):
        parents = tuple(parents)
        return func(obj, variable, parents, complete_samples_only)

    return _convert_param_to_tuples
