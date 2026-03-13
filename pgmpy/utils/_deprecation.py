"""Utility functions for deprecation handling."""

__all__ = ["_handle_deprec_args"]


def _handle_deprec_args(args, kwargs, var_names, defaults, msg=None):
    """
    Handle deprecated positional arguments for the query method.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the method.
    kwargs : dict
        Keyword arguments passed to the method.
    var_names : list
        Names of the method parameters in order.
    defaults : dict
        Default values for method parameters. Keys are parameter names.
    msg : str, optional
        Deprecation warning message to be shown if deprecated args are used.

    Returns
    -------
    dict
        A dictionary with final parameter values after handling deprecated args.
    """
    if isinstance(var_names, dict):
        var_names = list(var_names.keys())
    if len(args) > len(var_names):
        raise TypeError(
            f"Expected at most {len(var_names)} arguments, got {len(args)}."
        )
    kw_from_args = {var_names[i]: args[i] for i in range(len(args))}

    if msg is not None and (len(kw_from_args) > 0 or len(kwargs) > 0):
        import warnings

        warnings.warn(msg, FutureWarning)

    var_final = defaults.copy()
    var_final.update(kwargs)
    var_final.update(kw_from_args)
    return var_final
