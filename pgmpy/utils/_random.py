import functools
import inspect
from collections.abc import Callable
from typing import TypeVar

from pgmpy.utils._warnings import _warn_external

F = TypeVar("F", bound=Callable)

_SPEC7 = "pgmpy follows SPEC 7 for random number generation: https://scientific-python.org/specs/spec-0007/"


def _rename_to_rng(old_name: str) -> Callable[[F], F]:
    """Deprecate the randomness parameter ``old_name`` in favour of a keyword ``rng`` until v2.0.

    The decorated function must have a parameter named ``rng``. Passing ``old_name`` forwards it to ``rng``, and
    passing ``rng`` positionally is deprecated; both issue a ``DeprecationWarning`` at the caller.
    """

    def decorator(func: F) -> F:
        position = list(inspect.signature(func).parameters).index("rng")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if "rng" in kwargs or len(args) > position:
                    raise TypeError(f"{func.__qualname__}() got both `{old_name}` and `rng`.")
                _warn_external(
                    f"`{old_name}` is deprecated since v1.2.0 and will be removed in v2.0. Use `rng` instead. {_SPEC7}",
                    DeprecationWarning,
                )
                kwargs["rng"] = kwargs.pop(old_name)
            elif len(args) > position:
                _warn_external(
                    "Passing `rng` positionally is deprecated since v1.2.0. From v2.0 it is keyword-only; "
                    f"pass it as `rng=...`. {_SPEC7}",
                    DeprecationWarning,
                )
            return func(*args, **kwargs)

        wrapper._rng_old_name = old_name
        return wrapper

    return decorator


def _rename_param_to_rng(params: dict, old_name: str, qualname: str) -> None:
    """Move ``old_name`` to ``rng`` in a ``set_params`` keyword dict in place, with the ``_rename_to_rng`` warning.

    Raises ``TypeError`` if both ``old_name`` and ``rng`` are given.
    """
    if old_name not in params:
        return
    if "rng" in params:
        raise TypeError(f"{qualname}.set_params() got both `{old_name}` and `rng`.")
    _warn_external(
        f"`{old_name}` is deprecated since v1.2.0 and will be removed in v2.0. Use `rng` instead. {_SPEC7}",
        DeprecationWarning,
    )
    params["rng"] = params.pop(old_name)
