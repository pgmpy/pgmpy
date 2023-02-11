import functools
import weakref


def convert_args_tuple(func):
    def _convert_param_to_tuples(
        obj, variable, parents=tuple(), complete_samples_only=None, weighted=False
    ):
        parents = tuple(parents)
        return func(obj, variable, parents, complete_samples_only, weighted)

    return _convert_param_to_tuples


def weak_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper
