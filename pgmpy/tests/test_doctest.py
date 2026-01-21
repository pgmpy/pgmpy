# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Doctest checks directed through pytest with conditional skipping."""

import importlib
import inspect
import pkgutil
from functools import lru_cache

EXCLUDE_MODULES_STARTING_WITH = ("all", "test")


def _all_objects(module_name, obj_type="all"):
    """Get all functions from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.
    obj_type: str, "all" (default), "functions", or "classes"
        Type of objects to retrieve.

        * "functions": retrieve functions only.
        * "classes": retrieve classes only.
        * "all": retrieve both functions and classes.

    Returns
    -------
    obj_list : list
        List of tuples (object_name, object).
    """
    res = _all_objects_cached(module_name, obj_type=obj_type)
    # copy the result to avoid modifying the cached result
    return res.copy()


@lru_cache
def _all_objects_cached(module_name, obj_type="all"):
    """Get all functions from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.
    obj_type: str, "all" (default), "functions", or "classes"
        Type of objects to retrieve.

        * "functions": retrieve functions only.
        * "classes": retrieve classes only.
        * "all": retrieve both functions and classes.

    Returns
    -------
    obj_list : list
        List of tuples (object_name, object).
    """
    # Import the package
    package = importlib.import_module(module_name)

    # Initialize an empty list to hold all retrieved objects
    obj_list = []

    # Walk through the package's modules
    package_path = package.__path__[0]
    for _, modname, _ in pkgutil.walk_packages(
        path=[package_path], prefix=package.__name__ + "."
    ):
        # Skip modules starting with 'all' or 'test'
        if modname.split(".")[-1].startswith(EXCLUDE_MODULES_STARTING_WITH):
            continue

        # Import the module
        module = importlib.import_module(modname)

        if obj_type == "functions":
            get_members_fn = inspect.isfunction
        elif obj_type == "classes":
            get_members_fn = inspect.isclass
        elif obj_type == "all":

            def get_members_fn(x):
                return inspect.isfunction(x) or inspect.isclass(x)

        else:
            raise ValueError(f"Unknown obj_type: {obj_type}")

        # Get all objects from the module
        for name, obj in inspect.getmembers(module, get_members_fn):
            # if imported from another module, skip it
            if obj.__module__ != module.__name__:
                continue
            # add the function to the list
            obj_list.append((name, obj))

    return obj_list


def pytest_generate_tests(metafunc):
    """Test parameterization routine for pytest.

    Fixtures parameterized
    ----------------------
    obj : all objects from pgmpy, as returned by _all_objects_cached
    """
    objs_and_names = _all_objects("pgmpy")

    if len(objs_and_names) > 0:
        names, objs = zip(*objs_and_names)

        metafunc.parametrize("obj", objs, ids=names)
    else:
        metafunc.parametrize("obj", [])


def test_all_doctest(obj):
    """Run doctest for all functions in pgmpy."""
    from skbase.utils.doctest_run import run_doctest

    run_doctest(obj, name=f"{obj.__name__}")
