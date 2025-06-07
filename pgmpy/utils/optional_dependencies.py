# optional_dependencies.py
import importlib
import warnings
from types import ModuleType
from typing import List, Optional, Tuple, Union
from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, parse as parse_version
from packaging.specifiers import SpecifierSet
from importlib.metadata import version as get_version, PackageNotFoundError


# Dummy fallback module
class _DummyModule(ModuleType):
    def __init__(self, name: str, error_msg: str):
        super().__init__(name)
        self._error_msg = error_msg

    def __getattr__(self, _):
        raise ImportError(self._error_msg)


# Get package version
def _get_pkg_version(pkg_name: str) -> Optional[Version]:
    try:
        return parse_version(get_version(pkg_name))
    except PackageNotFoundError:
        return None


def _normalize_packages_groups(
    packages: Tuple[Union[str, List[str], Tuple[str, ...]]],
) -> List[List[str]]:
    groups = []
    for arg in packages:
        if isinstance(arg, str):
            groups.append([arg])
        elif isinstance(arg, (tuple, list)):
            if all(isinstance(x, str) for x in arg):
                groups.append(list(arg))
            else:
                raise TypeError(
                    "Invalid dependency requirement: nested structure beyond two levels is not supported. "
                    "Ensure each element in a group is a string."
                )
        else:
            raise TypeError(
                f"Invalid type in dependency requirement: {type(arg)}, expected str, tuple, or list"
            )
    return groups


def _check_one_requirement(req_str: str, normalize_reqs: bool) -> bool:
    try:
        req = Requirement(req_str)
    except InvalidRequirement as e:
        raise InvalidRequirement(f"Invalid requirement string: {req_str}") from e

    installed_version = _get_pkg_version(req.name)
    if installed_version is None:
        return False

    if normalize_reqs:
        new_specs = []
        for spec in req.specifier:
            v = parse_version(spec.version)
            base_ver = v.base_version
            new_spec = spec.operator + base_ver
            new_specs.append(new_spec)
        new_specifier = SpecifierSet(",".join(new_specs))
        normalized_installed = parse_version(installed_version.base_version)
        return normalized_installed in new_specifier
    else:
        return installed_version in req.specifier


def _check_soft_dependencies(
    *packages: Union[str, List[str]],
    severity: str = "error",
    obj: Optional[Union[object, str]] = None,
    msg: Optional[str] = None,
    normalize_reqs: bool = True,
) -> bool:
    try:
        groups = _normalize_packages_groups(packages)
    except TypeError as e:
        raise TypeError(f"Invalid package specification: {e}") from e

    failed_groups = []
    for group in groups:
        group_satisfied = False
        for req_str in group:
            try:
                if _check_one_requirement(req_str, normalize_reqs):
                    group_satisfied = True
                    break
            except InvalidRequirement as e:
                raise e
        if not group_satisfied:
            failed_groups.append(group)

    if not failed_groups:
        return True

    group_strs = []
    for group in failed_groups:
        if len(group) == 1:
            group_strs.append(group[0])
        else:
            group_strs.append("(" + " OR ".join(group) + ")")
    failed_str = " AND ".join(group_strs)

    if msg is None:
        if obj is not None:
            if isinstance(obj, str):
                obj_name = obj
            else:
                try:
                    obj_name = type(obj).__name__
                except AttributeError:
                    try:
                        obj_name = obj.__name__
                    except AttributeError:
                        obj_name = str(obj)
            msg = (
                f"Missing required dependencies for {obj_name}. "
                f"Required dependencies: {failed_str}"
            )
        else:
            msg = (
                f"Missing required dependencies. "
                f"Required dependencies: {failed_str}"
            )

    if severity == "error":
        raise ModuleNotFoundError(msg)
    elif severity == "warning":
        warnings.warn(msg, UserWarning)
    elif severity == "none":
        pass
    else:
        raise ValueError(f"Invalid severity: {severity}")

    return False


# Safe import
def _safe_import(module_name: str) -> Union[ModuleType, _DummyModule]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return _DummyModule(
            module_name,
            f"Missing module '{module_name}'. Install with: pip install {module_name.split('.')[0]}",
        )
