from __future__ import annotations

import os
from typing import Any

from pgmpy.utils._warnings import _warn_external


class BaseReader:
    """
    Base class for all model file readers.

    Subclasses define the class attributes ``format_name`` and ``file_extensions``, load ``path`` or ``string`` in their
    constructor, and implement ``read``. Format-specific options are constructor arguments.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Path of the file to read.

    string : str, optional
        The file contents as a string.

    Exactly one of ``path`` and ``string`` must be specified.
    """

    format_name: str
    file_extensions: list[str]

    def __init__(self, path: str | os.PathLike | None = None, string: str | None = None):
        if (path is None) == (string is None):
            raise ValueError(f"{type(self).__name__}: specify exactly one of `path` or `string`.")
        self.path = path
        self.string = string

    def read(self) -> Any:
        """Returns the model read from the file or string."""
        raise NotImplementedError

    def get_model(self, state_name_type: type | None = None) -> Any:
        """Deprecated alias of ``read``."""
        name = type(self).__name__
        _warn_external(
            f"`{name}.get_model` is deprecated since v1.2.0 and will be removed in v2.0. Use `{name}.read` instead, "
            f"and pass `state_name_type` to the `{name}` constructor.",
            FutureWarning,
        )
        if state_name_type is not None:
            self.state_name_type = state_name_type
        return self.read()


class BaseWriter:
    """
    Base class for all model file writers.

    Subclasses define the class attributes ``format_name``, ``file_extensions`` and ``supported_models``, and implement
    ``__str__``. Format-specific options are constructor arguments.

    Parameters
    ----------
    model : instance of one of ``supported_models``
        The model to write.
    """

    format_name: str
    file_extensions: list[str]
    supported_models: tuple[type, ...]

    def __init__(self, model: Any):
        if not isinstance(model, self.supported_models):
            supported = ", ".join(model_class.__name__ for model_class in self.supported_models)
            raise TypeError(
                f"{type(self).__name__} supports only instances of {supported}. Got {type(model).__name__}."
            )
        self.model = model

    def __str__(self) -> str:
        raise NotImplementedError

    def write(self, filename: str | os.PathLike) -> None:
        """
        Writes the model to a file.

        Parameters
        ----------
        filename : str or os.PathLike
            Path of the file to write.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(self))
