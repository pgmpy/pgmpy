from __future__ import annotations

import io
import os
from typing import Any


class _FileIOMixin:
    """
    Adds ``save`` and ``load`` methods that dispatch to the readers and writers registered in
    ``pgmpy.readwrite.FORMATS``.

    A format is available to a model class if the format's writer lists the class (or one of its bases) in
    ``supported_models``. The first available format in ``FORMATS`` is the model's default format, which is used when
    the format can't be inferred from the file extension.
    """

    @classmethod
    def _get_reader_writer(cls, filename: str | os.PathLike | io.IOBase, filetype: str | None) -> tuple[type, type]:
        from pgmpy.readwrite import FORMATS

        supported = [name for name, (_, writer) in FORMATS.items() if issubclass(cls, writer.supported_models)]

        if filetype is None:
            filetype = supported[0]
            if isinstance(filename, (str, os.PathLike)):
                extension = os.path.splitext(os.fspath(filename))[1][1:].lower()
                for name in supported:
                    if extension in FORMATS[name][1].file_extensions:
                        filetype = name
                        break

        if filetype not in supported:
            raise ValueError(
                f"{cls.__name__} doesn't support the file format {filetype!r}. "
                f"Supported formats: {', '.join(supported)}."
            )
        return FORMATS[filetype]

    def save(self, filename: str | os.PathLike, filetype: str | None = None, **kwargs: Any) -> None:
        """
        Writes the model to a file.

        Parameters
        ----------
        filename: str or os.PathLike
            The path along with the filename where to write the file.

        filetype: str (default: None)
            The format in which to write the model. If None, the format is inferred from the file extension, falling
            back to the model's default format (bif for DiscreteBayesianNetwork, json for
            LinearGaussianBayesianNetwork). See ``pgmpy.readwrite.FORMATS`` for the available formats.

        kwargs: kwargs
            Additional arguments for the writer class of the format. Please refer to the writer class for details.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> alarm = load_model("bnlearn/alarm")
        >>> alarm.save("alarm.bif")
        >>> alarm.save("alarm.xml", filetype="xmlbif")
        """
        _, writer = self._get_reader_writer(filename, filetype)
        writer(self, **kwargs).write(filename)

    @classmethod
    def load(cls, filename: str | os.PathLike | io.IOBase, filetype: str | None = None, **kwargs: Any) -> Any:
        """
        Reads the model from a file.

        Parameters
        ----------
        filename: str, os.PathLike or file-like object
            The path along with the filename of the file to read, or a file-like object with the file contents.

        filetype: str (default: None)
            The format of the file. If None, the format is inferred from the file extension, falling back to the
            model's default format (bif for DiscreteBayesianNetwork, json for LinearGaussianBayesianNetwork). See
            ``pgmpy.readwrite.FORMATS`` for the available formats.

        kwargs: kwargs
            Additional arguments for the reader class of the format. Please refer to the reader class for details.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> alarm = load_model("bnlearn/alarm")
        >>> alarm.save("alarm.bif")
        >>> alarm_model = DiscreteBayesianNetwork.load("alarm.bif", state_name_type=str)
        """
        reader, _ = cls._get_reader_writer(filename, filetype)

        if isinstance(filename, (str, os.PathLike)):
            model = reader(path=filename, **kwargs).read()
        else:
            content = filename.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            model = reader(string=content, **kwargs).read()

        if not isinstance(model, cls):
            raise TypeError(f"Expected the file to contain a {cls.__name__}. Got a {type(model).__name__}.")
        return model
