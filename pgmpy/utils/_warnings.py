import sys
import warnings
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
# Leave tests outside the skipped prefixes so warnings identify the test caller.
_SKIP_FILE_PREFIXES = tuple(str(path) for path in _PACKAGE_DIR.iterdir() if path.name != "tests")


def _warn_external(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue a warning at the first external caller, treating pgmpy's tests as external."""
    if sys.version_info >= (3, 12):
        warnings.warn(message, category, skip_file_prefixes=_SKIP_FILE_PREFIXES)
    else:
        import inspect

        test_dir = _PACKAGE_DIR / "tests"
        frame = inspect.currentframe()
        stack_level = 1
        try:
            while frame is not None:
                filename = Path(frame.f_code.co_filename).resolve()
                if not filename.is_relative_to(_PACKAGE_DIR) or filename.is_relative_to(test_dir):
                    break
                stack_level += 1
                frame = frame.f_back
        finally:
            del frame
        warnings.warn(message, category, stacklevel=max(2, stack_level))
