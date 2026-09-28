import sys
from typing import TypeVar

if sys.version_info >= (3, 11):  # noqa: UP036
    from typing import Self
else:
    Self = TypeVar("Self", bound="Self")
