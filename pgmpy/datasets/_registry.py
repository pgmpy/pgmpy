from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Type, Union


# Exceptions ---------------------------------------------------------
class DatasetRegistrationError(Exception):
    """Raised for problems during dataset registration (duplicate names, invalid attrs)."""


class DatasetNotFoundError(KeyError):
    """Raised when requested dataset name is not found in registry."""


class Registry:
    """
    Registry that maintains dataset classes and provides search utilities.

    Internal structures:
      - _by_name: mapping name -> dataset class
      - _by_tag:  mapping tag -> set of dataset classes

    Notes:
      - Dataset classes are expected to expose either `NAME` or `name` (string).
      - Dataset classes are expected to expose either `TAGS` or `tags` (iterable of strings).
      - Use the `dataset_class` decorator to register at import time.
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, Type[Any]] = {}
        self._by_tag: Dict[str, Set[Type[Any]]] = {}

    # --- internal helpers ---
    def _normalize_name(self, cls: Type[Any]) -> str:
        name = getattr(cls, "NAME", None) or getattr(cls, "name", None)
        if name is None:
            # fallback to class name
            name = cls.__name__
        if not isinstance(name, str) or not name:
            raise DatasetRegistrationError(f"Invalid dataset name on {cls!r}: {name!r}")
        return name

    def _normalize_tags(self, cls: Type[Any]) -> List[str]:
        raw = getattr(cls, "TAGS", None) or getattr(cls, "tags", None)
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw.lower()]
        try:
            return [str(t).lower() for t in raw]
        except TypeError as exc:
            raise DatasetRegistrationError(
                f"Invalid TAGS for {cls!r}: {raw!r}"
            ) from exc

    # --- public API ---
    def register(self, cls: Type[Any]) -> None:
        """
        Register a dataset class into the registry.

        Raises:
            DatasetRegistrationError if the name is already registered.
        """
        name = self._normalize_name(cls)
        if name in self._by_name:
            raise DatasetRegistrationError(
                f"Dataset name '{name}' is already registered by {self._by_name[name]!r}"
            )
        self._by_name[name] = cls

        for tag in self._normalize_tags(cls):
            self._by_tag.setdefault(tag, set()).add(cls)

    def get(self, name: str) -> Type[Any]:
        """Return the dataset class registered under `name` or raise DatasetNotFoundError."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise DatasetNotFoundError(f"Dataset '{name}' not found") from exc

    def list_all(
        self,
        tags: Optional[Union[str, Sequence[str]]] = None,
        match_all: bool = False,
        contains: Optional[str] = None,
        return_classes: bool = False,
    ) -> List[Union[str, Type[Any]]]:
        """
        List available datasets.

        Args:
            tags: single tag or iterable of tags to filter by (case-insensitive).
            match_all: if True, datasets must have ALL tags; if False, ANY tag is enough.
            contains: substring filter on dataset name (case-insensitive).
            return_classes: if True, return classes; otherwise return dataset names.

        Returns:
            List of dataset names (default) or list of dataset classes (if return_classes=True).
        """
        # build candidate set
        if tags is None:
            candidates = set(self._by_name.values())
        else:
            if isinstance(tags, str):
                qtags = [tags.lower()]
            else:
                qtags = [str(t).lower() for t in tags]

            sets = [set(self._by_tag.get(t, set())) for t in qtags]
            if match_all:
                candidates = set.intersection(*sets) if sets else set()
            else:
                candidates = set.union(*sets) if sets else set()

        # filter by 'contains'
        if contains:
            lc = contains.lower()
            candidates = {
                c for c in candidates if lc in self._normalize_name(c).lower()
            }

        # produce return values
        results = sorted(candidates, key=lambda c: self._normalize_name(c))
        if return_classes:
            return results
        return [self._normalize_name(c) for c in results]

    def iter_classes(self) -> Iterable[Type[Any]]:
        return iter(self._by_name.values())

    def clear(self) -> None:
        """Clear registry (useful for tests)."""
        self._by_name.clear()
        self._by_tag.clear()


# module-level registry instance and decorator
DATASETS = Registry()


def dataset_class(cls: Type[Any]) -> Type[Any]:
    """Decorator to register dataset classes at import time."""
    DATASETS.register(cls)
    return cls
