---
name: deprecate-api
description: Use when renaming, moving or removing a public pgmpy class, function, method or parameter, when replacing an old import path, or when a change would otherwise break existing user code.
---

# Deprecating a pgmpy API

## Overview

Public APIs are removed only after a deprecation period: the old path keeps working, warns, and names the replacement.
Removal is planned for v2.0. Removing an API without a shim, or changing what it does, needs explicit approval from a
maintainer.

## Quick reference

- **Function:** `_warn_external` from `pgmpy.utils._warnings`, not `warnings.warn`. It attributes the warning to the
  user's code instead of pgmpy internals.
- **Category:** `FutureWarning`. Python hides `DeprecationWarning` outside `__main__`, so users would never see it.
- **Message:** "`<old>` is deprecated since v<version> and will be removed in v2.0. Use `<new>` instead." `<version>` is
  the release that first ships the deprecation. Check the `since` versions in recent messages and ask the maintainer
  if unsure.
- **Docstring:** the first line says the API is deprecated and names the replacement.

## Patterns

| Change | Shim | Example |
|---|---|---|
| Class moved | Subclass the new class, warn in `__init__`, map legacy argument names | `pgmpy/estimators/ExpertKnowledge.py`, `pgmpy/estimators/PC.py` |
| Method renamed | Keep the old method as an alias that warns and calls the new one | `BaseReader.get_model` in `pgmpy/readwrite/_base.py` |
| Function replaced | Warn at the top of the old function | `get_example_model` in `pgmpy/utils/utils.py` |

The shim delegates to the new implementation. It never keeps its own copy of the logic.

## Steps

1. Implement the new API and move the logic there.
2. Turn the old path into a shim using the patterns above.
3. Move every internal caller, docstring example, docs page and notebook in `examples/` to the new API.
4. Test the shim: `with pytest.warns(FutureWarning, match=...)` around the old call, and check that it gives the same
   result as the new API (see `pgmpy/tests/test_readwrite/test_base.py`). Move existing tests of the old path to the
   new one.
5. Mention the deprecation in the PR description. Don't edit `CHANGELOG.md`; maintainers write it at release time.

## Common mistakes

- `warnings.warn(..., DeprecationWarning)`: hidden from users and attributed to the wrong frame.
- A shim with its own copy of the implementation, which then drifts from the new one.
- Examples, docs or notebooks left on the old API, so users copying them hit the warning.
- A message without the replacement or the removal version.
