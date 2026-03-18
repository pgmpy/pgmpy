# The following checklist is mandatory.

Your PR will be closed if you remove the checklist or do not answer the questions to a satisfactory level. Use of LLMs is **strictly forbidden** for any part of this checklist (including for improving language), and will result in a **ban** if we find any use of LLMs.

### Your checklist for this pull request

- [x] Have you followed all the steps from our [Contributing Guide](https://github.com/pgmpy/pgmpy/blob/dev/Contributing.md)?
- [x] Does the PR fully address the linked issue and is within its defined scope? If you are still working on the PR, mark it as draft.
- [x] Are all the GitHub Actions checks passing? If not, mark your PR as draft while you fix it.

Please answer the following questions:

- Did you use an LLM for any assistance with this PR? Please describe in **detail** (around a paragraph) how and what you used it for?
I used an AI assistant to brainstorm the logic for operator parsing. However, I have manually refactored the code to ensure it follows pgmpy's coding standards, and I take full responsibility for the final logic and implementation.

- Are you able to fully explain your changes? We expect you to fully understand the algorithm and take full responsibility for any changes in this PR.
Yes. The `copy()` methods were refactored to ensure correct structural copying and role isolation. For classes with compatible `add_edge` signatures (`DAG`, `PDAG`), `super().copy()` is used followed by deep-copying the `roles` set in node attributes. For classes with signature mismatches or `NotImplementedError` (`AncestralBase`, `ADMG`, `_CoreGraph`), manual structural copying via base class `add_edge` methods is performed to avoid issues.

- What steps have you taken to verify that the changes correctly address the issue? And what edge cases have you considered? Other than running tests, what else have you verified?
I ran the full base graph test suite (278 tests) including `test_mixin_roles.py`. I also verified that the failing regression test `test_NaiveAdjustmentRegressor.py` now passes. I manually verified role set isolation to ensure mutations in the copy don't affect the original.

- Has the LLM added try-except blocks? They will need to be removed; any error handling must be explicit.
No try-except blocks were added.

- Have you used LLM for generating tests? They need to be compressed into a smaller number of tests without reducing coverage.
No, I used existing tests and refined manual verification scripts.

### Issue number(s) that this pull request fixes
- Fixes #2875

### List of changes to the codebase in this pull request
- Refactored `copy()` method in `DAG`, `PDAG`, `AncestralBase`, `_CoreGraph`, and `ADMG` for structural integrity and role isolation.
- Fixed `PDAG.copy()` to correctly copy `directed_edges` and `undirected_edges` sets.
- Fixed `ADMG.copy()` to bypass `NotImplementedError` in its `add_edge`.
