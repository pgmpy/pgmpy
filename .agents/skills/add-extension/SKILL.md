---
name: add-extension
description: Use when adding a new causal discovery algorithm, conditional independence (CI) test, structure score, metric, dataset or example model to pgmpy, or when changing a base-class API that the extension templates in devtools/extension_templates/ follow.
---

# Adding an extension to pgmpy

## Overview

Each extension type has a template in `devtools/extension_templates/`. The template's header lists its own steps and
TODOs and is the source of truth for the class layout. This skill covers the steps around the template that are easy
to miss.

## Quick reference

| Type | Template | Goes in | Found by | Tests | API docs |
|---|---|---|---|---|---|
| Causal discovery | `_causal_discovery.py` | `pgmpy/causal_discovery/` | import + `__all__` in `__init__.py` | `pgmpy/tests/test_causal_discovery/` | `docs/api/structure_learning.rst` |
| CI test | `_ci_tests.py` | `pgmpy/ci_tests/` | import + `__all__`; `get_ci_test` matches the `name` tag | `pgmpy/tests/test_ci_tests/` | `docs/api/ci_test.rst` |
| Structure score | `_structure_score.py` | `pgmpy/structure_score/` | import + `__all__`; `get_scoring_method` matches the `name` tag | `pgmpy/tests/test_structure_score/` | `docs/api/structure_score.rst` |
| Metric | `_metrics.py` | `pgmpy/metrics/` | import + `__all__`; `get_metric` matches the `name` tag | `pgmpy/tests/test_metrics/` | `docs/api/metrics.rst` |
| Dataset | `_dataset.py` | `pgmpy/datasets/` | discovered automatically | add to `ALL_DATASETS` in `pgmpy/tests/test_datasets/test_datasets.py` | listed by `list_datasets()` |
| Example model | `_example_model.py` | `pgmpy/example_models/<source>/` | discovered automatically | add to the model lists in `pgmpy/tests/test_example_models/test_example_models.py` | listed by `list_models()` |

## Steps

1. Copy the template, follow its header steps and resolve every TODO. Remove the template's instruction comments.
2. **Comments.** The only comments are step comments, so that reading a method's comments in order gives its
   pseudocode:
   - Put `# Step N: <what the block does>` before each block, and `# Step N.M:` before each sub-block. Numbering
     starts at 1 in every method; `# Step 0:` is for argument validation.
   - Leave a blank line before each step. A step too long for one line continues on the next comment line, indented
     to line up with the text after `Step N: `.
   - No other comments: no end-of-line comments, no notes explaining a choice (that goes in the PR description), no
     commented-out code. A method short enough to read at a glance has no comments.

   ```python
   # Step 1: Check the arguments.
   # Step 1.1: Resolve the scoring method.
   # Step 1.2: Build the starting DAG and add the required edges.
   # Step 2: Apply the best-scoring edge operation until none improves the score.
   # Step 3: Store the results.
   ```

   See `__init__` in `pgmpy/ci_tests/hsic.py` (steps 0 to 3.2) and `_evaluate` in `pgmpy/metrics/adjacency_cm.py`.
3. **Tags.** Set every tag that `pgmpy.registry.all_tags("<object_type>")` lists. A new tag needs a documented
   `_BaseTag` subclass in `pgmpy/registry/_tags.py`. Multi-valued tags are lists, not tuples, because skbase's tag
   filters treat a tuple as one value. Class-level values are the most permissive value over all configurations, so
   tag searches find the class; set values that depend on a hyperparameter with `set_tags` in `__init__`.
4. **Docstring.** numpydoc, with an Examples section that runs as a doctest and a References section citing with
   ``:footcite:t:`key` ``. Add the BibTeX entry to `docs/references.bib`.
5. **Tests.** Take expected values from a reference implementation where one exists. **REQUIRED SUB-SKILL:**
   cross-check-reference.
6. **Fixtures.** Datasets and example models download from the Hugging Face Hub, but tests run offline:
   `pgmpy/tests/conftest.py` serves files from `pgmpy/tests/test_data/example_datasets/<data_url>` and
   `pgmpy/tests/test_data/example_models/<data_url>`. Commit the fixture there.
7. **Docs.** Add the class to the API page from the table.
8. **Verify.** Run the new test file, the module's doctests (`pytest --doctest-modules pgmpy/<package>/<file>.py`) and
   `pre-commit run --files <changed files>`.

## Changing a base-class API

Update the matching template in the same change. Contributors copy the templates, so a stale template spreads the old
API.

## Common mistakes

- A dataset, example model, structure score or metric file whose name starts with `_` is not discovered.
- A missing fixture makes tests fail with `FileNotFoundError: Local test fixture not found`.
- New code under `pgmpy/estimators/`, which is a legacy layer.
- A CI test or structure score that raises on degenerate input (e.g. collinear columns). These run per candidate inside
  search loops that don't catch exceptions: a CI test returns independence and a score returns `-inf`.
- Computing fit-time tags in `_fit`. `BaseCausalDiscovery.fit` narrows the tags from the `estimator_`, `ci_test_`,
  `scoring_method_` and `pairwise_estimator_` attributes that `_fit` sets.
- Comments other than step comments, such as leftover template TODOs or end-of-line notes, or step numbers that skip
  or repeat.
