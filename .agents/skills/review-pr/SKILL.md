---
name: review-pr
description: Use when reviewing a pgmpy pull request, branch or diff, or when asked to check changes before a human commits them.
---

# Reviewing pgmpy changes

## Overview

Correctness of the probabilistic and causal results comes first; formatting and import order are ruff's job. Report
findings to the person who asked for the review. Don't post comments, approve or request changes on GitHub unless they
ask.

## Process

1. Read the linked issue. Flag changes outside its scope, and a PR description that drops the mandatory checklist from
   `.github/PULL_REQUEST_TEMPLATE.md`.
2. Read the diff and the code it calls, and check it against the rules below and `AGENTS.md`. Where a reference
   implementation exists, compare results. **REQUIRED SUB-SKILL:** cross-check-reference.
3. Run the smallest relevant tests, the doctests of touched modules and `pre-commit run --files <changed files>`. If a
   dependency is missing, say so.
4. Report findings ranked by severity. Each gives `file:line`, what is wrong, and a concrete input that triggers it.

## Correctness

- CPDs and factors stay normalized; variable order, cardinality and state names stay aligned when factors are reduced,
  marginalized or reordered.
- Edge cases: empty, single-node or disconnected graphs, single-state variables, evidence on every variable, pandas vs.
  numpy input.
- Code that supports both the numpy and torch backends keeps both working.
- CI tests and structure scores run per candidate inside search loops that don't catch exceptions. On degenerate input
  a CI test returns independence and a score returns `-inf`; raising is only right at single-query entry points such
  as `fit`.

## Graphs

- `PDAG`, `MAG` and `ADMG` (`_CoreGraph`, an `nx.MultiGraph`) and `DAG` (`nx.DiGraph`) differ in `has_edge`/`edges`
  semantics; check callers when code moves between them.
- What a graph class supports is a class-attribute flag (`SUPPORTED_EDGE_TYPES`, `SUPPORTS_LATENTS`) enforced once in
  the base class, at the lowest-level method every path goes through, not per-class overrides.
- ADMGs have no latent nodes: latent confounding is only bidirected edges. Latent handling belongs on the DAG → ADMG
  path, so don't ask for it on ADMG inputs.
- `ExpertKnowledge.forbidden_edges` are directional: forbidding u→v still allows v→u, and only a pair forbidden in both
  directions is non-adjacent.

## Design

- Reuse or fix the existing implementation instead of adding a parallel one. When a second caller needs a class's
  logic, extend the class's inputs rather than extracting module-level functions.
- Speed-ups use public APIs: no networkx private attributes (`_adj`) and no inline reimplementations of graph
  operations. Fix slowness in the base method instead.
- Registries map names to implementations that share one calling convention.
- Fitted results: one raw per-item attribute (e.g. a DataFrame) over several derived attributes with hidden rules.
- Tags: class-level values are the most permissive over all configurations; fit-time tag updates live in
  `BaseCausalDiscovery.fit`, not in `_fit`; multi-valued tags are lists.
- Reference `self._attr` at each use instead of assigning it to a short local alias.
- New functionality doesn't go in the legacy `pgmpy/estimators/` package.

## Don't flag

- CI tests that copy `data` in `__init__` and so can't be cloned. This is intentional.
- Formatting and import order.

The correctness and graph rules overlap with `reviews.path_instructions` in `.coderabbit.yaml`; update both together.
