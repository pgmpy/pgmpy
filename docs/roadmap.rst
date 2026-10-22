.. meta::
   :description: Broad goals and release-oriented roadmap for pgmpy development and contribution areas.

Roadmap
=======

This page outlines the broad direction for pgmpy and the focus areas for
upcoming release lines. It is intended to help contributors plan work and give
users a clearer view of what is likely to improve next.

The release sections below are planning guidance rather than a hard delivery
guarantee. Scope can change based on maintainer time, community feedback, and
the complexity of ongoing work. Patch releases within a release line are
expected to focus primarily on bug fixes, documentation improvements, and
compatibility updates.

Broad Goals
-----------

Across releases, pgmpy is aiming to keep improving a few long-running goals:

- make core workflows easier to discover and easier to compose across causal
  discovery, parameter estimation, inference, causal identification and
  estimation, simulation, and evaluation;
- improve documentation so users can move smoothly between quickstarts,
  task-oriented guides, worked examples, and API reference pages;
- expand built-in examples, datasets, and example models so workflows can be
  evaluated quickly without extensive setup;
- continue broadening modeling and algorithm coverage while keeping the public
  APIs consistent and maintainable;
- improve the contributor experience with clearer extension points, stronger
  validation for docs and examples, and a more predictable public module
  layout.

Release Line v1.2
-----------------

The next release line is intended to focus on usability and workflow coherence.
The main areas of work are expected to be:

- stronger documentation and learning resources, especially clearer guide
  structure, better cross-linking, and more task-oriented examples for common
  workflows;
- better API consistency across workflows so structure learning, parameter
  estimation, inference, and causal effect workflows feel more aligned;
- improved discovery and coverage of bundled datasets, example models, and
  benchmark-friendly examples;
- quality-of-life improvements around common user journeys, error messages, and
  smaller interface inconsistencies that affect day-to-day usage.

Release Line v1.3
-----------------

The following release line is expected to build on the v1.2 cleanup work and
push further into library depth and extensibility. Likely focus areas include:

- broader modeling and algorithm coverage across probabilistic and causal
  workflows;
- stronger extension mechanisms and reusable templates for adding new methods
  or components;
- better benchmarking and evaluation support where it helps compare workflows
  or validate model quality;
- continued improvements to developer experience, including documentation,
  examples, and automated validation around contributor-facing tooling.

How To Read Release Sections
----------------------------

If an item appears under a release line, it should be interpreted as an area of
focus rather than a promise that every related issue will ship in that release.
Exact contents will depend on review bandwidth, contributor activity, and the
size of ongoing changes.

How To Contribute Against The Roadmap
-------------------------------------

If you want to work on one of these areas:

1. Start with the :doc:`Contributing Guide <started/contributing>`.
2. Check open issues and pull requests on GitHub.
3. Open an issue or discussion if you want to propose a roadmap addition or a
   larger design change.

Useful links:

- `GitHub Issues <https://github.com/pgmpy/pgmpy/issues>`_
- `GitHub Pull Requests <https://github.com/pgmpy/pgmpy/pulls>`_
- :doc:`Getting Involved <development>`
