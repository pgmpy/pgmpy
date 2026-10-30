# Governance

This document describes the governance model for the pgmpy project. It is intentionally lightweight and reflects how the project is run today. As the community grows, this document is expected to evolve.

## Project scope

pgmpy is a Python library for causal and probabilistic reasoning with graphical models. It is released under the MIT license and developed in the open at <https://github.com/pgmpy/pgmpy>. This governance document applies to the [pgmpy GitHub organization](https://github.com/pgmpy) and all repositories within it.

## Roles

### Contributors

Anyone who contributes to pgmpy in any form is a contributor. Contributions include code, documentation, tutorials and examples, bug reports, code review, issue triage, and helping other users on GitHub or the project Discord. No formal status or permission is required to contribute.

### Maintainers

Maintainers are contributors with merge rights to one or more repositories in the pgmpy organization. They are responsible for:

- Reviewing and merging pull requests in their areas of expertise.
- Triaging issues and helping new contributors get started.
- Participating in release planning and decisions.
- Upholding the [Code of Conduct](https://github.com/pgmpy/pgmpy/blob/dev/CODE_OF_CONDUCT.md).

The maintainer list below is the source of truth for who currently holds maintainer status, and is kept up to date via pull requests to this document.

The current maintainers of the project are:

- **Nimish Purohit** ([@Nimish-4](https://github.com/Nimish-4))
- **Raunak Dev** ([@DARHWOLF](https://github.com/DARHWOLF))

Emeritus Maintainers and Leads:

- **Abinash Panda** ([@abinashpanda](https://github.com/abinashpanda))
- **Utkarsh Gupta** ([@khalibartan](https://github.com/khalibartan))
- **Yashu Seth** ([@yashu-seth](https://github.com/yashu-seth))
- **Zonunmawia Zadeng** ([@Nuna7](https://github.com/Nuna7))

### Project Lead

pgmpy follows a Benevolent Dictator (BDFL) model. The Project Lead has final say on decisions where the maintainers cannot reach agreement, and holds overall responsibility for the long-term technical direction of pgmpy.

The current Project Lead is **Ankur Ankan** ([@ankurankan](https://github.com/ankurankan)).

Former Project (Co-)Leads:

- **Abinash Panda** ([@abinashpanda](https://github.com/abinashpanda))

## Decision making

pgmpy uses a consensus-seeking model with the Project Lead as the tie-breaker. In practice:

- **Routine changes**: bug fixes, documentation, small features, internal refactors may be merged by any maintainer after a normal pull-request review. They do not require broader sign-off.
- **Significant changes**: public API changes, new modules, deprecations, backward-incompatible changes, new runtime dependencies, and changes to governance or project policy are proposed via a GitHub issue or pull request and discussed openly. They require approval from at least one other maintainer and no objection from the Project Lead, with a review window of at least two weeks.
- **Disagreements**: Any decisions that cannot be resolved through discussion are decided by the Project Lead. The Project Lead is expected to use this authority sparingly and to explain the reasoning publicly.

Decisions are made by **lazy consensus** where silence after a reasonable review window is taken as agreement. Maintainers should raise objections explicitly and in writing rather than waiting to be polled. Discussion happens on GitHub (issues and pull requests) by default, so that the record is public and durable. Real-time discussion on Discord or elsewhere is encouraged, but decisions of substance must be summarized back on GitHub before they take effect.

## Becoming a maintainer

There is no fixed application process. Maintainers are added when they have demonstrated sustained, high-quality contributions and good judgment in code review and community interactions over a meaningful period of time (typically several months of regular contributions, but there is no hard threshold).

The process is:

1. An existing maintainer nominates the contributor for discussion on the maintainers channel on Discord.
2. The nomination is then formalized as a pull request to this document adding the contributor to the maintainer list.
3. The pull request is open for at least one week. With sign-off from the Project Lead and no unresolved objections from other maintainers (lazy consensus), the pull request is merged and the new maintainer is added to the relevant GitHub teams.

## Stepping down and inactivity

Maintainers may step down at any time by opening a pull request to this document moving themselves from the maintainer list to the Emeritus Maintainers section. This keeps their contribution history visible while accurately reflecting the active roster.

Maintainers who have not participated in the project for approximately 12 months will be moved to emeritus status by the Project Lead, who will open the pull request on their behalf and notify them. Their commit history and attribution remain intact, and they are welcome to return to active status at any time by request.

This policy exists to keep the active maintainer list accurate and to reduce supply-chain risk from unused commit and release credentials. It is not a judgment of past contributions.

## Changes to this document

Changes to this governance document are themselves a "significant change" under the rules above. They must be proposed by pull request, discussed openly for at least two weeks, and approved by the Project Lead. Routine updates to the maintainer list follow the dedicated processes in [Becoming a maintainer](#becoming-a-maintainer) and [Stepping down and inactivity](#stepping-down-and-inactivity) instead.

## Acknowledgements

This document draws on conventions common across small scientific Python projects, the BDFL governance template from [OSS Watch](http://oss-watch.ac.uk/resources/benevolentdictatorgovernancemodel), and governance documents from the broader open source community.
