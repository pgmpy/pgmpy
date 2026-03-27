# Contributing to pgmpy

Welcome and thank you for your interest in making pgmpy even better! This guide walks you through everything you need to know, from finding your first issue to submitting pull requests and getting feedback.

Questions at any point? Join our [Discord server](https://discord.gg/DRkdKaumBs) or drop into our weekly [community meeting](https://discord.com/events/1248540985894633492/1394200944807247922).

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. All participants in the pgmpy community are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## First-Time Contributors

If this is your first contribution to pgmpy (or to open source in general), here's how to get started:

1. Look for issues labelled [**Good First Issue**](https://github.com/pgmpy/pgmpy/labels/good%20first%20issue). These are scoped, well-defined tasks that don't require deep familiarity with the codebase.
2. **Comment on the issue** before you start working, so a maintainer can assign it to you and avoid duplicate effort.
3. **Discuss your approach** on the issue thread before opening a PR. A short description of your planned design or solution is enough.
4. Follow the setup steps below, then open your PR against the `dev` branch.

Please follow the instructions as closely as possible, but don't worry about getting everything perfect on the first try. We would be happy to help with anything that doesn't match our project standards.

## Getting Started

Before you write any code, please:

1. **Fork the repository** on GitHub: https://github.com/pgmpy/pgmpy
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:<your-username>/pgmpy.git
   cd pgmpy
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```
4. **Create and switch** to a feature branch based on `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-descriptive-name
   ```

## Installing from Source

Install pgmpy (plus testing dependencies) in editable mode:

For `bash` users:
```bash
pip install -e .[tests]
```

For `zsh` users:
```bash
pip install -e ".[tests]"
```

This lets you tweak code and immediately see your changes without re-installing.

## Running Tests

We use **pytest** for testing and GitHub Actions for Continuous Integration (CI).

To run the full test suite locally:
```
pytest -v pgmpy
```

**Tip:** Use test-driven development: Write your tests first, then implement the functionality.

## Pre-commit Hooks

We use **pre-commit** with ruff, Black, Flake8, and other checks to keep formatting consistent.

1. Install hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
2. On each commit, code will be automatically formatted and linted. If a hook modifies a file, re-stage it, and commit again.

## Reporting Issues

Use GitHub Issues to report:

- **Bugs:** Include a minimal reproducible example, the full traceback, and your environment details (Python version, pgmpy version, OS).
- **Questions:** Describe what you're trying to achieve and any roadblocks.
- **Suggestions:** Propose new features or enhancements with a short motivation.

Please fill out the issue template as completely as possible so maintainers have all the context they need.

## Proposing New Features

If you plan to add a model, algorithm, or major feature:

1. **Open an issue first**, describing:
   - The feature or algorithm you want to add.
   - Why it's useful for pgmpy users.
   - A rough implementation plan or API sketch.
2. **Wait for feedback** and approval from maintainers. This prevents duplicated effort and ensures alignment with project goals.

## Branching & Pull Requests

We follow a lightweight GitFlow on top of our `dev` branch:

1. Work in your feature branch (e.g., `feature/infer-optimization`).
2. Commit early and often. Ensure tests pass before each commit.
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-descriptive-name
   ```
4. Open a pull request against the `dev` branch via GitHub's web interface.
5. Respond to review comments and make any requested changes.

### What makes a good PR

- **Link the related issue** (e.g., "Closes #1234").
- **Write a clear title and description:** summarise what changed and why.
- **Keep it focused:** Should only contain changes required for solving the linked issue. Large PRs are harder to review and more likely to stall.
- **Include before/after examples**.
- **Fill in the PR checklist.** PRs where the checklist is removed or left empty will be closed. The checklist is included in the [PR template](https://github.com/pgmpy/pgmpy/blob/dev/.github/PULL_REQUEST_TEMPLATE.md).

## Writing Tests

Every new function or bug fix must include tests:

- **Unit tests** for individual methods and edge cases.
- **Integration tests** if your change spans multiple modules.
- Aim for **meaningful coverage** rather than 100% line coverage: focus on boundary conditions, error paths, and realistic usage.

## AI Usage Policy

We welcome thoughtful use of AI coding tools (Copilot, ChatGPT, Claude, Cursor, etc.) as development aids. That said, every contributor is fully responsible for the code they submit, regardless of how it was produced. Additionally, we expect the contributor to **fully disclose** the tool used and how it was used. 

**Guidelines:**

- **You must be able to explain and defend every line of your contribution.** If you cannot walk a reviewer through your changes, you do not understand the changes well enough, and your PR is not ready.
- **Disclose any AI usage.** If AI generated any part of your PR, fill in the LLM usage questionnaire in the PR template.
- **Test thoroughly.** AI-generated code often introduces subtle edge-case bugs. Do not rely on the AI tool to verify correctness. Think of your own edge cases for tests.
- **Follow project conventions.** Changes that ignore pgmpy's existing patterns are a common sign of unreviewed AI output. Go through your changes before asking for a review.
- **Keep conversations human.** Use AI tools to help you understand issues and feedback, but write issues, comments, and responses in your own words. Do not copy and paste AI-generated text.

Misrepresenting AI tool usage in your contributions is a **violation** of our policy and may result in a **permanent ban** from contributing to the repository.

## Documentation

Documentation is built with Sphinx. To build docs locally, follow the steps in our [Maintenance Guide — Building Docs](https://github.com/pgmpy/pgmpy/wiki/Maintenance-Guide#building-docs).

If your PR introduces or changes a user-facing API, please update the relevant docstrings and, where appropriate, add an example to the documentation.

## Seeking Help

If you get stuck, want to discuss an idea, or just want to say hello:

- **Discord:** [discord.gg/DRkdKaumBs](https://discord.gg/DRkdKaumBs) — the
  fastest way to reach maintainers and other contributors.
- **Weekly meetings:** Dev and community meetings are open to everyone. Check the
  pinned the schedule in Discord for times.

All contributions — big and small — are welcome. Let's build a better pgmpy
together!
