# Contributing to pgmpy

Welcome and thank you for your interest in making pgmpy even better! This guide
walks you through everything you need to know to get started, from setting up
your development environment to submitting pull requests and getting feedback.

## Getting Started
Before you write any code, please:

1. **Fork the repository** on GitHub: https://github.com/pgmpy/pgmpy
2. **Clone your fork** locally:
```
git clone git@github.com:<your-username>/pgmpy.git
cd pgmpy
```
3. **Create and switch** to a feature branch based on `dev`:
```
git checkout dev
git pull origin dev
git checkout -b feature/your-descriptive-name
```

## Installing from Source

Install pgmpy (plus testing dependencies) in editable mode:
```
pip install -e .[tests]
```
This lets you tweak code and immediately see your changes without re-installing.

## Running Tests

We use **pytest** for testing and GitHub Actions for Continuous Integration (CI).

* To **run tests** locally:
```
pytest -v pgmpy
```
* **Tip:** Use test-driven development—write your tests first, then implement functionality.

## Pre-commit Hooks

To ensure consistent formatting, we use **pre-commit** with Black, Flake8, etc.

1. Install hooks:
```
pip install pre-commit
pre-commit install
```
2. On each commit, code will be automatically formatted and linted.

## Documentation

Documentation is built with Sphinx. Please follow the steps in our Maintenance Guide to build docs locally: https://github.com/pgmpy/pgmpy/wiki/Maintenance-Guide#building-docs

## Reporting Issues

Use GitHub issues to report:

* Bugs: include a minimal reproducible example and environment details.
* Questions: describe what you’re trying to achieve and any roadblocks.
* Suggestions: propose new features or enhancements.

Try to fill out the issue template as much as possible so maintainers have all the information they need.

## Proposing New Features

If you plan to add a model, algorithm, or major feature:

1. Open an issue first, describing:
* The feature or algorithm you want to add
* Why it’s useful for pgmpy
* A rough implementation plan or API sketch

2. Wait for feedback and approval from maintainers. This prevents duplicated effort and ensures alignment with project goals.

## Branching & Pull Requests

We follow a lightweight GitFlow on top of our dev branch:
1. Work in your feature branch (e.g., feature/infer-optimization).
2. Commit early and often—ensure tests pass before each commit.
3. Push your branch to your fork:
```
git push origin feature/your-descriptive-name
```
4. Open a pull request against the `dev` branch via GitHub’s web interface.
5. Respond to review comments and make any requested changes.

## Code Style & Best Practices

* **Formatting:** Black is our standard—let pre-commit handle this.
* **Naming:** Choose clear, descriptive names (avoid single-letter variables).
* **Strings:** Use f-strings (f"{var} = {value}").
* **File I/O:** Use context managers (with open(...) as f:).
* **Remove dead code:** Turn all # TODO comments into GitHub issues.
* **Commit messages:** Write concise, informative messages (see Thoughtbot’s guide).

## Writing Tests

Every new function or bug fix must include tests:
* Unit tests for individual methods and edge cases.
* Integration tests if your change spans multiple modules.
* Aim for meaningful coverage rather than 100% lines.

## Seeking Help & Discussion
If you have questions or want to brainstorm:
* Discord channel: invite link in the repo’s README.
* Mailing list: check the README for subscription info.


All contributions—big and small—are welcome. Let’s build a better pgmpy together! Happy coding! 🚀
