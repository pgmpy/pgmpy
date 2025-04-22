.. orphan::

Contributing to pgmpy
=====================

Welcome — and thanks for helping make **pgmpy** better!
Use the quick links below for common actions, then read the short guides that follow.

.. list-table::
   :header-rows: 0
   :widths: 33 67

   * - **📄 Report a bug**
     - https://github.com/pgmpy/pgmpy/issues/new
   * - **💡 Propose a feature**
     - https://github.com/pgmpy/pgmpy/issues/new
   * - **🛠  Open a pull request**
     - https://github.com/pgmpy/pgmpy/compare/

------------------------------------------------------------------------

1  Development workflow
-----------------------

#. **Fork** `pgmpy/pgmpy` to your GitHub account.
#. **Clone** your fork:
   ``git clone git@github.com:<your_github_username>/pgmpy.git``
#. **Create a branch** from ``dev`` with a descriptive name, e.g.

   ``git checkout -b feature/bic-score``
#. Make small, focused commits (`git add <file>` ➜ `git commit -m "feat: add BIC score"`).
#. **Push** the branch and open a **pull request** against ``dev``.
#. Address review comments; when approved, we merge.

*We follow the `Git Flow <https://nvie.com/posts/a-successful-git-branching-model/>` convention; nothing lands directly on ``dev`` without a PR.*

------------------------------------------------------------------------

2  Code style & quality
-----------------------

* We have a pre-commit hook in the repository to automatically test code style. You can install it by running ``pre-commit install`` from the pgmpy directory.

------------------------------------------------------------------------

3  Testing
----------

* Write **pytest** tests for every new feature or bug‑fix.
* Run the full suite locally: ``pytest -q``.
  The **dev** branch must stay green at all times.
* Continuous integration runs on **GitHub Actions**; your PR must pass before merge.

> **Tip:** try *test‑driven development* — write the failing test first, then code until it passes.

------------------------------------------------------------------------

4  Commit & PR etiquette
------------------------

* Make each commit atomic and write an informative message
  (`type: short imperative summary`, e.g. `fix: guard zero‑division in score.py`).
* Never use ``git add .`` – stage only the files you intend to commit.
* Run ``git diff`` (unstaged) and ``git diff --cached`` (staged) before every commit.
* Reference related issues in your PR description (`Closes #123`).

------------------------------------------------------------------------

5  Communication channels
-------------------------

* **Issues & Pull Requests** – primary place for technical discussion.
* **Discord** – quick questions and chat: https://discord.gg/DRkdKaumBs
* **Mailing list** – pgmpy@googlegroups.com (long‑form design topics).

We’re friendly and responsive—don’t hesitate to ask questions.

------------------------------------------------------------------------

Happy hacking 🚀
