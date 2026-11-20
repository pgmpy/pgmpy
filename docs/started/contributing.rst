Contributing to pgmpy
=====================

Thank you for your contribution to **pgmpy**. Use the links below for common actions. Then
read the sections that follow.

.. list-table::
   :header-rows: 0
   :widths: 33 67

   * - **Report a bug**
     - https://github.com/pgmpy/pgmpy/issues/new
   * - **Propose a feature**
     - https://github.com/pgmpy/pgmpy/issues/new
   * - **Open a pull request**
     - https://github.com/pgmpy/pgmpy/compare/

------------------------------------------------------------------------

1  Development workflow
-----------------------

#. **Fork** `pgmpy/pgmpy` to your GitHub account.
#. **Clone** your fork:
   ``git clone git@github.com:<your_github_username>/pgmpy.git``
#. **Create a branch** from ``dev``. Give it a name that describes the change, for example:

   ``git checkout -b feature/bic-score``
#. Make small commits, with one change in each commit
   (`git add <file>` ➜ `git commit -m "feat: add BIC score"`).
#. **Push** the branch. Then open a **pull request** to the ``dev`` branch.
#. Make the changes that the reviewers ask for. After the approval, we merge the pull request.

We use the `Git Flow <https://nvie.com/posts/a-successful-git-branching-model/>`_ branch model. All changes to
``dev`` come through a pull request.

------------------------------------------------------------------------

2  Code style & quality
-----------------------

* The repository has a pre-commit hook that checks the code style. To install it, run
  ``pre-commit install`` in the pgmpy directory.

------------------------------------------------------------------------

3  Testing
----------

* Write **pytest** tests for each new feature and bug fix.
* Run all tests on your computer: ``pytest -q``.
  All tests on the **dev** branch must always pass.
* Continuous integration runs on **GitHub Actions**. Your pull request must pass all checks
  before we merge it.

**Tip:** Use *test-driven development*. First, write a test that fails. Then write the code that
makes the test pass.

------------------------------------------------------------------------

4  Commit & PR etiquette
------------------------

* Put one change in each commit. Write the commit message in the form
  `type: short imperative summary`, for example `fix: guard zero‑division in score.py`.
* Do not use ``git add .``. Stage only the files that you want to commit.
* Run ``git diff`` (unstaged) and ``git diff --cached`` (staged) before every commit.
* Give the related issues in the description of your pull request (`Closes #123`).

------------------------------------------------------------------------

5  Communication channels
-------------------------

* **Issues & Pull Requests**: the main place for technical discussions.
* **Discord**: short questions and chat: https://discord.gg/DRkdKaumBs
* **Mailing list**: pgmpy@googlegroups.com, for long design discussions.

Ask questions at any time.
