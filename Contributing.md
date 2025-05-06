# Contributing to pgmpy

Hi! Thanks for your interest in contributing to [pgmpy](https://pgmpy.org). This
document summarizes everything that you need to know to get started.

## Development

### Code
The latest codebase is available in the `dev` branch of the repository.

### Building from Source
To install pgmpy from the source code:
```
$ git clone https://github.com/pgmpy/pgmpy
$ cd pgmpy/
$ pip install -e .[tests]
```

To run the tests, you can use pytest:
```
$ pytest -v pgmpy
```

We also have a pre-commit [hook](https://github.com/pgmpy/pgmpy/blob/dev/.pre-commit-config.yaml) to test for code formatting before each commit. You can install it by running the following from pgmpy base directory:
```
$ pip install pre-commit
$ pre-commit install
```

If you face any problems during installation, let us know via issues or at our discord channel.

### Building Documentation
We use sphinx to build the documentation. Please refer: https://github.com/pgmpy/pgmpy/wiki/Maintenance-Guide#building-docs for steps to build docs locally.


## Code and Issues

We use [Github](https://github.com/pgmpy/pgmpy) to host all our code. We also use github
as our [issue tracker](https://github.com/pgmpy/pgmpy/issues). Please feel free to
create a new issue for any bugs, questions etc. It is very helpful if you follow the
issue template while creating new issues as it gives us enough information to reproduce
the problem. You can also refer to github's
[guide](https://guides.github.com/features/issues/) on how to efficiently use github
issues.

### Git and our Branching model

#### Git

We use [Git](http://git-scm.com/) as our [version control
system](http://en.wikipedia.org/wiki/Revision_control), so the best way to contribute is
to learn how to use it and put your changes on a Git repository. There is plenty of
online resources available to get started with Git:
- Online tool to try git: [try git
  tutorial](https://try.github.io/levels/1/challenges/1)
- Quick intro to opening your first Pull Request:
  https://www.freecodecamp.org/news/how-to-make-your-first-pull-request-on-github-3/
- Git reference: [Pro Git book](http://git-scm.com/book/).

#### Forks + GitHub Pull Requests

We use [gitflow](http://nvie.com/posts/a-successful-git-branching-model/) to manage our
branches.

Summary of our git branching model:
- Fork the desired repository on GitHub to your account.
- Clone your forked repository locally: `git clone git@github.com:your-username/repository-name.git`.
- Create a new branch off of `dev` branch with a descriptive name (for example:
  `feature/portuguese-sentiment-analysis`, `hotfix/bug-on-downloader`). You can
  do it by switching to `dev` branch: `git checkout dev` and then
  creating a new branch: `git checkout -b name-of-the-new-branch`.
- Make changes to the codebase and commit it. <b> [Imp] </b> Make sure that tests pass for each of your commits.
- Rebase your branch on the current dev and push to your fork on GitHub (with the name as your local branch:
  `git push origin branch-name`
- Create a pull request using GitHub's Web interface (asking us to pull the
  changes from your new branch and add the changes to our `dev` branch).;
- Wait for reviews and comments.


#### Tips

- <b> [Imp] </b>  Write [helpful commit
  messages](http://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message).
- Anything in the `dev` branch should be deployable (no failing tests).
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding then to the index (stage
  area) and with `git diff --cached` before committing;
- If you have push access to the main repository, please do not commit directly
  to `dev`: your access should be used only to accept pull requests; if you
  want to make a new feature, you should use the same process as other
  developers so that your code can be reviewed.


### Code Guidelines

- We use `black`(https://black.readthedocs.io/en/stable/) for our code formatting.
- Write tests for your new features (please see "Tests" topic below);
- Always remember that [commented code is dead
  code](http://www.codinghorror.com/blog/2008/07/coding-without-comments.html);
- Name identifiers (variables, classes, functions, module names) with readable
  names (`x` is always wrong);
- When manipulating strings, use [Python's f-Strings](https://realpython.com/python-f-strings/)
  (`f'{a} = {b}'` instead of `'{} = {}'.format(a, b)`);
- When working with files use `with open(<filename>, <option>) as f` instead of
  ` f = open(<filename>, <option>)`;
- All `#TODO` comments should be turned into issues (use our
  [GitHub issue system](https://github.com/pgmpy/pgmpy/issues));
- Run all tests before pushing (just execute `nosetests`) so you will know if your
  changes broke something;

### Tests

We use GitHub actions for continuous integration for all platforms. We use
[pytest](https://docs.pytest.org/en/stable/) for writing tests.  You should
write tests for every feature you add or bug you solve in the code. For a
better design of your code, we recommend using a technique called [test-driven
development](https://en.wikipedia.org/wiki/Test-driven_development), where you
write your tests **before** writing the actual code that implements the desired
feature.


## Discussion

Please feel free to contact us through our discord channel or mailing list if
you have any questions or suggestions. All contributions are very welcome!

Happy hacking! ;)
