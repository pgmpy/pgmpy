# Contributing to pgmpy

Hi! Thanks for your interest in contributing to [pgmpy](pgmpy — pgmpy 0.1.0 documentation). 
In this document we'll try to summarize everything that you need to know to do a good job.


## Code and Issues

We use [Github](https://github.com/pgmpy/pgmpy) to host our code repositories
and issues.You can look at [issues](https://github.com/pgmpy/pgmpy/issues) to report any
issues related to pgmpy. Here is a [guide](https://guides.github.com/features/issues/)
on how to report better issues.

### Git and our Branching model

#### Git

We use [Git](http://git-scm.com/) as our [version control
system](http://en.wikipedia.org/wiki/Revision_control), so the best way to
contribute is to learn how to use it and put your changes on a Git repository.
There are plenty of documentation about Git -- you can start with the [Pro Git
book](http://git-scm.com/book/).
Or You can go through the [try git tutorial](https://try.github.io/levels/1/challenges/1
)

#### Forks + GitHub Pull requests

We use
[gitflow](http://nvie.com/posts/a-successful-git-branching-model/) to manage our
branches.

Summary of our git branching model:
- Fork the desired repository on GitHub to your account;
- Clone your forked repository locally
  (`git clone git@github.com:your-username:repository-name.git`);
- Create a new branch off of `develop` with a descriptive name (for example:
  `feature/portuguese-sentiment-analysis`, `hotfix/bug-on-downloader`). You can
  do it by switching to `develop` branch (`git checkout develop`) and then
  creating a new branch (`git checkout -b name-of-the-new-branch`);
- Make changes to the codebase and commit it. <b> [Imp] </b> Make sure that tests pass for each of your commits.
- Rebase your branch on the current dev and push to your fork on GitHub (with the name as your local branch:
  `git push origin branch-name`);
- Create a pull request using the GitHub Web interface (asking us to pull the
  changes from your new branch and add the changes to our `develop` branch).;
- Wait for comments.


#### Tips

- <b> [Imp] </b>  Write [helpful commit
  messages](http://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message).
- Anything in the `dev` branch should be deployable (no failing tests).
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding then to the index (stage
  area) and with `git diff --cached` before commiting;
- If you have push access to the main repository, please do not commit directly
  to `dev`: your access should be used only to accept pull requests; if you
  want to make a new feature, you should use the same process as other
  developers so that your code can be reviewed.


### Code Guidelines

- We use [PEP8](http://www.python.org/dev/peps/pep-0008/);
- We permit 120 characters in a line, rather 79 as suggested in PEP8
- Write tests for your new features (please see "Tests" topic below);
- Always remember that [commented code is dead
  code](http://www.codinghorror.com/blog/2008/07/coding-without-comments.html);
- Name identifiers (variables, classes, functions, module names) with readable
  names (`x` is always wrong);
- When manipulating strings, use [Python's new-style
  formatting](http://docs.python.org/library/string.html#format-string-syntax)
  (`'{} = {}'.format(a, b)` instead of `'%s = %s' % (a, b)`);
- When working with files use `with open(<filename>, <option>) as f` instead of
  ` f = open(<filename>, <option>)`;
- All `#TODO` comments should be turned into issues (use our
  [GitHub issue system](https://github.com/pgmpy/pgmpy/issues));
- Run all tests before pushing (just execute `nosetests`) so you will know if your
  changes broke something;
- Try to write both Python 2 and Python3-friendly code so won't be a pain for
  us to support both versions.We use [six](https://pypi.python.org/pypi/six) to help
  us in integration across various python versions.


### Tests

We use [Travis CI](https://travis-ci.org/) for continous integration
and python [unittest module](https://docs.python.org/2/library/unittest.html) for writing tests.
You should write tests for every feature you add or bug you solve in the code.
Having automated tests for every line of our code let us make big changes
without worries: there will always be tests to verify if the changes introduced
bugs or lack of features. If we don't have tests we will be blind and every
change will come with some fear of possibly breaking something.

For a better design of your code, we recommend using a technique called
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development),
where you write your tests **before** writing the actual code that implements
the desired feature.


## Discussion

Please feel free to contact us through mail list if
you have any questions or suggestions.
Connect with us at [gitter](https://gitter.im/pgmpy/pgmpy). 
Every contribution is very welcome!

*Mailing list* : pgmpy@googlegroups.com

Happy hacking! ;)
