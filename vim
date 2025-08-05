usage: pytest [options] [file_or_dir] [file_or_dir] [...]

positional arguments:
  file_or_dir

general:
  -k EXPRESSION         Only run tests which match the given substring
                        expression. An expression is a Python evaluable
                        expression where all names are substring-matched against
                        test names and their parent classes. Example: -k
                        'test_method or test_other' matches all test functions
                        and classes whose name contains 'test_method' or
                        'test_other', while -k 'not test_method' matches those
                        that don't contain 'test_method' in their names. -k 'not
                        test_method and not test_other' will eliminate the
                        matches. Additionally keywords are matched to classes
                        and functions containing extra names in their
                        'extra_keyword_matches' set, as well as functions which
                        have names assigned directly to them. The matching is
                        case-insensitive.
  -m MARKEXPR           Only run tests matching given mark expression. For
                        example: -m 'mark1 and not mark2'.
  --markers             show markers (builtin, plugin and per-project ones).
  -x, --exitfirst       Exit instantly on first error or failed test
  --maxfail=num         Exit after first num failures or errors
  --strict-config       Any warnings encountered while parsing the `pytest`
                        section of the configuration file raise errors
  --strict-markers      Markers not registered in the `markers` section of the
                        configuration file raise errors
  --strict              (Deprecated) alias to --strict-markers
  --fixtures, --funcargs
                        Show available fixtures, sorted by plugin appearance
                        (fixtures with leading '_' are only shown with '-v')
  --fixtures-per-test   Show fixtures per test
  --pdb                 Start the interactive Python debugger on errors or
                        KeyboardInterrupt
  --pdbcls=modulename:classname
                        Specify a custom interactive Python debugger for use
                        with --pdb.For example:
                        --pdbcls=IPython.terminal.debugger:TerminalPdb
  --trace               Immediately break when running each test
  --capture=method      Per-test capturing method: one of fd|sys|no|tee-sys
  -s                    Shortcut for --capture=no
  --runxfail            Report the results of xfail tests as if they were not
                        marked
  --lf, --last-failed   Rerun only the tests that failed at the last run (or all
                        if none failed)
  --ff, --failed-first  Run all tests, but run the last failures first. This may
                        re-order tests and thus lead to repeated fixture
                        setup/teardown.
  --nf, --new-first     Run tests from new files first, then the rest of the
                        tests sorted by file mtime
  --cache-show=[CACHESHOW]
                        Show cache contents, don't perform collection or tests.
                        Optional argument: glob (default: '*').
  --cache-clear         Remove all cache contents at start of test run
  --lfnf, --last-failed-no-failures={all,none}
                        With ``--lf``, determines whether to execute tests when
                        there are no previously (known) failures or when no
                        cached ``lastfailed`` data was found. ``all`` (the
                        default) runs the full test suite again. ``none`` just
                        emits a message about no known failures and exits
                        successfully.
  --sw, --stepwise      Exit on test failure and continue from last failing test
                        next time
  --sw-skip, --stepwise-skip
                        Ignore the first failing test but stop on the next
                        failing test. Implicitly enables --stepwise.
  --sw-reset, --stepwise-reset
                        Resets stepwise state, restarting the stepwise workflow.
                        Implicitly enables --stepwise.

Reporting:
  --durations=N         Show N slowest setup/test durations (N=0 for all)
  --durations-min=N     Minimal duration in seconds for inclusion in slowest
                        list. Default: 0.005 (or 0.0 if -vv is given).
  -v, --verbose         Increase verbosity
  --no-header           Disable header
  --no-summary          Disable summary
  --no-fold-skipped     Do not fold skipped tests in short summary.
  --force-short-summary
                        Force condensed summary output regardless of verbosity
                        level.
  -q, --quiet           Decrease verbosity
  --verbosity=VERBOSE   Set verbosity. Default: 0.
  -r chars              Show extra test summary info as specified by chars:
                        (f)ailed, (E)rror, (s)kipped, (x)failed, (X)passed,
                        (p)assed, (P)assed with output, (a)ll except passed
                        (p/P), or (A)ll. (w)arnings are enabled by default (see
                        --disable-warnings), 'N' can be used to reset the list.
                        (default: 'fE').
  --disable-warnings, --disable-pytest-warnings
                        Disable warnings summary
  -l, --showlocals      Show locals in tracebacks (disabled by default)
  --no-showlocals       Hide locals in tracebacks (negate --showlocals passed
                        through addopts)
  --tb=style            Traceback print mode (auto/long/short/line/native/no)
  --xfail-tb            Show tracebacks for xfail (as long as --tb != no)
  --show-capture={no,stdout,stderr,log,all}
                        Controls how captured stdout/stderr/log is shown on
                        failed tests. Default: all.
  --full-trace          Don't cut any tracebacks (default is to cut)
  --color=color         Color terminal output (yes/no/auto)
  --code-highlight={yes,no}
                        Whether code should be highlighted (only if --color is
                        also enabled). Default: yes.
  --pastebin=mode       Send failed|all info to bpaste.net pastebin service
  --junitxml, --junit-xml=path
                        Create junit-xml style report file at given path
  --junitprefix, --junit-prefix=str
                        Prepend prefix to classnames in junit-xml output

pytest-warnings:
  -W, --pythonwarnings PYTHONWARNINGS
                        Set which warnings to report, see -W option of Python
                        itself

collection:
  --collect-only, --co  Only collect tests, don't execute them
  --pyargs              Try to interpret all arguments as Python packages
  --ignore=path         Ignore path during collection (multi-allowed)
  --ignore-glob=path    Ignore path pattern during collection (multi-allowed)
  --deselect=nodeid_prefix
                        Deselect item (via node id prefix) during collection
                        (multi-allowed)
  --confcutdir=dir      Only load conftest.py's relative to specified dir
  --noconftest          Don't load any conftest.py files
  --keep-duplicates     Keep duplicate tests
  --collect-in-virtualenv
                        Don't ignore tests in a local virtualenv directory
  --continue-on-collection-errors
                        Force test execution even if collection errors occur
  --import-mode={prepend,append,importlib}
                        Prepend/append to sys.path when importing test modules
                        and conftest files. Default: prepend.
  --doctest-modules     Run doctests in all .py modules
  --doctest-report={none,cdiff,ndiff,udiff,only_first_failure}
                        Choose another output format for diffs on doctest
                        failure
  --doctest-glob=pat    Doctests file matching pattern, default: test*.txt
  --doctest-ignore-import-errors
                        Ignore doctest collection errors
  --doctest-continue-on-failure
                        For a given doctest, continue to run after the first
                        failure
  --xdoctest-modules, --xdoctest, --xdoc
                        Run doctests in all .py modules using new style parsing
  --xdoctest-glob, --xdoc-glob=pat
                        Text files matching this pattern will be checked for
                        doctests. This option may be specified multiple times.
                        XDoctest does not check any text files by default. For
                        compatibility with doctest set this to test*.txt
  --xdoctest-ignore-syntax-errors
                        Ignore xdoctest SyntaxErrors
  --xdoctest-style, --xdoc-style={freeform,google,auto}
                        Basic style used to write doctests
  --xdoctest-analysis, --xdoc-analysis={static,dynamic,auto}
                        How doctests are collected. Can either be static,
                        dynamic, or auto
  --xdoctest-colored, --xdoc-colored=XDOCTEST_COLORED
                        Enable or disable ANSI coloration in stdout
  --xdoctest-nocolor, --xdoc-nocolor
                        Disable ANSI coloration in stdout
  --xdoctest-offset, --xdoc-offset
                        If True formatted source linenumbers will agree with
                        their location in the source file. Otherwise they will
                        be relative to the doctest itself.
  --xdoctest-report, --xdoc-report={none,cdiff,ndiff,udiff,only_first_failure}
                        Choose another output format for diffs on xdoctest
                        failure
  --xdoctest-options, --xdoc-options=XDOCTEST_OPTIONS
                        Default directive flags for doctests
  --xdoctest-global-exec, --xdoc-global-exec=XDOCTEST_GLOBAL_EXEC
                        Custom Python code to execute before every test
  --xdoctest-supress-import-errors, --xdoc-supress-import-errors
                        Removes tracebacks from errors in implicit imports
  --xdoctest-verbose, --xdoc-verbose=XDOCTEST_VERBOSE
                        Verbosity level. 0 is silent, 1 prints out test names, 2
                        additionally prints test stdout, 3 additionally prints
                        test source
  --xdoctest-quiet, --xdoc-quiet
                        sets verbosity to 1
  --xdoctest-silent, --xdoc-silent
                        sets verbosity to 0

test session debugging and configuration:
  -c, --config-file FILE
                        Load configuration from `FILE` instead of trying to
                        locate one of the implicit configuration files.
  --rootdir=ROOTDIR     Define root directory for tests. Can be relative path:
                        'root_dir', './root_dir', 'root_dir/another_dir/';
                        absolute path: '/home/user/root_dir'; path with
                        variables: '$HOME/root_dir'.
  --basetemp=dir        Base temporary directory for this test run. (Warning:
                        this directory is removed if it exists.)
  -V, --version         Display pytest version and information about plugins.
                        When given twice, also display information about
                        plugins.
  -h, --help            Show help message and configuration info
  -p name               Early-load given plugin module name or entry point
                        (multi-allowed). To avoid loading of plugins, use the
                        `no:` prefix, e.g. `no:doctest`. See also --disable-
                        plugin-autoload.
  --disable-plugin-autoload
                        Disable plugin auto-loading through entry point
                        packaging metadata. Only plugins explicitly specified in
                        -p or env var PYTEST_PLUGINS will be loaded.
  --trace-config        Trace considerations of conftest.py files
  --debug=[DEBUG_FILE_NAME]
                        Store internal tracing debug information in this log
                        file. This file is opened with 'w' and truncated as a
                        result, care advised. Default: pytestdebug.log.
  -o, --override-ini OVERRIDE_INI
                        Override ini option with "option=value" style, e.g. `-o
                        xfail_strict=True -o cache_dir=cache`.
  --assert=MODE         Control assertion debugging tools.
                        'plain' performs no assertion debugging.
                        'rewrite' (the default) rewrites assert statements in
                        test modules on import to provide assert expression
                        information.
  --setup-only          Only setup fixtures, do not execute tests
  --setup-show          Show setup of fixtures while executing tests
  --setup-plan          Show what fixtures and tests would be executed but don't
                        execute anything

logging:
  --log-level=LEVEL     Level of messages to catch/display. Not set by default,
                        so it depends on the root/parent log handler's effective
                        level, where it is "WARNING" by default.
  --log-format=LOG_FORMAT
                        Log format used by the logging module
  --log-date-format=LOG_DATE_FORMAT
                        Log date format used by the logging module
  --log-cli-level=LOG_CLI_LEVEL
                        CLI logging level
  --log-cli-format=LOG_CLI_FORMAT
                        Log format used by the logging module
  --log-cli-date-format=LOG_CLI_DATE_FORMAT
                        Log date format used by the logging module
  --log-file=LOG_FILE   Path to a file when logging will be written to
  --log-file-mode={w,a}
                        Log file open mode
  --log-file-level=LOG_FILE_LEVEL
                        Log file logging level
  --log-file-format=LOG_FILE_FORMAT
                        Log format used by the logging module
  --log-file-date-format=LOG_FILE_DATE_FORMAT
                        Log date format used by the logging module
  --log-auto-indent=LOG_AUTO_INDENT
                        Auto-indent multiline messages passed to the logging
                        module. Accepts true|on, false|off or an integer.
  --log-disable=LOGGER_DISABLE
                        Disable a logger by name. Can be passed multiple times.

distributed and subprocess testing:
  -n, --numprocesses numprocesses
                        Shortcut for '--dist=load --tx=NUM*popen'.
                        With 'logical', attempt to detect logical CPU count
                        (requires psutil, falls back to 'auto').
                        With 'auto', attempt to detect physical CPU count. If
                        physical CPU count cannot be determined, falls back to
                        1.
                        Forced to 0 (disabled) when used with --pdb.
  --maxprocesses=maxprocesses
                        Limit the maximum number of workers to process the tests
                        when using --numprocesses with 'auto' or 'logical'
  --max-worker-restart=MAXWORKERRESTART
                        Maximum number of workers that can be restarted when
                        crashed (set to zero to disable this feature)
  --dist=distmode       Set mode for distributing tests to exec environments.
                        each: Send each test to all available environments.
                        load: Load balance by sending any pending test to any
                        available environment.
                        loadscope: Load balance by sending pending groups of
                        tests in the same scope to any available environment.
                        loadfile: Load balance by sending test grouped by file
                        to any available environment.
                        loadgroup: Like 'load', but sends tests marked with
                        'xdist_group' to the same worker.
                        worksteal: Split the test suite between available
                        environments, then re-balance when any worker runs out
                        of tests.
                        (default) no: Run tests inprocess, don't distribute.
  --loadscope-reorder   Pytest-xdist will default reorder tests by number of
                        tests per scope when used in conjunction with loadscope.
                        This option will enable loadscope reorder which will
                        improve the parallelism of the test suite.
                        However, the partial order of tests might not be
                        retained.
  --no-loadscope-reorder
                        Pytest-xdist will default reorder tests by number of
                        tests per scope when used in conjunction with loadscope.
                        This option will disable loadscope reorder, and the
                        partial order of tests can be retained.
                        This is useful when pytest-xdist is used together with
                        other plugins that specify tests in a specific order.
  --tx=xspec            Add a test execution environment. Some examples:
                        --tx popen//python=python2.5 --tx
                        socket=192.168.1.102:8888
                        --tx ssh=user@codespeak.net//chdir=testcache
  --px=xspec            Add a proxy gateway to pass to test execution
                        environments using `via`. Example:
                        --px id=my_proxy//socket=192.168.1.102:8888 --tx
                        5*popen//via=my_proxy
  -d                    Load-balance tests. Shortcut for '--dist=load'.
  --rsyncdir=DIR        Add directory for rsyncing to remote tx nodes
  --rsyncignore=GLOB    Add expression for ignores when rsyncing to remote tx
                        nodes
  --testrunuid=TESTRUNUID
                        Provide an identifier shared amongst all workers as the
                        value of the 'testrun_uid' fixture.
                        If not provided, 'testrun_uid' is filled with a new
                        unique string on every test run.
  --maxschedchunk=MAXSCHEDCHUNK
                        Maximum number of tests scheduled in one step for
                        --dist=load.
                        Setting it to 1 will force pytest to send tests to
                        workers one by one - might be useful for a small number
                        of slow tests.
                        Larger numbers will allow the scheduler to submit
                        consecutive chunks of tests to workers - allows reusing
                        fixtures.
                        Due to implementation reasons, at least 2 tests are
                        scheduled per worker at the start. Only later tests can
                        be scheduled one by one.
                        Unlimited if not set.
  -f, --looponfail      Run tests in subprocess: wait for files to be modified,
                        then re-run failing test set until all pass.

coverage reporting with distributed testing support:
  --cov=[SOURCE]        Path or package name to measure during execution (multi-
                        allowed). Use --cov= to not do any source filtering and
                        record everything.
  --cov-reset           Reset cov sources accumulated in options so far.
  --cov-report=TYPE     Type of report to generate: term, term-missing,
                        annotate, html, xml, json, lcov (multi-allowed). term,
                        term-missing may be followed by ":skip-covered".
                        annotate, html, xml, json and lcov may be followed by
                        ":DEST" where DEST specifies the output location. Use
                        --cov-report= to not generate any output.
  --cov-config=PATH     Config file for coverage. Default: .coveragerc
  --no-cov-on-fail      Do not report coverage if test run fails. Default: False
  --no-cov              Disable coverage report completely (useful for
                        debuggers). Default: False
  --cov-fail-under=MIN  Fail if the total coverage is less than MIN.
  --cov-append          Do not delete coverage but append to current. Default:
                        False
  --cov-branch          Enable branch coverage.
  --cov-precision=COV_PRECISION
                        Override the reporting precision.
  --cov-context=CONTEXT
                        Dynamic contexts to use. "test" for now.

[pytest] ini-options in the first pytest.ini|tox.ini|setup.cfg|pyproject.toml file found:

  markers (linelist):   Register new markers for test functions
  empty_parameter_set_mark (string):
                        Default marker for empty parametersets
  filterwarnings (linelist):
                        Each line specifies a pattern for
                        warnings.filterwarnings. Processed after
                        -W/--pythonwarnings.
  norecursedirs (args): Directory patterns to avoid for recursion
  testpaths (args):     Directories to search for tests when no files or
                        directories are given on the command line
  collect_imported_tests (bool):
                        Whether to collect tests in imported modules outside
                        `testpaths`
  consider_namespace_packages (bool):
                        Consider namespace packages when resolving module names
                        during import
  usefixtures (args):   List of default fixtures to be used with this project
  python_files (args):  Glob-style file patterns for Python test module
                        discovery
  python_classes (args):
                        Prefixes or glob names for Python test class discovery
  python_functions (args):
                        Prefixes or glob names for Python test function and
                        method discovery
  disable_test_id_escaping_and_forfeit_all_rights_to_community_support (bool):
                        Disable string escape non-ASCII characters, might cause
                        unwanted side effects(use at your own risk)
  console_output_style (string):
                        Console output: "classic", or with additional progress
                        information ("progress" (percentage) | "count" |
                        "progress-even-when-capture-no" (forces progress even
                        when capture=no)
  verbosity_test_cases (string):
                        Specify a verbosity level for test case execution,
                        overriding the main level. Higher levels will provide
                        more detailed information about each test case executed.
  xfail_strict (bool):  Default for the strict parameter of xfail markers when
                        not given explicitly (default: False)
  tmp_path_retention_count (string):
                        How many sessions should we keep the `tmp_path`
                        directories, according to `tmp_path_retention_policy`.
  tmp_path_retention_policy (string):
                        Controls which directories created by the `tmp_path`
                        fixture are kept around, based on test outcome.
                        (all/failed/none)
  enable_assertion_pass_hook (bool):
                        Enables the pytest_assertion_pass hook. Make sure to
                        delete any previously generated pyc cache files.
  truncation_limit_lines (string):
                        Set threshold of LINES after which truncation will take
                        effect
  truncation_limit_chars (string):
                        Set threshold of CHARS after which truncation will take
                        effect
  verbosity_assertions (string):
                        Specify a verbosity level for assertions, overriding the
                        main level. Higher levels will provide more detailed
                        explanation when an assertion fails.
  junit_suite_name (string):
                        Test suite name for JUnit report
  junit_logging (string):
                        Write captured log messages to JUnit report: one of
                        no|log|system-out|system-err|out-err|all
  junit_log_passing_tests (bool):
                        Capture log information for passing tests to JUnit
                        report:
  junit_duration_report (string):
                        Duration time to report: one of total|call
  junit_family (string):
                        Emit XML for schema: one of legacy|xunit1|xunit2
  doctest_optionflags (args):
                        Option flags for doctests
  doctest_encoding (string):
                        Encoding used for doctest files
  cache_dir (string):   Cache directory path
  log_level (string):   Default value for --log-level
  log_format (string):  Default value for --log-format
  log_date_format (string):
                        Default value for --log-date-format
  log_cli (bool):       Enable log display during test run (also known as "live
                        logging")
  log_cli_level (string):
                        Default value for --log-cli-level
  log_cli_format (string):
                        Default value for --log-cli-format
  log_cli_date_format (string):
                        Default value for --log-cli-date-format
  log_file (string):    Default value for --log-file
  log_file_mode (string):
                        Default value for --log-file-mode
  log_file_level (string):
                        Default value for --log-file-level
  log_file_format (string):
                        Default value for --log-file-format
  log_file_date_format (string):
                        Default value for --log-file-date-format
  log_auto_indent (string):
                        Default value for --log-auto-indent
  faulthandler_timeout (string):
                        Dump the traceback of all threads if a test takes more
                        than TIMEOUT seconds to finish
  addopts (args):       Extra command line options
  minversion (string):  Minimally required pytest version
  pythonpath (paths):   Add paths to sys.path
  required_plugins (args):
                        Plugins that must be present for pytest to run
  rsyncdirs (paths):    list of (relative) paths to be rsynced for remote
                        distributed testing.
  rsyncignore (paths):  list of (relative) glob-style paths to be ignored for
                        rsyncing.
  looponfailroots (paths):
                        directories to check for changes. Default: current
                        directory.
  xdoctest_encoding (string):
                        encoding used for xdoctest files

Environment variables:
  CI                       When set (regardless of value), pytest knows it is running in a CI process and does not truncate summary info
  BUILD_NUMBER             Equivalent to CI
  PYTEST_ADDOPTS           Extra command line options
  PYTEST_PLUGINS           Comma-separated plugins to load during startup
  PYTEST_DISABLE_PLUGIN_AUTOLOAD Set to disable plugin auto-loading
  PYTEST_DEBUG             Set to enable debug tracing of pytest's internals


to see available markers type: pytest --markers
to see available fixtures type: pytest --fixtures
(shown according to specified file_or_dir or current dir if not specified; fixtures with leading '_' are only shown with the '-v' option
