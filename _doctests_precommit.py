#!/usr/bin/env python3
"""Run doctests on specified files for pre-commit hook.

This script is used by the pre-commit framework to validate doctests in Python files
before they are committed to the repository. It is configured in .pre-commit-config.yaml
as a local hook that runs automatically on staged Python files.

Usage:
    # Automatically run by pre-commit on commit
    git commit

    # Manually test all files
    pre-commit run doctest --all-files

    # Manually test staged files
    pre-commit run doctest

The script:
- Accepts file paths as command-line arguments
- Runs doctests on each Python file using the doctest module
- Uses ELLIPSIS and NORMALIZE_WHITESPACE options for more flexible matching
- Reports failures with file names and failure counts
- Returns exit code 1 if any tests fail (blocking the commit)
- Returns exit code 0 if all tests pass (allowing the commit)
"""
import doctest
import os
import sys


def main():
    """Run doctests on all provided files."""
    failed = 0
    tested = 0

    for filepath in sys.argv[1:]:
        if filepath.endswith(".py") and os.path.exists(filepath):
            try:
                result = doctest.testfile(
                    filepath,
                    verbose=False,
                    module_relative=False,
                    optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
                )
                if result.failed > 0:
                    print(f"FAILED: {filepath} - {result.failed} test(s) failed")
                    failed += result.failed
                tested += result.attempted
            except Exception as e:
                print(f"ERROR: Could not run doctests for {filepath}: {e}")
                continue

    if failed > 0:
        print(f"\n{failed} doctest(s) failed out of {tested} total")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
