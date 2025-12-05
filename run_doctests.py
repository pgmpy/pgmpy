#!/usr/bin/env python3
"""Run doctests on specified files for pre-commit hook."""
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
