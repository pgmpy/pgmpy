#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == "__main__":
    import sys

    import pytest

    package_name = "pgmpy"
    pytest_args = [
        "--cov-config",
        ".coveragerc",
        "--cov-report",
        "html",
        "--cov-report",
        "term",
        "--cov=" + package_name,
        "--verbose",
    ]
    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
