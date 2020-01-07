#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from os.path import sys
from setuptools import setup, find_packages


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast

    if not exists(fpath):
        raise ValueError("fpath={!r} does not exist".format(fpath))
    with open(fpath, "r") as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)

    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, "id", None) == "__version__":
                    self.version = node.value.s

    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_requirements(fname="requirements.txt", with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items
    """
    from os.path import exists
    import re

    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def parse_description(fpath):
    """
    Parse the description in the README file
    """
    # Check this exists, otherwise pip install breaks
    if exists(fpath):
        with open(fpath, "r") as f:
            text = f.read()
        return text
    return ""


NAME = "pgmpy"
VERSION = parse_version("pgmpy/__init__.py")  # must be global for git tags


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description="A library for Probabilistic Graphical Models",
        packages=find_packages(exclude=["tests"]),
        author="Ankur Ankan",
        author_email="ankurankan@gmail.com",
        url="https://github.com/pgmpy/pgmpy",
        license="MIT",
        classifiers=[
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Intended Audience :: Developers",
            "Operating System :: Unix",
            "Operating System :: POSIX",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS",
            "Topic :: Scientific/Engineering",
        ],
        long_description=parse_description("README.md"),
        long_description_content_type="text/markdown",
        install_requires=parse_requirements("requirements/runtime.txt"),
        extras_require={
            "all": parse_requirements("requirements.txt"),
            "tests": parse_requirements("requirements/tests.txt"),
        },
    )
