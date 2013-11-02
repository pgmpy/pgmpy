#!/usr/bin/env python3
"""Contains all the user-defined Exceptions created for PgmPy"""


class MissingParentsError(Exception):
    def __init__(self, missing):
        Exception.__init__(self)
        self.missing = missing

    def __str__(self):
        return repr("Parents are missing: " + str(missing))


class ExtraParentsError(Exception):
    def __init__(self, extra):
        Exception.__init__(self)
        self.extra = extra

    def __str__(self):
        return repr("Following are not parents: " + str(extra))

class MissingStatesError(Exception):
    def __init__(self, missing):
        Exception.__init__(self)
        self.missing = missing

    def __str__(self):
        return repr("States are missing: " + str(missing))

class ExtraStatesError(Exception):
    def __init__(self, extra):
        Exception.__init__(self)
        self.extra = extra

    def __str__(self):
        return repr("Following are not states: " + str(extra))