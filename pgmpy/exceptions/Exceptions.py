#!/usr/bin/env python3
"""Contains all the user-defined exceptions created for PgmPy"""


class MissingParentsError(Exception):
    def __init__(self, *missing):
        self.missing = missing

    def __str__(self):
        return repr("Parents are missing: " + str(self.missing))


class ExtraParentsError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr("Following are not parents: " + str(self.extra))


class MissingStatesError(Exception):
    def __init__(self, *missing):
        self.missing = missing

    def __str__(self):
        return repr("States are missing: " + str(self.missing))


class ExtraStatesError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr("Following are not states: " + str(self.extra))


class SelfLoopError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class CycleError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class StateError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class NodeNotFoundError(Exception):
    def __init__(self, *extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class ScopeError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class SizeError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class CardinalityError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class RequiredError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class ModelError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))


class InvalidValueError(Exception):
    def __init__(self, extra):
        self.extra = extra

    def __str__(self):
        return repr(str(self.extra))
