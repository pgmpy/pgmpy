from importlib.metadata import version

from .global_vars import config, logger

__all__ = ["config", "logger"]

__version__ = version("pgmpy")
