# import math
# from itertools import combinations
# from typing import Any, Callable, Optional
#
# import networkx as nx
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn.metrics import f1_score
# from tqdm import tqdm
#
# from pgmpy import config
# from pgmpy.base import DAG
# from pgmpy.models import DiscreteBayesianNetwork
#
#
# def get_metrics(metrics: Optional[tuple[str, Callable]] = None) -> Any:
#
#     name_to_fn = {
#         "correlation": correlation_score,
#         "log-likelihood": log_likelihood_score,
#         "aic": structure_score,
#         "bic": structure_score,
#         "implied-cis": implied_cis,
#         "fisher-c": fisher_c,
#     }
#     fn_to_name = {v: k for k, v in name_to_fn.items()}
#
#     if metrics is None:
#         return name_to_fn
#
#     callable_metrics = {}
#     for metric in metrics:
#         if isinstance(metric, str):
#             metric = metric.lower()
#             if metric not in name_to_fn:
#                 raise ValueError(
#                     f"Unknown metric method. Available metrics are: {list(name_to_fn.keys())}"
#                 )
#
#             callable_metrics[metric] = name_to_fn[metric]
#
#         elif callable(metric):
#             if metric not in fn_to_name:
#                 raise ValueError(
#                     f"Got unknown metric function {metric}. Available metrics are: {list(fn_to_name.keys())}"
#                 )
#             metric_name = fn_to_name.get(metric)
#             callable_metrics[metric_name] = metric
#
#         else:
#             raise ValueError(
#                 f"`metric` must be one of {list(name_to_fn.keys())} and of type str or Callable"
#             )
#
#     return callable_metrics
#
#
# def structure_score(model, data, scoring_method="bic-g", **kwargs):
#     """
#     Uses the standard model scoring methods to give a score for each structure.
#     The score doesn't have very straight forward interpretebility but can be
#     used to compare different models. A higher score represents a better fit.
#     This method only needs the model structure to compute the score and parameters
#     aren't required.
#
#     Parameters
#     ----------
#     model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork instance
#         The model whose score needs to be computed.
#
#     data: pd.DataFrame instance
#         The dataset against which to score the model.
#
#     scoring_method: str
#         Options are: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg
#
#     kwargs: kwargs
#         Any additional parameters that needs to be passed to the
#         scoring method. Check pgmpy.estimators.StructureScore for details.
#
#     Returns
#     -------
#     Model score: float
#         A score value for the model.
#
#     Examples
#     --------
#     >>> from pgmpy.utils import get_example_model
#     >>> from pgmpy.metrics import structure_score
#     >>> model = get_example_model("alarm")
#     >>> data = model.simulate(int(1e4))
#     >>> structure_score(model, data, scoring_method="bic-g")
#     -106665.9383064447
#     """
#     from pgmpy.estimators import (
#         AIC,
#         BIC,
#         K2,
#         AICCondGauss,
#         AICGauss,
#         BDeu,
#         BDs,
#         BICCondGauss,
#         BICGauss,
#         LogLikelihoodCondGauss,
#         LogLikelihoodGauss,
#     )
#
#     supported_methods = {
#         "k2": K2,
#         "bdeu": BDeu,
#         "bds": BDs,
#         "bic-d": BIC,
#         "aic-d": AIC,
#         "ll-g": LogLikelihoodGauss,
#         "aic-g": AICGauss,
#         "bic-g": BICGauss,
#         "ll-cg": LogLikelihoodCondGauss,
#         "aic-cg": AICCondGauss,
#         "bic-cg": BICCondGauss,
#     }
#
#     # Step 1: Test the inputs
#     if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
#         raise ValueError(
#             f"model must be an instance of pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork. Got {type(model)}"
#         )
#     elif not isinstance(data, pd.DataFrame):
#         raise ValueError(f"data must be a pandas.DataFrame instance. Got {type(data)}")
#     elif set(model.nodes()) != set(data.columns):
#         raise ValueError(
#             f"Missing columns in data. Can't find values for the following variables: "
#             f" {set(model.nodes()) - set(data.columns)}"
#         )
#     elif (scoring_method not in supported_methods.keys()) and (
#         not callable(scoring_method)
#     ):
#         raise ValueError(
#             f"scoring method not supported and not a callable. Got {scoring_method}"
#         )
#
#     # Step 2: Compute the score and return
#     return supported_methods[scoring_method](data, **kwargs).score(model)
