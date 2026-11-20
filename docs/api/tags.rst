Tags
====

pgmpy objects describe themselves with `skbase <https://skbase.readthedocs.io>`_ tags. For example, tags give the
data types that a causal discovery algorithm accepts, or the assumptions of its guarantees. Read tags with
``get_tag`` or ``get_class_tag``. Use :func:`skbase.lookup.all_objects` to find objects by their tags:

.. code-block:: python

    from skbase.lookup import all_objects

    from pgmpy.causal_discovery._base import BaseCausalDiscovery

    all_objects(
        object_types=BaseCausalDiscovery,
        package_name="pgmpy.causal_discovery",
        filter_tags={"data_types": "continuous", "assumption:linearity": False},
    )

:mod:`pgmpy.registry` registers all tags.

.. autosummary::
   :toctree: generated/registry
   :template: autosummary/function.rst

   ~pgmpy.registry.all_tags
   ~pgmpy.registry.check_tag_is_valid

Causal Discovery Algorithms
---------------------------

.. pgmpy-tags:: causal_discovery

Conditional Independence (CI) Tests
-----------------------------------

.. pgmpy-tags:: ci_test

Structure Scores
----------------

.. pgmpy-tags:: structure_score

Bivariate Scores
----------------

.. pgmpy-tags:: bivariate_score

Supervised Metrics
------------------

.. pgmpy-tags:: supervised_metric

Unsupervised Metrics
--------------------

.. pgmpy-tags:: unsupervised_metric
