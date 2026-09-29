Tags
====

pgmpy objects describe themselves with `skbase <https://skbase.readthedocs.io>`_ tags, for example the data types a
causal discovery algorithm can handle, or the assumptions its guarantees rely on. Tags can be read with
``get_tag`` / ``get_class_tag`` and used to find objects with :func:`skbase.lookup.all_objects`:

.. code-block:: python

    from skbase.lookup import all_objects

    from pgmpy.causal_discovery._base import BaseCausalDiscovery

    all_objects(
        object_types=BaseCausalDiscovery,
        package_name="pgmpy.causal_discovery",
        filter_tags={"data_types": "continuous", "assumption:linearity": False},
    )

All tags are registered in :mod:`pgmpy.registry`.

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

Supervised Metrics
------------------

.. pgmpy-tags:: supervised_metric

Unsupervised Metrics
--------------------

.. pgmpy-tags:: unsupervised_metric
