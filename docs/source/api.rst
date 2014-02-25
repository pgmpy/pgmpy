The pgmpy API Reference
=======================

.. automodule:: pgmpy
   :members:

"BayesianModel" module
----------------------

.. module:: pgmpy.BayesianModel

.. autosummary::
   ..toctree: generated/

.. autoclass:: BayesianModel
   :members:

      **Base class for "Bayesian Model"**.

      A BayesianModel stores nodes and edges with conditional probability
      distribution (cpd) and other attributes.

      BayesianModel hold directed edges.  Self loops are not allowed neither
      multiple (parallel) edges.

      Nodes should be strings.

      Edges are represented as links between nodes.

      Parameters
      ----------
      data : input graph
          Data to initialize graph.  If data=None (default) an empty
          graph is created.  The data can be an edge list, or any
          NetworkX graph object.

      See Also
      --------

      Examples
      --------
      Create an empty bayesian model with no nodes and no edges.

      >>> from pgmpy import BayesianModel as bm
      >>> G = bm.BayesianModel()

      G can be grown in several ways.

      **Nodes:**

      Add one node at a time:

      >>> G.add_node('a')

      Add the nodes from any container (a list, set or tuple or the nodes
      from another graph).

      >>> G.add_nodes_from(['a', 'b'])

      **Edges:**

      G can also be grown by adding edges.

      Add one edge,

      >>> G.add_edge('a', 'b')

      a list of edges,

      >>> G.add_edges_from([('a', 'b'), ('b', 'c')])

      If some edges connect nodes not yet in the model, the nodes
      are added automatically.  There are no errors when adding
      nodes or edges that already exist.

      **Shortcuts:**

      Many common graph features allow python syntax to speed reporting.

      >>> 'a' in G     # check if node in graph
      True
      >>> len(G)  # number of nodes in graph
      3


.. autoclass:: IMap
   :members:

"Factor" module
---------------

.. automodule:: pgmpy.Factor
   :members:
