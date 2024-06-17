===============
Plotting Models
===============

pgmpy offers a few different options to visualize the model.

1. Using `pygraphviz` (https://pygraphviz.github.io/)
2. Using `networkx.drawing` module (https://networkx.org/documentation/stable/reference/drawing.html)
3. Using `daft` (https://docs.daft-pgm.org/)

Using `pygraphviz`
------------------

Lastly, pgmpy models can be converted to pygraphviz objects that can then be used to make the plots.

.. code-block:: python

   # Get an example model
   from pgmpy.utils import get_example_model
   model = get_example_model("sachs")

   # Convert model into pygraphviz object
   model_graphviz = model.to_graphviz()

   # Plot the model.
   model_graphviz.draw("sachs.png", prog="dot")

   # Other file formats can also be specified.
   model_graphviz.draw("sachs.pdf", prog="dot")
   model_graphviz.draw("sachs.svg", prog="dot")

The output `sachs.png` would look like. Users can also tryout other layout methods supported by pygraphviz such as: `neato`, `dot`, `twopi`, `circo`, `fdp`, `nop`.

.. image:: sachs.png


Using `daft`
------------
Daft is a python package that uses matplotlib to render models suitable for publication.

.. code-block:: python

   # Get an example model
   from pgmpy.utils import get_example_model
   model = get_example_model("sachs")

   # Get a daft object.




Using `networkx.drawing`
------------------------

As both `pgmpy.models.BayesianNetwork` and `pgmpy.base.DAG` inherit `networkx.DiGraph`, all of networkx's drawing functionality can be directly used on both DAGs and Bayesian Networks.

.. code-block:: python

   # Get an example model
   from pgmpy.utils import get_example_model
   model = get_example_model("sachs")

   # Plot the model
