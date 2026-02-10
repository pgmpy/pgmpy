Exporting / Importing Models
============================

.. meta::
   :description: Import and export Bayesian Networks using common file formats like BIF, UAI, XMLBIF, and XDSL.

pgmpy can read and write Bayesian Networks in common file formats to make your
models portable across tools.

In practice, these formats store both structure and CPDs so models can be
reused for inference, simulation, or further editing.

Example
-------

.. code-block:: python

    from pgmpy.inference import VariableElimination
    from pgmpy.readwrite import BIFReader, BIFWriter
    from pgmpy.utils import get_example_model

    model = get_example_model("asia")
    BIFWriter(model).write_bif("asia.bif")

    imported = BIFReader("asia.bif").get_model()
    infer = VariableElimination(imported)
    variable = list(imported.nodes())[0]
    query = infer.query(variables=[variable])
    print(query)

Supported Formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Format
     - Reader
     - Writer
     - Description
   * - BIF
     - :class:`~pgmpy.readwrite.BIF.BIFReader`
     - :class:`~pgmpy.readwrite.BIF.BIFWriter`
     - Bayesian Interchange Format. Widely used by bnlearn and other tools.
   * - UAI
     - :class:`~pgmpy.readwrite.UAI.UAIReader`
     - :class:`~pgmpy.readwrite.UAI.UAIWriter`
     - UAI inference competition format.
   * - XMLBIF
     - :class:`~pgmpy.readwrite.XMLBIF.XMLBIFReader`
     - :class:`~pgmpy.readwrite.XMLBIF.XMLBIFWriter`
     - XML-based Bayesian Interchange Format.
   * - XDSL
     - :class:`~pgmpy.readwrite.XDSL.XDSLReader`
     - :class:`~pgmpy.readwrite.XDSL.XDSLWriter`
     - GeNIe / SMILE file format.
   * - PomdpX
     - :class:`~pgmpy.readwrite.PomdpX.PomdpXReader`
     - :class:`~pgmpy.readwrite.PomdpX.PomdpXWriter`
     - POMDP XML format for partially observable models.
   * - XBN
     - :class:`~pgmpy.readwrite.XMLBeliefNetwork.XBNReader`
     - :class:`~pgmpy.readwrite.XMLBeliefNetwork.XBNWriter`
     - XML Belief Network format (Microsoft).

See Also
--------

- **API Reference:** :doc:`Reading/Writing API <../readwrite/base>`
- **Previous:** :doc:`example_models` -- pre-built Bayesian Networks
