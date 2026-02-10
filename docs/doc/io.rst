Exporting / Importing Models
============================

pgmpy supports reading and writing Bayesian Networks in several standard file
formats. This allows interoperability with other tools such as bnlearn, GeNIe,
SamIam, and others.

Usage
-----

.. code-block:: python

    from pgmpy.utils import get_example_model
    from pgmpy.readwrite import BIFWriter, BIFReader

    # Export a model to BIF format
    model = get_example_model("asia")
    writer = BIFWriter(model)
    writer.write_bif("asia.bif")

    # Import a model from BIF format
    reader = BIFReader("asia.bif")
    imported_model = reader.get_model()

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
