Example Models
==============

.. meta::
   :description: Load pre-built Bayesian Networks for inference, simulation, and benchmarking.

pgmpy provides pre-built Bayesian Networks for testing, benchmarking, and
learning.

These models are fully parameterized, so you can run inference immediately or
simulate data from the joint distribution of the network.

Example
-------

.. code-block:: python

    from pgmpy.inference import VariableElimination
    from pgmpy.utils import get_example_model

    model = get_example_model("alarm")
    infer = VariableElimination(model)
    variable = list(model.nodes())[0]
    query = infer.query(variables=[variable])
    print(query)

Available Models
----------------

Discrete Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model
     - Size
     - Description
   * - asia
     - Small
     - Lung disease diagnosis network
   * - cancer
     - Small
     - Cancer diagnosis network
   * - earthquake
     - Small
     - Earthquake and burglary alarm network
   * - sachs
     - Small
     - Protein signaling network
   * - survey
     - Small
     - Student survey network
   * - alarm
     - Medium
     - Medical monitoring alarm network
   * - barley
     - Medium
     - Barley crop yield network
   * - child
     - Medium
     - Congenital heart disease diagnosis
   * - insurance
     - Medium
     - Insurance risk assessment
   * - mildew
     - Medium
     - Mildew crop disease network
   * - water
     - Medium
     - Water treatment network
   * - hailfinder
     - Large
     - Severe weather forecasting
   * - hepar2
     - Large
     - Liver disorder diagnosis
   * - win95pts
     - Large
     - Windows 95 printer troubleshooting
   * - andes
     - Very Large
     - Intelligent tutoring system
   * - diabetes
     - Very Large
     - Diabetes patient management
   * - link
     - Very Large
     - Linkage analysis network
   * - munin1 -- munin4, munin
     - Very Large
     - Electromyography diagnosis
   * - pathfinder
     - Very Large
     - Pathology diagnosis
   * - pigs
     - Very Large
     - Pedigree genetics network

Gaussian Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - ecoli70
     - E. coli gene regulatory network
   * - magic-niab
     - MAGIC wheat population (NIAB)
   * - magic-irri
     - MAGIC rice population (IRRI)
   * - arth150
     - Arabidopsis gene network

Conditional Linear Gaussian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
