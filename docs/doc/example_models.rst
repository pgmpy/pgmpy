Example Models
==============

pgmpy provides pre-built Bayesian Networks from the
`bnlearn repository <http://www.bnlearn.com/bnrepository>`_ and
`dagitty <https://www.dagitty.net/>`_. These models come fully parameterized
and can be used for testing, benchmarking, and learning.

Usage
-----

.. code-block:: python

    from pgmpy.utils import get_example_model

    # Load a pre-built Bayesian Network
    model = get_example_model("alarm")

    # Inspect the model
    print(model.nodes())
    print(model.edges())
    print(model.get_cpds("HISTORY"))

    # Simulate data from the model
    df = model.simulate(n_samples=1000)

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

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - sangiovese
     - Sangiovese grape quality
   * - mehra
     - Mehra dataset

DAGs (Structure Only)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - M-bias
     - Classic M-bias confounding structure
   * - confounding
     - Simple confounding example
   * - mediator
     - Mediation DAG
   * - paths
     - Multiple causal paths example
   * - Sebastiani_2005
     - Sickle cell disease (Sebastiani et al., 2005)
   * - Polzer_2012
     - Obesity and health outcomes (Polzer et al., 2012)
   * - Schipf_2010
     - Metabolic syndrome (Schipf et al., 2010)
   * - Shrier_2008
     - Sports injury prevention (Shrier et al., 2008)
