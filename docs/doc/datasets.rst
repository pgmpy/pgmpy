Example Datasets
================

pgmpy includes a collection of built-in datasets for testing, benchmarking, and
learning. Datasets can be loaded with a single function call and are returned
as pandas DataFrames.

Usage
-----

.. code-block:: python

    from pgmpy.datasets import load_dataset, list_datasets

    # List all available datasets
    print(list_datasets())

    # Load a dataset
    df = load_dataset("sachs_discrete")
    print(df.head())

Available Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Dataset
     - Type
     - Description
   * - sachs_discrete
     - Discrete
     - Flow cytometry protein signaling (discretized)
   * - sachs_continuous
     - Continuous
     - Flow cytometry protein signaling (continuous)
   * - sachs_mixed
     - Mixed
     - Flow cytometry protein signaling (mixed types)
   * - adult
     - Mixed
     - UCI Adult income dataset
   * - wine_quality_red
     - Continuous
     - UCI red wine quality
   * - wine_quality_white
     - Continuous
     - UCI white wine quality
   * - boston_housing
     - Continuous
     - Boston housing prices
   * - pima_diabetes
     - Mixed
     - Pima Indians diabetes dataset
   * - auto_mpg
     - Continuous
     - Auto MPG fuel consumption
   * - student_performance
     - Mixed
     - Student academic performance
   * - abalone_continuous
     - Continuous
     - Abalone age prediction
   * - abalone_mixed
     - Mixed
     - Abalone age prediction (mixed types)
   * - south_german_credit
     - Mixed
     - South German credit scoring
   * - credit_approval
     - Mixed
     - UCI credit approval
   * - cover_type
     - Mixed
     - Forest cover type prediction
   * - dry_bean
     - Continuous
     - Dry bean classification
   * - hitters
     - Mixed
     - Baseball player salary data
   * - htru2
     - Continuous
     - Pulsar star identification
   * - airfoil
     - Continuous
     - NASA airfoil self-noise
   * - yacht_hydrodynamics
     - Continuous
     - Yacht hull resistance prediction
   * - seoul_bike
     - Mixed
     - Seoul bike sharing demand
   * - superconductivity
     - Continuous
     - Superconductor critical temperature
   * - residential_building
     - Continuous
     - Residential building cost estimation
   * - hungary_chickenpox
     - Continuous
     - Hungary chickenpox cases (time series)
   * - algerian_forest
     - Mixed
     - Algerian forest fire prediction
   * - blue_driver
     - Continuous
     - Blue driver dataset
   * - apple_watch_fitbit
     - Continuous
     - Apple Watch and Fitbit sensor data
   * - cities
     - Mixed
     - City characteristics dataset
   * - college_plans
     - Discrete
     - Student college plan survey
   * - contraceptive_method
     - Mixed
     - Contraceptive method choice
   * - cystic_fibrosis
     - Mixed
     - Cystic fibrosis diagnosis
   * - depression_coping
     - Discrete
     - Depression and coping mechanisms
   * - dropouts
     - Mixed
     - Academic dropout prediction
   * - galton_stature
     - Continuous
     - Galton's height inheritance data
   * - goldberg
     - Mixed
     - Goldberg dataset
   * - iq_brain_size
     - Continuous
     - IQ and brain size relationship
   * - lead
     - Mixed
     - Lead exposure dataset
   * - myocardial_infarction
     - Mixed
     - Myocardial infarction risk factors
   * - pittsburgh_bridges
     - Discrete
     - Pittsburgh bridges dataset
   * - spartina
     - Continuous
     - Spartina grass species data
   * - uscrime
     - Mixed
     - US crime statistics
