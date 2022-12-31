# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.21] - 2022-12-31
### Added
1. `BayesianNetwork.get_state_probability` method to compute the probability of a given evidence.
2. `BayesianEstimator.estimate_cpd` accepts weighted datasets.

### Fixed
1. Fixes bug in `CausalInference.estimate_ate` with front-door criterion.
2. Fixes inference bugs when variable has a single state.

## [0.1.20] - 2022-09-30
### Added
1. `BayesianNetwork.get_random_cpds` method to randomly parameterize a network structure.
2. Faster Variable Elimination using tensor contraction.
3. `factors.factor_sum_product` method for faster sum-product operations using tensor contraction.

### Fixed
1. Bug in `DynamicBayesianNetwork.initialize_initial_state`. #1564
2. Bug in `factors.factor_product`. #1565

### Changed
1. Runtime improvements in `DiscreteFactor.marginalize` and `DiscreteFactor.copy` methods.

## [0.1.19] - 2022-06-30
### Added
1. Adds checks for arguments to `BayesianNetwork.simulate` method.

### Fixed
1. Fixes TAN algorithm to use conditional information metric.
2. Speed ups for all estimation and inference methods.
3. Fix in stable variant of PC algorithm to give reproducible results.
4. Fix in `GibbsSampling` for it to work with variables with integral names.
5. `DAG.active_trail_nodes` allows tuples as variable names.
6. Fixes CPD and edge creation in `UAIReader`.

## [0.1.18] - 2022-03-30
### Fixed
1. Fixes `CausalInference.is_valid_backdoor_adjustment_set` to accept str arguments for `Z`.
2. Fixes `BayesianNetwork.remove_cpd` to work with integral node names.
3. Fixes `MPLP.map_query` to return the variable states instead of probability values.
4. Fixes BIFWriter to generate output in standard BIF format.

## [0.1.17] - 2021-12-30
### Added
1. Adds BayesianNetwork.states property to store states of all the variables.
2. Adds extra checks in check model for state names

### Fixed
1. Fixes typos in BayesianModel deprecation warning
2. Bug fix in printing Linear Gaussian CPD
3. Update example notebooks to work on latest dev.

## [0.1.16] - 2021-09-30
### Added
1. Adds a `fit_update` method to `BayesianNetwork` for updating model using new data.
2. Adds `simulate` method to `BayesianNetwork` and `DynamicBayesianNetwork` to simulated data under different conditions.
3. Adds `DynamicBayesianNetwork.fit` method to learn model paramters from data.
4. `ApproxInference` class to do approximate inference on models using sampling.
5. Robust tests for all sampling methods.
6. Adds `BayesianNetwork.load` and `BayesianNetwork.save` to quickly read and write files.

### Changed
1. `BayesianModel` and `MarkovModel` renamed to `BayesianNetwork` and `MarkovNetwork` respectively.
2. The default value of node position in `DAG.to_daft` method.
3. Documentation updated on the website.

### Fixed
1. Fixes bug in `DAG.is_iequivalent` method.
2. Automatically truncate table when CPD is too large.
3. Auto-adjustment of probability values when they don't exactly sum to 1.
4. tqdm works both in notebooks and terminal.
5. Fixes bug in `CausalInference.query` method.

## [0.1.15] - 2021-06-30
### Added
1. Adds network pruning for inference algrithms to reduce the size of network before
   running inference.
2. Adds support for latent variables in DAG and BayesianModel.
3. Parallel implementation for parameter estimation algorithms.
4. Adds `DAG.get_random` and `BayesianModel.get_random` methods to be able to generate random models.
5. Adds `CausalInference.query` method for doing do operation inference with or without adjustment sets.
6. Adds functionality to treesearch to do auto root and class node selection (#1418)
7. Adds option to specify virtual evidence in bayesian network inference.
8. Adds Expectation-Maximization (EM) algorithm for parameter estimation in latent variable models.
9. Add `BDeuScore` as another option for structure score when using HillClimbSearch.
10. Adds CausalInference.get_minimal_adjustment_set` for finding adjustment sets.

### Changed
1. Renames `DAG.is_active_trail` to `is_dconnected`.
2. `DAG.do` can accept multiple variables in the argument.
3. Optimizes sampling methods.
4. CI moved from travis and appveyor to github actions.
5. Drops support for python 3.6. Requires 3.7+.

### Fixed
1. Example model files were not getting included in the pypi and conda packages.
2. The order of values returned by CI tests was wrong. #1403
3. Adjusted and normalized MI wasn't working properly in TreeSearch.
4. #1423: Value error in bayesian estimation.
5. Fixes bug in `DiscreteFactor.__eq__` to also consider the state names order.

## [0.1.14] - 2021-03-31
### Added
1. Adds support for python 3.9.
2. `BayesianModelProbability` class for calculating pmf for BNs.
3. BayesianModel.predict has a new argument `stochastic` which returns stochastic results instead of MAP.
4. Adds new method pgmpy.base.DAG.to_daft to easily convert models into publishable plots.

### Changed
1. `pgmpy.utils.get_example_model` now doesn't need internet connection to work. Files moved locally.

### Fixed
1. Latex output of `pgmpy.DAG.get_independencies`.
2. Bug fix in PC algorithm as it was skipping some combinations.
3. Error in sampling because of seed not correctly set.

## [0.1.13] - 2020-12-30
### Added
1. New conditional independence tests for discrete variables

### Changed
1. Adds warning in BayesianEstimator when using dirichlet prior.

### Fixed
1. Bug in `PC.skeleton_to_pdag`.
2. Bug in `HillClimbSearch` when no legal operations.

### Removed

## [0.1.12] - 2020-09-30
### Added
1. PC estimator with original, stable, and parallel variants.
2. PDAG class to represent partially directed DAGs.
3. `pgmpy.utils.get_example_model` function to fetch models from bnlearn repository.
4. Refactor HillClimbSearch with a new feature to specify fixed edges in the model.
5. Adds a global `SHOW_PROGRESS` variable.
6. Adds Chow-Liu structure learning algorithm.
7. Add `pgmpy.utils.get_example_model` to fetch models from bnlearn's repository.
8. Adds `get_value` and `set_value` method to `DiscreteFactor` to get/set a single value.
9. Adds `get_acestral_graph` to `DAG`.

### Changed
1. Refactors ConstraintBasedEstimators into PC with a lot of general improvements.
2. Improved (faster, new arguments) indepenedence tests with changes in argument.
3. Refactors `sample_discrete` method. Sampling algorithms much faster.
4. Refactors `HillClimbSearch` to be faster.
5. Sampling methods now return dataframe of type categorical.

### Fixed

### Removed
1. `Data` class.

## [0.1.11] - 2020-06-30
### Added
- New example notebook: Alarm.ipynb
- Support for python 3.8
- Score Caching support for scoring methods.

### Changed
- Code quality check moved to codacy from landscape
- Additional parameter `max_ci_vars` for `ConstraintBasedEstimator`.
- Additional parameter `pseudo_count` for K2 score.
- Sampling methods return state names instead of number when available.
- XMLBIFReader and BIFReader not accepts argument for specifying state name type.

### Fixed
- Additional checks for TabularCPD values shape.
- `DiscreteFactor.reduce` accepts both state names and state numbers for variables.
- `BeliefPropagation.query` fixed to return normalized CPDs.
- Bug in flip operation in `HillClimbSearch`.
- BIFWriter to write the state names to file if available.
- `BayesianModel.to_markov_model` fixed to work with disconnected graphs.
- VariableElimination fixed to not ignore identifical factors.
- Fixes automatic sorting of state names in estimators.

### Removed
- No support for ProbModelXML file format.

## [0.1.10] - 2020-01-22
### Added
- Documentation updated to include Structural Equation Models(SEM) and Causal Inference.
- Adds Mmhc estimator.

### Changed
- BdeuScore is renamed to BDeuScore.
- Refactoring of NaiveBayes
- Overhaul of CI and setup infrastructure.
- query methods check for common variabls in variable and evidence argument.

### Fixed
- Example notebooks for Inference.
- DAG.moralize gives consistent results for disconnected graphs.
- Fixes problems with XMLBIF and BIF reader and writer classes to be consistent.
- Better integration of state names throughout the package.
- Improves remove_factors and add_factors methods of FactorGraph
- copy method of TabularCPD and DiscreteFactor now makes a copy of state names.

### Removed
- six not a dependency anymore.
