# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
