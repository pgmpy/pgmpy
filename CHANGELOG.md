# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
1. Uniform warning behavior for commas in state names across file writers (BIF, XMLBIF, NET, XDSL).
   - Writers now issue a warning when state names contain commas, which can cause issues when loading the file.
   - The warning helps users identify potentially problematic state names.
   - This affects `BIFWriter`, `XMLBIFWriter`, `NETWriter`, and `XDSLWriter` classes.
   - Note: While XDSL format can handle commas in state names, a warning is still issued for consistency.

### Changed
1. `BIFWriter.get_states` no longer silently replaces commas with underscores in state names.
2. `XMLBIFWriter._make_valid_state_name` now issues a warning when it encounters commas in state names.
3. `NETWriter.get_states` now issues a warning when it encounters commas in state names.
4. `XDSLWriter.get_cpds` now issues a warning when it encounters commas in state names.

### Fixed
1. Improved consistency in how different file writers handle state names with special characters.

## [1.0.0] - 2024-03-31
### Added
1. Option to specify the node names in random model generation methods.
2. ExpertInLoop.estimate method now accepts any LLM model supported by litellm.
3. AIC and BIC structure scoring methods for Gaussian variables and Conditional Gaussian variables.
4. Greedy Equivalence Search algorithm for causal discovery.
5. Causal Discovery algorithms can automatically figure out the data types of the variables.
6. Support for continuous and mixed data types for all causal discovery / structure learning algorithms.
7. Adds LinearGaussianBayesianNetwork and FunctionalBayesianNetwork classes to represent Gaussian and hybrid Bayesian Networks.
8. Add `pgmpy.metrics.SHD` to compute the Structural Hamming Distance between two DAGs.
9. Adds `NoisyORCPD` class to represent NoisyOr models.
10. BayesianNetwork.simulate method can not simulate different types of missing data.
11. BayesianNetwork.predict now predicts any missing value in the dataframe instead of missing columns.
12. Adds continuous example models from bnlearn repository.
13. Adds `ExpertKnowledge` class to specify expert knowledge for structure learning algorithms.
14. Option to initialize DAG and DiscreteBayesianNetwork with adjacency matrix, a dagitty model, or a lavaan model string.
15. Adds method for reading and writing XDSL file format (used by GeNIe).
16. Adds Generalized Covariance Measure (GCM) conditional independence test.

### Fixed
1. Fixes bug in `pgmpy.estimators.SEMEstimator`.

### Changed
1. Renames `pgmpy.estimators.CITests.ci_pillai` to `pgmpy.estimators.CITests.pillai_trace`.
2. `BayesianNetwork.fit` method moved to `DAG.fit` so that fitting can be done on either model types.
3. All structure scoring methods have been renamed to simplify.

### Removed
1. Removes some of the CI test variants of chi-squared test.
2. Removes `pgmpy.factors.continuous.ContinuousFactor` class.
2. Removes discretization methods for ContinuousFactor.
3. `BayesianModel` and `MarkovModel` classes have been removed.
4. `BayesianNetwork` class have been removed. Use `DiscreteBayesianNetwork` instead.

## [0.1.26] - 2024-08-09
### Added
1. Support for returning Belief Propagation messages in Factor Graph BP.
2. Maximum Likelihood Estimator for Junction Tree.
3. Adds a simple discretization method: `pgmpy.utils.discretize`.
4. Two new metrics for model testing: `pgmpy.metrics.implied_cis` and `pgmpy.metrics.fisher_c`.
5. Support for Linear Gaussian Bayesian Networks: estimation, prediction, simulation and random model generation.
7. New mixed data Conditional Independence test based on canonical correlations.
8. New LLM based structure learning / causal discovery algorithm. Also LLM based pairwise variable orientation method.


### Fixed
1. Reading and Writing from XBN file format.
2. Documentation for plotting models.
3. Fixes PC algorithm to add disconnected nodes in the final model.
4. Allows `.` in variables names in BIF file format.

### Changed
1. Allows `virtual_evidence` parameter in inference methods to accept DiscreteFactor objects.

## [0.1.25] - 2024-03-08
### Added
1. `init_cpds` argument to `ExpecattionMaximiation.get_parameters` to specify initialization values.
2. BeliefPropagation with message passing for Factor Graphs.
3. Marginal Inference for undirected graphs.

### Fixed
1. Incompatibality with networkx==3.2.
2. `CausalInference.get_minimal_adjustment_set` to accept string variable names.
3. Bug in EM when latent varaibles are present.
4. `compat_fns.copy` to consider the case when int or float is passed.
5. Fixes issue with `BayesianNetwork.fit_update` when running with CUDA backend.

### Changed
1. Documentation Updates
2. Optimizations for Hill Climb Search algorithm.
3. Tests shutdown parallel workers in teardown.
4. Removes the `complete_samples_only` argument from `BaseEstimator.state_counts`.
5. Default number of cores to use changed to 1 for parameter estimation methods.

## [0.1.24] - 2023-06-30
### Added
1. Added support for python 3.11.
2. Adds `DAG.to_graphviz` and `PDAG.to_graphviz` methods to convert model to graphviz objects.
3. Adds pytorch as an alternative backend.
4. Adds unicode support for BIFReader.

### Fixed
1. Warnings use a logger instance.
2. Fixes documentation.
3. Fixes variables name arguments for `CausalInference.get_minimal_adjustment_set`

### Changed
1. Adds argument to specify samples for ApproxInference.
2. Memory optimizations for computing structure scores.
3. Switches joblib backed to loky.
4. Runtime optimizations for sampling.
5. Runtime optimizations for Variable Elimination.
6. All config variables moved to `pgmpy.global_vars`.

## [0.1.23] - 2023-06-30
### Added
1. BIFReader made compatible with the output of PyAgrum
2. Support for all available CI tests in PC algorithm.
3. References for read/write file formats.

### Removed
1. Removes `DAG.to_pdag` method.

### Changed
1. Fixes for ApproxInference for DBNs.
2. Make `xml.etree` the default parser instead of using lxml.

## [0.1.22] - 2023-04-08
### Added
1. AIC score metric from score based structure learning.
2. Adds support for NET (HUGIN) file format.
3. Adds argument reindex to `state_counts` method.

### Fixed
1. Bug in GibbsSampling when sampling from Bayesian Networks.
2. Fix seed for all simulation methods.
3. Memory leaks when using `lru_cache`.

### Changed
1. Caching disabled for computing state name counts during structure learning.
2. Pre-computation for sampling methods are optimized.

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
