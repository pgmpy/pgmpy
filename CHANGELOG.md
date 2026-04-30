# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-04-30
### Added
1. Prioritization option for PC orientation to control rule application order (#3363).
2. Effect sizes for CI tests (#3358).
3. Caching support to CI tests for faster repeated queries (#3349).
4. New CI tests (#3344).
5. Estimator argument to Pillai Trace CI test (#3072).
6. `DAG.get_random` can now generate graphs with a fixed number of edges (#3086).
7. Causal stats in `DAG.get_stats()` method (#3143).
8. Exposes estimators as attributes in residual based CI tests (#3367).

### Changed
1. Optimizes `power_divergence` test and PC algorithm for better performance (#3356).
2. Refactors parameter estimation methods into a dedicated `pgmpy.parameter_estimator` package (#3325).
3. Refactors `SHD` to accept covariant arguments in `__init__` for composability (#3310).
4. Improves deprecation warning messages (#3343).
5. Uses combinations instead of permutations in `_orient_colliders` for better performance (#3195).
6. Revamps documentation website (#2615).
7. Reject negative probability values in `is_valid_cpd` (#3052).

### Fixed
1. Fixes bug in collider orientation for PC (#3362).
2. Fixes edge orientation in GES (#3304).
3. Fixes GES forward/backward/turning phases to be deterministic across hash seeds (#3303).
4. Fixes `__eq__` to check structure equality in `LinearGaussianBayesianNetwork` (#3276).
5. Fixes incorrect `has_missing_data` tag for tubingen dataset (#3231).
6. Fixes minor typo in ValueError messages in `state_name.py` (#3309).
7. Fixes failing doctests in `BIF`, `NET`, `XMLBIF`, and `UAI` readwrite modules (#3226, #3230, #3262, #3263).

## [1.1.0] - 2026-04-01
### Added
1. New role-aware causal graph infrastructure, including `PDAG`, `ADMG`, `MAG`, `AncestralBase`, and `SimpleCausalModel`.
2. New graph utilities in `DAG`, including `to_pdag`, `to_dagitty`, `to_lavaan`, public `get_ancestors`, `edge_strength`, `get_stats`, and `__hash__`.
3. `DAG.to_daft` and `DAG.to_graphviz` can now annotate plots with computed edge strengths.
4. `DAG.from_dagitty` can now construct `LinearGaussianBayesianNetwork` instances when the dagitty model includes beta coefficients.
5. New sklearn-compatible causal discovery package `pgmpy.causal_discovery` with refactored `PC`, `GES`, `HillClimbSearch`, and `ExpertInLoop` estimators.
6. `ExpertKnowledge` now supports `search_space`, richer string representations, temporal-order handling, and tighter integration with causal discovery and `ExpertInLoop`.
7. New causal effect APIs built around `pgmpy.identification`, including shared identification base classes and adjustment/frontdoor workflows.
8. New `pgmpy.prediction` module with sklearn-compatible causal prediction estimators: `NaiveAdjustmentRegressor`, `DoubleMLRegressor`, and `NaiveIVRegressor`.
9. New `pgmpy.ci_tests` package with class-based CI tests and registry-based lookup, including `FisherZ`, `PearsonrEquivalence`, and estimator-configurable `GCM`.
10. New `pgmpy.structure_score` package and new causal graph evaluation metrics: `AdjacencyConfusionMatrix` and `OrientationConfusionMatrix`.
11. `LinearGaussianBayesianNetwork` now supports JSON `load`/`save`, `log_likelihood`, `predict_probability`, improved `predict`, and richer simulation features for interventions, evidence, virtual interventions, and missing-data generation.
12. `FunctionalBayesianNetwork` sampling now supports vectorized `FunctionalCPD` functions and interventional simulation.
13. `ExpectationMaximization` gained `apply_smoothing`, configurable `init_cpds`, and support for node-specific priors.
14. `DynamicBayesianNetwork.simulate` gained a `return_format` argument.
15. New dataset and example-model registries, `pgmpy.datasets` and `pgmpy.example_models`, with filterable catalogs and on-demand loading.
16. Many new benchmark datasets were added, including Boston Housing, Wine Quality, Galton Stature, Seoul Bike, Pima Diabetes, Student Performance, South German Credit, Cystic Fibrosis, Yacht Hydrodynamics, HTRU2, Myocardial Infarction, IQ Brain Size, Contraceptive Method, Pittsburgh Bridges, Residential Building, Hitters, Superconductivity, Lead, Cities, College Plans, Depression Coping, US Crime, Hungary chickenpox, the Angrist-Krueger 1980 census extract, dropout covariance data, and the Tuebingen pairs.
17. Many new example models were added through `bnlearn`, `bnrep`, and DAG-only registries, including Andes, Barley, Child, Hailfinder, Hepar2, Insurance, Link, Munin 1-4, Pathfinder, Pigs, Sachs, Survey, Water, and continuous models such as `magic_irri`.
18. `DiscreteBayesianNetwork` / readwrite support now includes NET/HUGIN files, improved XDSL support, and a JSON schema for Linear Gaussian model serialization.

### Changed
1. `PC`, `GES`, `HillClimbSearch`, and `ExpertInLoop` were refactored into sklearn-style causal discovery estimators with `fit`, learned `causal_graph_`, and automatic CI-test / structure-score selection based on the data type.
2. Graph role names are now standardized to plural keys such as `exposures`, `outcomes`, and `confounders` across graph, identification, and prediction APIs.
3. `ExpertKnowledge` now lives under `pgmpy.causal_discovery`, and structure scores are organized under `pgmpy.structure_score`.
4. Example datasets and example models are now loaded through Hugging Face-backed registries instead of being bundled directly in the repository.
5. `MarkovNetwork` is now deprecated in favor of `DiscreteMarkovNetwork`.
6. `DAG.fit` moved to `DiscreteBayesianNetwork.fit`, and Bayesian-network simulation argument ordering was made more consistent across model classes.
7. Causal discovery estimators can now score themselves, and structure scoring defaults are more automatic and BIC-oriented.
8. CI-test internals were refactored toward registry/class-based implementations, and `scikit-base` is now used for object lookup in datasets, metrics, example models, and CI tests.
9. Optional dependencies such as `torch` / `pyro`, Graphviz, documentation tooling, and LLM integrations are isolated more cleanly as extras.
10. Packaging migrated from `setup.py` to `pyproject.toml`, and `pyparsing` is now an explicit dependency for read/write features.
11. `list_models` and `list_datasets` now validate filter tags explicitly and expose richer metadata such as `has_missing_data` and `has_index_col`.
12. `BIF`, `XMLBIF`, `NET`, and `XDSL` writers now warn when state names contain commas, and `BIFWriter` no longer silently rewrites those state names.

### Fixed
1. Fixes correctness issues in `GES`, including the latest algorithmic fix in `#3228`.
2. Fixes order-independence issues in the stable variant of `PC`, along with earlier bugs around `max_cond_vars` propagation and skeleton building.
3. Fixes role validation and pretreatment handling in causal prediction / DML workflows, plus multiple exposure/outcome naming inconsistencies in identification and examples.
4. Fixes random model generation edge cases in `LinearGaussianBayesianNetwork`, including handling of isolated latent nodes and `seed=0` in `get_random_cpds`.
5. Fixes `LinearGaussianBayesianNetwork.predict`, `predict_probability`, invalid virtual evidence errors, unbiased standard deviation estimation, and several simulation / documentation mismatches.
6. Fixes inference and sampling issues in `FunctionalBayesianNetwork`, `ApproxInference`, and `DBNInference`.
7. Improves `ExpectationMaximization` handling of missing data by treating all-missing columns as latent variables, dropping partially missing rows explicitly, and adding clearer validation.
8. Improves BIF/XDSL/NET/XMLBIF read-write robustness, including faster and more reliable BIF parsing, better node/state-name handling, and cleaner warnings for punctuation in state names.
9. Fixes `SHD` on graphs with isolated nodes, `CorrelationScore` on very small graphs, `MarkovChain.add_transition_model` validation, `FactorGraph.add_edge` defaults, and Bayesian-estimator model reconstruction with isolated nodes.
10. Fixes compatibility issues with newer dependency and platform versions, including pandas 3 string dtype changes, newer xgboost, Python 3.14, and macOS / Windows CI environments.
11. Many notebooks, docstrings, and usage examples across causal discovery, inference, factors, and model APIs were corrected to match the current behavior.

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
3. Bug in EM when latent variables are present.
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
3. Adds `DynamicBayesianNetwork.fit` method to learn model parameters from data.
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
1. Adds network pruning for inference algorithms to reduce the size of network before
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
2. Improved (faster, new arguments) independence tests with changes in argument.
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
- query methods check for common variables in variable and evidence argument.

### Fixed
- Example notebooks for Inference.
- DAG.moralize gives consistent results for disconnected graphs.
- Fixes problems with XMLBIF and BIF reader and writer classes to be consistent.
- Better integration of state names throughout the package.
- Improves remove_factors and add_factors methods of FactorGraph
- copy method of TabularCPD and DiscreteFactor now makes a copy of state names.

### Removed
- six not a dependency anymore.
