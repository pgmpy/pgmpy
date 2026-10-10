# PCMCI Architecture Design for pgmpy

## 1. Purpose

This document describes the proposed architecture for implementing the PCMCI causal discovery algorithm in pgmpy.

The design has three important goals:

1. Integrate PCMCI with pgmpy's existing `BaseCausalDiscovery` estimator interface.
2. Keep the implementation faithful to the conceptual structure of PCMCI.
3. Separate causal discovery from temporal graph representation so that PCMCI does not become coupled to the implementation of `TimeSeriesDAG`.

The proposed architecture is:

```text
                    BaseCausalDiscovery
                            │
                            │ fit(X)
                            ▼
                          PCMCI
                            │
                            │ uses
                            ▼
                       _BasePCMCI
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
          PC phase       MCI phase    Statistical
                                      processing
              │             │             │
              └─────────────┼─────────────┘
                            │
                            ▼
                     PCMCI results
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
          p_matrix      val_matrix      graph/
                                        links
                            │
                            ▼
                 TimeSeriesDAG conversion
                       (separate layer)
                            │
                            ▼
                       TimeSeriesDAG
```

The central design decision is that **PCMCI produces causal-discovery results; it does not construct the `TimeSeriesDAG` itself**.

The conversion from the PCMCI result to `TimeSeriesDAG` is intentionally kept as a separate concern.

---

# 2. Existing pgmpy integration point

pgmpy already provides `BaseCausalDiscovery`, which handles generic estimator concerns such as:

* input validation,
* pandas DataFrame handling,
* feature names,
* sample/feature metadata,
* sklearn compatibility,
* the public `fit()` interface.

The relevant conceptual flow is:

```python
model.fit(X)
```

which eventually performs:

```python
X = self._check_fit_data(X)
return self._fit(X)
```

Therefore, PCMCI does not need to redefine the generic fitting process.

Instead:

```python
class PCMCI(BaseCausalDiscovery, _BasePCMCI):

    def _fit(self, X):
        ...
```

is sufficient.

This is important because it keeps the responsibilities of `BaseCausalDiscovery` intact.

`BaseCausalDiscovery` answers:

> "How does a causal discovery estimator behave as a pgmpy/sklearn estimator?"

PCMCI answers:

> "How do we perform PCMCI causal discovery on the validated data?"

These are different responsibilities.

---

# 3. Why introduce `_BasePCMCI`?

PCMCI is not a single statistical operation.

It is a multi-stage algorithm involving:

1. candidate parent construction,
2. conditional independence testing,
3. iterative condition-selection,
4. MCI testing,
5. p-value processing,
6. temporal link interpretation,
7. final discovery-result construction.

If all of this were placed directly inside:

```python
class PCMCI(BaseCausalDiscovery):
```

the public estimator would become responsible for both:

* pgmpy estimator behavior, and
* the entire PCMCI implementation.

That creates a large and difficult-to-maintain class.

Instead, `_BasePCMCI` contains the **algorithmic machinery shared by PCMCI-family algorithms**.

Conceptually:

```text
BaseCausalDiscovery
    generic estimator behavior

_BasePCMCI
    PCMCI-family algorithm behavior

PCMCI
    public PCMCI estimator
```

This also leaves room for future algorithms such as PCMCI+ to reuse genuinely common infrastructure without forcing them to duplicate everything.

---

# 4. Responsibility of `_BasePCMCI`

`_BasePCMCI` is responsible for the internal mechanics of PCMCI.

Its responsibilities include:

### Data preparation

```python
_prepare_data()
```

This converts the already validated pgmpy DataFrame into whatever representation the internal algorithm requires.

The public API remains:

```python
X: pandas.DataFrame
```

with rows representing time points and columns representing variables.

This keeps pgmpy's public API simple and avoids exposing Tigramite-specific data structures.

---

### Conditional independence testing

```python
_configure_conditional_independence_test()
_run_ci_test()
```

PCMCI fundamentally depends on conditional independence tests.

Abstractly:

```text
X ⟂ Y | Z
```

is tested using a conditional independence test.

The algorithm should not need to know the internal details of the specific test.

Instead, it should have an interface conceptually equivalent to:

```python
val, p_value = self._run_ci_test(...)
```

This gives the algorithm two important quantities:

* `val`: the test statistic / dependency measure,
* `p_value`: evidence against conditional independence.

This separation is theoretically important because **PCMCI is a causal discovery procedure built around conditional independence testing**, rather than being itself a single particular statistical test.

---

# 5. Temporal structure

Ordinary causal discovery often works with relationships such as:

```text
X → Y
```

PCMCI works with time-indexed relationships:

```text
X(t-1) → Y(t)
```

or:

```text
X(t-2) → Y(t)
```

Therefore the natural internal representation needs to preserve:

```text
source
target
lag
```

A useful conceptual representation is:

```python
(source, target, lag)
```

For example:

```python
("temperature", "sales", -1)
```

could represent:

```text
temperature(t-1) → sales(t)
```

The exact lag convention can be decided in the implementation/API, but the important point is that **lag is a first-class part of the causal relationship**.

This is why the statistical results should not initially be reduced to an ordinary 2D adjacency matrix.

---

# 6. Why the p-value and statistic matrices are temporal

The proposed result is conceptually:

```python
p_matrix[target, source, lag]
val_matrix[target, source, lag]
```

For example:

```python
p_matrix[Y, X, 1]
```

represents the p-value associated with:

```text
X(t-1) → Y(t)
```

Similarly:

```python
val_matrix[Y, X, 1]
```

contains the corresponding test statistic/dependency value.

This representation is structurally appropriate because PCMCI evaluates **specific lagged relationships**, rather than merely evaluating whether two variables are connected.

A normal 2D matrix:

```text
         X    Y    Z
X
Y
Z
```

would lose information about whether:

```text
X(t-1) → Y(t)
```

or:

```text
X(t-2) → Y(t)
```

was discovered.

Therefore a temporal representation is required.

---

# 7. Link assumptions

The `_BasePCMCI` layer also handles:

```python
_set_link_assumptions()
_build_default_link_assumptions()
_validate_link_assumptions()
_link_is_allowed()
```

The purpose is to define which relationships PCMCI is allowed to consider.

For example:

```text
X(t-1) → Y(t)
X(t-2) → Y(t)
Z(t-1) → Y(t)
```

may be allowed while other source-target-lag combinations may be excluded.

This is not merely an implementation convenience.

PCMCI operates over a space of candidate temporal relationships. Link assumptions define that search space.

Therefore the logical sequence is:

```text
Temporal variables
       │
       ▼
Allowed source-target-lag relationships
       │
       ▼
Candidate parents
       │
       ▼
Conditional independence testing
       │
       ▼
Discovered relationships
```

Keeping this logic inside `_BasePCMCI` is therefore structurally appropriate.

---

# 8. Candidate initialization

The next responsibility is:

```python
_initialize_candidates()
```

Suppose we have variables:

```text
X, Y, Z
```

and:

```text
tau_max = 2
```

The candidate relationships for `Y(t)` could include:

```text
X(t-1) → Y(t)
X(t-2) → Y(t)

Y(t-1) → Y(t)
Y(t-2) → Y(t)

Z(t-1) → Y(t)
Z(t-2) → Y(t)
```

At this stage, these are only **candidate relationships**.

They are not yet causal discoveries.

This distinction is important.

PCMCI does not initially say:

> "Every candidate is a causal edge."

Instead:

```text
Candidate relationship
        │
        ▼
Conditional independence testing
        │
        ▼
Remove relationships inconsistent
with the observed conditional dependencies
        │
        ▼
Remaining candidates
```

This is the role of the PC/condition-selection phase.

---

# 9. PC / condition-selection phase

The method:

```python
_run_pc_stable()
```

represents the first major algorithmic stage.

Its purpose is to identify a set of plausible parents for each target variable.

Conceptually, it repeatedly asks questions such as:

```text
Are X(t-1) and Y(t)
conditionally independent given Z?
```

or:

```text
Are X(t-1) and Y(t)
conditionally independent given Z(t-1), W(t-2), ...?
```

If the conditional independence test indicates independence, the candidate relationship can be removed.

The important theoretical idea is:

> A candidate relationship should not remain merely because X and Y are marginally associated.

It must survive appropriate conditional independence testing.

This is one of the reasons the PC phase is separated from the MCI phase.

---

# 10. MCI phase

After candidate selection, PCMCI performs Momentary Conditional Independence (MCI) tests.

The corresponding method is:

```python
_run_mci_tests()
```

At this stage the algorithm evaluates the remaining candidate relationships using the appropriate conditioning sets.

The conceptual result is:

```text
candidate temporal relationship
            │
            ▼
       MCI test
            │
       ┌────┴────┐
       ▼         ▼
   statistic    p-value
```

These values are stored in:

```python
val_matrix
p_matrix
```

The distinction between the PC phase and MCI phase is therefore reflected directly in the software structure.

---

# 11. Why p-values and test statistics are separate

The implementation should preserve both:

```python
p_matrix_
val_matrix_
```

rather than only storing a boolean graph.

A boolean representation would tell us:

```text
X(t-1) → Y(t) = True
```

but would discard statistical information.

The p-value tells us about the statistical evidence associated with the tested relationship.

The test statistic/dependency value provides additional information about the result of the conditional independence test.

Therefore:

```text
Raw statistical result
        │
        ├── p-value
        └── test statistic
```

should be retained before converting the result into a graph representation.

This is also useful for future functionality such as:

* significance threshold changes,
* multiple-testing correction,
* diagnostics,
* result inspection,
* scientific analysis,
* comparison between CI tests.

---

# 12. Multiple-testing correction

PCMCI can perform many conditional independence tests.

If there are:

```text
N variables
×
N possible source variables
×
T possible lags
```

then the number of statistical tests can become large.

Consequently, multiple-testing correction becomes relevant.

The proposed structure therefore contains:

```python
_correct_pvalues()
_fdr_bh()
```

The conceptual flow is:

```text
Raw p-values
     │
     ▼
Multiple-testing correction
     │
     ▼
Corrected p-values
     │
     ▼
Determine significant relationships
```

Benjamini-Hochberg/FDR correction is treated as a statistical processing step, not as part of graph construction.

That distinction matters because p-value correction is a property of the statistical inference process, whereas `TimeSeriesDAG` is a representation of the resulting causal structure.

---

# 13. Why PCMCI should not construct `TimeSeriesDAG`

This is one of the most important architectural decisions.

It may initially seem natural to write:

```python
pcmci.causal_graph_ = TimeSeriesDAG(...)
```

inside `PCMCI`.

However, the agreed design instead has PCMCI produce its discovery results and lets the `TimeSeriesDAG` layer perform the conversion.

The reason is separation of concerns.

PCMCI answers:

> "Which temporal relationships are supported by the conditional independence analysis?"

`TimeSeriesDAG` answers:

> "How should a temporal causal graph be represented and manipulated in pgmpy?"

These are different questions.

Therefore:

```text
PCMCI
  =
causal discovery algorithm
```

while:

```text
TimeSeriesDAG
  =
temporal graph representation
```

Keeping them separate means PCMCI does not need to know the internal design of `TimeSeriesDAG`.

---

# 14. Why this is especially useful in pgmpy

The other contributor is independently developing `TimeSeriesDAG`.

If PCMCI directly depended on the implementation details of that class, the two pieces of work would become tightly coupled.

For example, PCMCI might start depending on:

```python
TimeSeriesDAG.add_edge(...)
TimeSeriesDAG.from_pcmci(...)
TimeSeriesDAG.some_internal_method(...)
```

Then changes to `TimeSeriesDAG` could force changes to PCMCI.

Instead, the boundary becomes:

```text
PCMCI
  │
  │ produces agreed result format
  ▼
Conversion layer
  │
  ▼
TimeSeriesDAG
```

The conversion layer becomes the integration boundary.

Therefore both contributors can develop independently.

---

# 15. Why `TimeSeriesDAG.from_pcmci()` is not necessary

A method such as:

```python
TimeSeriesDAG.from_pcmci(...)
```

would make `TimeSeriesDAG` explicitly aware of PCMCI.

That creates an undesirable dependency:

```text
TimeSeriesDAG
      │
      └── knows about PCMCI
```

Instead, the desired dependency is:

```text
PCMCI result
      │
      ▼
conversion layer
      │
      ▼
TimeSeriesDAG
```

The graph abstraction should not need to know whether its edges came from:

```text
PCMCI
PCMCI+
DYNOTEARS
Granger causality
manual specification
another causal discovery algorithm
```

This makes `TimeSeriesDAG` algorithm-agnostic.

That is an important architectural property.

---

# 16. The role of `graph_`

There is one nuance here.

The draft API contains:

```python
self.graph_ = result["graph"]
```

This does **not** mean that `graph_` has to be a `TimeSeriesDAG`.

It can simply be PCMCI's internal discovery representation.

For example:

```text
graph_[target, source, lag]
```

could indicate whether a relationship was retained after the relevant statistical procedure.

The final conversion layer can then use:

```text
p_matrix_
val_matrix_
graph_
var_names_
lag information
```

to construct the `TimeSeriesDAG`.

The exact contract can be decided later.

---

# 17. Why the conversion layer should consume results rather than algorithm internals

The conversion layer should not need to understand:

```python
_run_pc_stable()
_run_mci_tests()
_build_condition_set()
_run_ci_test()
```

Those are implementation details of PCMCI.

It should only need the final result.

This gives a clean abstraction boundary:

```text
                 PCMCI internals
                       │
             ┌─────────┴─────────┐
             │                   │
          PC phase             MCI
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                 Result contract
                       │
                       ▼
               Conversion layer
                       │
                       ▼
                  TimeSeriesDAG
```

If the internal implementation of PCMCI changes later, the conversion layer does not necessarily need to change as long as the result contract remains stable.

---

# 18. Why `PCMCI` itself should be small

The public class is intentionally thin:

```python
class PCMCI(BaseCausalDiscovery, _BasePCMCI):

    def _fit(self, X):
        ...
```

This is desirable because `PCMCI` is the user-facing estimator.

Its job is essentially orchestration:

```text
validate data
     ↓
prepare PCMCI
     ↓
run algorithm
     ↓
store results
```

The complicated algorithmic details live in `_BasePCMCI`.

This makes the public API easier to understand.

A user should be able to read:

```python
pcmci = PCMCI(...)
pcmci.fit(data)
```

without needing to understand the implementation of every PCMCI phase.

---

# 19. Relationship to Tigramite's PCMCIbase

The proposed `_BasePCMCI` is conceptually inspired by the responsibilities present in Tigramite's `PCMCIbase`.

The Tigramite implementation contains functionality for:

* link assumptions,
* temporal graph conversion,
* p-value correction,
* adjacency extraction,
* graph representations,
* bootstrap/sliding-window analysis,
* PCMCI/PCMCI+ related graph processing.

For example, the source contains `_set_link_assumptions`, `_dict_to_matrix`, `get_corrected_pvalues`, temporal adjacency helpers, and graph conversion utilities.
However, pgmpy does not need to reproduce Tigramite's entire internal architecture.

The relevant question is:

> Which responsibilities belong to a PCMCI implementation in pgmpy?

rather than:

> Which methods existed in Tigramite and therefore must all be copied?

This distinction is particularly important because some Tigramite functionality exists to support Tigramite's own graph representation.

If pgmpy has a dedicated temporal graph abstraction, those responsibilities can instead belong to the temporal graph layer.

---

# 20. Methods intentionally not included initially

Several methods from Tigramite's `PCMCIbase` should not automatically be placed into `_BasePCMCI`.

### PCMCI+ CPDAG orientation

For example:

```python
_get_simplicial_node()
_get_dag_from_cpdag()
```

These are associated with PCMCI+/graph orientation logic rather than the generic core of PCMCI.

They should be introduced when the PCMCI+ implementation requires them rather than prematurely placing them in the base class.

---

### Tigramite-specific graph conversion

Methods such as:

```python
_dict_to_graph()
graph_to_dict()
get_graph_from_dict()
```

should not automatically be copied.

If their primary purpose is converting between Tigramite's internal graph representation and dictionaries, that representation does not necessarily need to become pgmpy's representation.

---

### Sliding-window and bootstrap utilities

Methods such as:

```python
run_sliding_window_of()
run_bootstrap_of()
return_summary_results()
```

are useful analysis functionality, but they are not necessary to establish the core PCMCI estimator.

They can be added later as:

* utilities,
* separate analysis functionality,
* mixins,
* or additional estimator methods.

The initial implementation should focus on getting the core algorithm and result contract correct.

---

# 21. Theoretical correctness of the separation

The architecture follows the conceptual stages of causal discovery.

PCMCI can be viewed as:

```text
Observed time series
        │
        ▼
Candidate temporal dependencies
        │
        ▼
Conditional independence testing
        │
        ▼
Candidate pruning / condition selection
        │
        ▼
MCI testing
        │
        ▼
Statistical inference
        │
        ▼
Temporal causal relationships
```

The software architecture mirrors this:

```text
_prepare_data()
        │
        ▼
_initialize_candidates()
        │
        ▼
_run_pc_stable()
        │
        ▼
_run_mci_tests()
        │
        ▼
_correct_pvalues()
        │
        ▼
_build_discovered_links()
        │
        ▼
PCMCI result
```

This is not an arbitrary decomposition.

The software methods correspond to conceptually distinct stages of the statistical procedure.

---

# 22. Separation between inference and representation

There are really two different transformations happening:

### Statistical inference

```text
Data
  ↓
CI tests
  ↓
p-values/statistics
  ↓
significant temporal relationships
```

### Graph representation

```text
Temporal relationships
  ↓
Temporal graph
  ↓
TimeSeriesDAG
```

These should not be conflated.

Therefore:

```text
PCMCI
  └── statistical inference

TimeSeriesDAG
  └── graph representation
```

This is arguably the most important structural principle behind the design.

---

# 23. Why this makes the implementation extensible

Suppose later pgmpy supports:

```text
PCMCI
PCMCI+
DYNOTEARS
Granger
other temporal causal discovery algorithms
```

If every algorithm directly constructs `TimeSeriesDAG`, every algorithm needs to know its API.

Instead, each algorithm can produce its own well-defined discovery result:

```text
PCMCI ───────┐
             │
PCMCI+ ──────┤
             ├──> temporal graph conversion
DYNOTEARS ───┤
             │
Granger ─────┘
```

This makes the temporal graph abstraction reusable across algorithms.

---

# 24. Proposed class responsibilities

| Component              | Responsibility                                          |
| ---------------------- | ------------------------------------------------------- |
| `BaseCausalDiscovery`  | Generic estimator behavior and input validation         |
| `_BasePCMCI`           | PCMCI-family algorithm infrastructure                   |
| `PCMCI`                | Public PCMCI estimator and orchestration                |
| CI-test implementation | Statistical conditional independence test               |
| Conversion layer       | Convert PCMCI result into temporal graph representation |
| `TimeSeriesDAG`        | Represent and manipulate temporal causal graphs         |

This division prevents responsibilities from leaking between layers.

---

# 25. Final proposed API shape

The user-facing API should remain simple:

```python
pcmci = PCMCI(
    cond_ind_test=cond_ind_test,
    tau_min=1,
    tau_max=3,
    pc_alpha=0.05,
)

pcmci.fit(data)
```

After fitting:

```python
pcmci.p_matrix_
pcmci.val_matrix_
pcmci.graph_
pcmci.var_names_
```

The exact names and shapes can be finalized later with the `TimeSeriesDAG` contributor.

The important architectural contract is:

```text
PCMCI
  ↓
produces temporal causal-discovery results
```

rather than:

```text
PCMCI
  ↓
constructs TimeSeriesDAG
```

The latter is intentionally outside PCMCI's responsibility.

---

# 26. Final architecture

The complete design can therefore be summarized as:

```text
                         pgmpy
                           │
                           ▼
                 BaseCausalDiscovery
                           │
                           │ generic fit()
                           ▼
                         PCMCI
                           │
                           │ delegates
                           ▼
                      _BasePCMCI
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Candidate         PC / condition     CI tests
     generation          selection
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                      MCI testing
                           │
                           ▼
                   p-value correction
                           │
                           ▼
                  Discovered links
                           │
                           ▼
                  PCMCI result object
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          p_matrix     val_matrix     graph
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                 Conversion layer
                           │
                           ▼
                      TimeSeriesDAG
```

## Architectural principle

The fundamental rule is:

> **PCMCI owns causal discovery. TimeSeriesDAG owns temporal graph representation. The conversion layer connects the two.**

This gives each component a single clear purpose, keeps the PCMCI implementation independent of the evolving `TimeSeriesDAG` API, preserves the temporal/statistical information required by PCMCI, and provides a natural extension point for other time-series causal discovery algorithms.

The exact matrix dimensions, lag convention, result dictionary, and conversion API should therefore be treated as an **interface contract to be finalized jointly**, rather than hard-coded into the first PCMCI implementation.
