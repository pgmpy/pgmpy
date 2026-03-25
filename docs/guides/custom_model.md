# Defining a Custom Model

```{meta}
:description: Define custom graphical models and CPD types in pgmpy to build Bayesian and Markov models.
```

pgmpy lets you define different graphical model families and their
corresponding factor or CPD types.

For a DAG model, the joint distribution factorizes as a product of local
conditionals over parents:

```{math}
P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid Pa_i)
```

## When to use

- Use a custom model definition when you want full control over graph
  structure, variable types, and parameterization.
- Start with a discrete Bayesian network when working with tabular CPDs.
- Choose linear Gaussian or functional Bayesian networks when the data or
  model family requires continuous or hybrid CPDs.
- Use factor-graph or Markov-network families when the problem is naturally
  undirected.

## Example

```python
from pgmpy.datasets import load_dataset
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

dataset = load_dataset("sachs_discrete")
data = dataset.data
model = DiscreteBayesianNetwork([("PKA", "ERK"), ("ERK", "Akt")])
fitted = model.fit(data, estimator=MaximumLikelihoodEstimator)
print(fitted.get_cpds())
```

## Model Types

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Model
     - CPD Type
     - API Reference
   * - Bayesian Network (Discrete)
     - TabularCPD
     - :class:`~pgmpy.models.BayesianNetwork.BayesianNetwork`
   * - Linear Gaussian BN
     - LinearGaussianCPD
     - :class:`~pgmpy.models.LinearGaussianBayesianNetwork.LinearGaussianBayesianNetwork`
   * - Functional BN
     - FunctionalCPD
     - :class:`~pgmpy.models.FunctionalBayesianNetwork.FunctionalBayesianNetwork`
   * - Dynamic BN
     - TabularCPD
     - :class:`~pgmpy.models.DynamicBayesianNetwork.DynamicBayesianNetwork`
   * - Naive Bayes
     - TabularCPD
     - :class:`~pgmpy.models.NaiveBayes.NaiveBayes`
   * - Markov Network
     - DiscreteFactor
     - :class:`~pgmpy.models.MarkovNetwork.MarkovNetwork`
   * - Factor Graph
     - DiscreteFactor
     - :class:`~pgmpy.models.FactorGraph.FactorGraph`
   * - Junction Tree
     - DiscreteFactor
     - :class:`~pgmpy.models.JunctionTree.JunctionTree`
   * - Cluster Graph
     - DiscreteFactor
     - :class:`~pgmpy.models.ClusterGraph.ClusterGraph`
   * - Structural Equation Model
     - --
     - :class:`~pgmpy.models.SEM.SEM`
   * - Markov Chain
     - --
     - :class:`~pgmpy.models.MarkovChain.MarkovChain`
```

## Factor / CPD Types

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Factor
     - API Reference
   * - TabularCPD
     - :class:`~pgmpy.factors.discrete.CPD.TabularCPD`
   * - DiscreteFactor
     - :class:`~pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor`
   * - NoisyOrCPD
     - :class:`~pgmpy.factors.discrete.NoisyOR.NoisyOrCPD`
   * - LinearGaussianCPD
     - :class:`~pgmpy.factors.continuous.LinearGaussianCPD.LinearGaussianCPD`
   * - FunctionalCPD
     - :class:`~pgmpy.factors.hybrid.FunctionalCPD.FunctionalCPD`
```

## See Also

- **Examples:** {doc}`Creating a Discrete BN <../examples/Creating_Discrete_BN>` | {doc}`Creating a Linear BN <../examples/Creating_Linear_BN>` | {doc}`Dynamic BN <../examples/Dynamic_BN>` | {doc}`Defining CPDs <../examples/Defining_CPDs>`
- **API Reference:** {doc}`Models <../api/models>` | {doc}`Factors <../api/factors>`
- **Previous:** {doc}`io` -- import and export models in various formats
- **Next:** {doc}`causal_discovery` -- learn graph structure from data
