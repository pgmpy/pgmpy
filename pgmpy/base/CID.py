from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from numbers import Integral, Real

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base.DAG import DAG
from pgmpy.parameterization._base import BaseParameter


class CausalInfluenceDiagram(DAG):
    """A directed acyclic graph with chance, decision, and utility nodes.

    Chance nodes have conditional probability parameterizations, decision nodes have policies, and utility nodes have
    functions of their parents. Nodes added without a role default to ``"chance"``.

    Parameters
    ----------
    ebunch : input graph, optional
        Data used to initialize the graph, as accepted by :class:`pgmpy.base.DAG`.
    chance_nodes : iterable of hashable, optional
        Nodes to assign the ``"chance"`` role.
    decision_nodes : iterable of hashable, optional
        Nodes to assign the ``"decision"`` role.
    utility_nodes : iterable of hashable, optional
        Nodes to assign the ``"utility"`` role. Utility nodes must be leaves.

    Examples
    --------
    >>> from pgmpy.base import CausalInfluenceDiagram
    >>> cid = CausalInfluenceDiagram(
    ...     [("Treatment", "Health"), ("Treatment", "Utility"), ("Health", "Utility")],
    ...     decision_nodes=["Treatment"],
    ...     utility_nodes=["Utility"],
    ... )
    >>> cid.decision_nodes, cid.utility_nodes
    (['Treatment'], ['Utility'])
    >>> cid.add_node("Age")
    >>> cid.chance_nodes
    ['Health', 'Age']

    Notes
    -----
    Utility nodes are evaluated during :meth:`simulate`; the sum of their values in each row is the total utility for
    that sample. Decision optimization and expected-utility inference are not implemented.
    """

    _NODE_ROLES = frozenset({"chance", "decision", "utility"})

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        chance_nodes: Iterable[Hashable] | None = None,
        decision_nodes: Iterable[Hashable] | None = None,
        utility_nodes: Iterable[Hashable] | None = None,
    ):
        self._probabilities: dict[Hashable, BaseParameter] = {}
        self._policies: dict[Hashable, BaseParameter] = {}
        self._utilities: dict[Hashable, Callable[[pd.DataFrame], object]] = {}

        declared_roles = {
            "chance": self._as_node_list(chance_nodes),
            "decision": self._as_node_list(decision_nodes),
            "utility": self._as_node_list(utility_nodes),
        }
        assigned_roles = {}
        for role, nodes in declared_roles.items():
            for node in nodes:
                if node in assigned_roles:
                    raise ValueError(f"Cannot assign more than one role to node {node!r}.")
                assigned_roles[node] = role

        super().__init__(ebunch=ebunch)

        self._check_cycles()

        for node in self.nodes:
            self.nodes[node].setdefault("role", "chance")

        for role, nodes in declared_roles.items():
            for node in nodes:
                self.add_node(node, role=role)

    @staticmethod
    def _as_node_list(nodes: Iterable[Hashable] | None):
        if nodes is None:
            return []
        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
            return [nodes]
        return list(nodes)

    def add_node(self, node_for_adding: Hashable, **attr):
        current_role = self.nodes[node_for_adding].get("role", "chance") if self.has_node(node_for_adding) else "chance"
        role = attr.pop("role", current_role)

        if not isinstance(role, str) or role not in self._NODE_ROLES:
            raise ValueError(f"role must be one of {sorted(self._NODE_ROLES)}, but got {role!r}.")

        super().add_node(node_for_adding, role=role, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for item in nodes_for_adding:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
                node, node_attr = item
                self.add_node(node, **{**attr, **node_attr})
            else:
                self.add_node(item, **attr)

    def remove_node(self, n: Hashable):
        """Remove a node and any parameterization assigned to it."""
        super().remove_node(n)
        self._probabilities.pop(n, None)
        self._policies.pop(n, None)
        self._utilities.pop(n, None)

    def remove_nodes_from(self, nodes: Iterable[Hashable]):
        """Remove nodes and any parameterizations assigned to them."""
        for node in nodes:
            if self.has_node(node):
                self.remove_node(node)

    def clear(self):
        """Remove all nodes, edges, and assigned parameterizations."""
        super().clear()
        self._probabilities.clear()
        self._policies.clear()
        self._utilities.clear()

    def copy(self):
        """Return a graph copy, retaining shallow copies of assigned parameterizations."""
        copied = type(self)()
        copied.graph.update(self.graph)
        copied.add_nodes_from((node, data.copy()) for node, data in self.nodes(data=True))
        for u, v, data in self.edges(data=True):
            nx.DiGraph.add_edge(copied, u, v)
            copied.edges[u, v].update(data.copy())
        copied._probabilities = self._probabilities.copy()
        copied._policies = self._policies.copy()
        copied._utilities = self._utilities.copy()
        return copied

    @property
    def chance_nodes(self):
        """Nodes with the chance role, in graph order."""
        return [node for node, data in self.nodes(data=True) if data.get("role") == "chance"]

    @property
    def decision_nodes(self):
        """Nodes with the decision role, in graph order."""
        return [node for node, data in self.nodes(data=True) if data.get("role") == "decision"]

    @property
    def utility_nodes(self):
        """Nodes with the utility role, in graph order."""
        return [node for node, data in self.nodes(data=True) if data.get("role") == "utility"]

    def _check_node_role(self, node: Hashable, role: str):
        if not self.has_node(node):
            raise ValueError(f"Node {node!r} is not in the diagram.")
        if self.nodes[node].get("role") != role:
            raise ValueError(f"{node!r} is a {self.nodes[node].get('role')!r} node; expected a {role} node.")

    def check_model(self):
        """Validate graph roles, utility assignments, and conditional parameterizations."""
        self._check_cycles()

        for node, attributes in self.nodes(data=True):
            role = attributes.get("role")
            if role not in self._NODE_ROLES:
                raise ValueError(f"Node {node!r} has invalid role {role!r}.")

            if role == "utility":
                if self.out_degree(node):
                    raise ValueError(f"Utility nodes must be leaves, but {node!r} has outgoing edges.")

        for node, utility in self._utilities.items():
            if not self.has_node(node) or self.nodes[node].get("role") != "utility":
                raise ValueError(f"Utility assignment target {node!r} is not an existing utility node.")
            if not callable(utility):
                raise TypeError(f"Utility assigned to node {node!r} must be callable.")

        for node, attributes in self.nodes(data=True):
            role = attributes["role"]
            if role == "utility":
                if node not in self._utilities:
                    raise ValueError(f"No utility function assigned to utility node {node!r}.")
                continue

            assignments = self._probabilities if role == "chance" else self._policies
            assignment_kind = "probability parameterization" if role == "chance" else "policy"

            parameterization = assignments.get(node)
            if parameterization is None:
                raise ValueError(f"No {assignment_kind} assigned to {role} node {node!r}.")

            if not isinstance(parameterization, BaseParameter):
                raise TypeError(f"Expected pgmpy BaseParameter, got {type(parameterization).__name__}.")

            if not parameterization.is_fitted:
                raise ValueError(f"The {assignment_kind} assigned to node {node!r} is not fitted.")

            self._check_parameterization(node, parameterization)

        return True

    def _check_parameterization(self, node: Hashable, parameterization: BaseParameter):
        """Validate a conditional parameterization against its node and graph parents.

        Unfitted parameterizations are accepted here and rejected by :meth:`check_model`.

        Raises
        ------
        TypeError
            If ``parameterization`` is not a pgmpy ``BaseParameter``.
        ValueError
            If the fitted parameterization targets a different node or has evidence that does not match the node's
            graph parents.
        """
        if not isinstance(parameterization, BaseParameter):
            raise TypeError(f"Expected pgmpy BaseParameter, got {type(parameterization).__name__}.")
        if not parameterization.is_fitted:
            return
        if parameterization.variable_ != node:
            raise ValueError(f"Parameterization for node {node!r} targets node {parameterization.variable_!r}.")

        graph_parents = set(self.predecessors(node))
        evidence = parameterization.evidence_
        if graph_parents != set(evidence) or len(evidence) != len(graph_parents):
            raise ValueError(
                f"Parameterization evidence {evidence} does not match "
                f"the parents of node {node!r}: {list(self.predecessors(node))}."
            )

    def set_probabilities(self, node: Hashable, parameterization: BaseParameter):
        """Assign a conditional probability parameterization to a chance node.

        Fitted parameterizations are checked immediately. An unfitted parameterization may be assigned and is checked
        after fitting by :meth:`check_model` or :meth:`simulate`.

        Parameters
        ----------
        node : hashable
            Chance node whose probability model is being assigned.
        parameterization : pgmpy.parameterization.BaseParameter
            Conditional distribution targeting ``node`` with evidence matching its graph parents.
        """

        self._check_node_role(node, "chance")
        self._check_parameterization(node, parameterization)
        self._probabilities[node] = parameterization

    def set_policy(self, node: Hashable, policy: BaseParameter):
        """Assign a conditional policy to a decision node.

        A deterministic discrete policy can be represented by a one-hot :class:`TabularCPD`.
        Fitted policies are validated immediately; unfitted policies are rejected by :meth:`check_model` or
        :meth:`simulate` until fitting is complete.

        Parameters
        ----------
        node : hashable
            Decision node whose policy is being assigned.
        policy : pgmpy.parameterization.BaseParameter
            Conditional distribution or deterministic policy targeting ``node`` with evidence matching its parents.
        """
        self._check_node_role(node, "decision")
        self._check_parameterization(node, policy)
        self._policies[node] = policy

    def set_utility(self, node: Hashable, utility: Callable[[pd.DataFrame], object]):
        """Assign a utility function to a utility node.

        The function receives a DataFrame containing the node's parent values and must return one finite numeric value
        per row. The target node and callable are validated by :meth:`check_model`.

        Parameters
        ----------
        node : hashable
            Utility node whose function is being assigned.
        utility : callable
            Function from parent assignments to utility values.

        """
        if not callable(utility):
            raise TypeError(f"Utility assigned to node {node!r} must be callable, got {type(utility).__name__}")

        self._utilities[node] = utility

    def simulate(
        self,
        n_samples: int = 1000,
        seed: int | np.random.Generator | None = None,
    ):
        """Generate samples from assigned chance distributions and decision policies.

        Nodes are sampled in topological order. Utility callables are then evaluated and their values are returned in
        columns named after the utility nodes. The row-wise sum of these columns is the total utility for each sample.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of rows to sample.
        seed : int or numpy.random.Generator, optional
            Seed or random number generator used for all draws.

        Returns
        -------
        pandas.DataFrame
            Sampled chance and decision values, plus one evaluated utility column per utility node.

        Raises
        ------
        ValueError
            If an assignment is missing or unfitted, if its target or parents no longer match the graph, or if a
            utility function returns values with an invalid shape or non-finite/non-numeric values.
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 0:
            raise ValueError(f"n_samples must be a non-negative integer, but got {n_samples!r}.")

        self.check_model()
        random_state = np.random.default_rng(seed)
        samples = pd.DataFrame(index=pd.RangeIndex(n_samples))
        for node in nx.topological_sort(self):
            role = self.nodes[node]["role"]
            if role == "utility":
                continue

            assignments = self._probabilities if role == "chance" else self._policies
            parameterization = assignments[node]

            parent_samples = None
            if parameterization.evidence_:
                parent_samples = samples.reindex(columns=parameterization.evidence_)
                draws = parameterization.sample(parent_samples, random_state=random_state)
            else:
                draws = parameterization.sample(n_samples=n_samples, random_state=random_state)
            samples[node] = draws.iloc[:, 0].to_numpy()

        for node in self.utility_nodes:
            utility = self._utilities.get(node)
            if utility is None:
                raise ValueError(f"No utility function assigned to utility node {node!r}.")
            parent_samples = samples.reindex(columns=list(self.predecessors(node)))
            values = utility(parent_samples)
            values = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
            if values.ndim != 1:
                raise ValueError(f"Utility function for {node!r} must return a one-dimensional result.")
            if len(values) != n_samples:
                raise ValueError(f"Utility function for {node!r} must return one value per row ({n_samples}).")
            if not all(isinstance(value, Real) for value in values):
                raise ValueError(f"Utility function for {node!r} must return numeric values.")
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Utility function for {node!r} must return finite values.")
            samples[node] = values

        return samples.reindex(columns=list(self.nodes()))
