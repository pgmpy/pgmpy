from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from numbers import Integral, Real

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base.DAG import DAG
from pgmpy.parameterization._base import BaseParameter


class CausalInfluenceDiagram(DAG):
    """A representation class for influence diagrams with chance, decision, and utility nodes.
    This class extends :class:`pgmpy.base.DAG`.

    * Nodes represent random variables, decisions, or utility quantities.
    * A directed edge ``A -> B`` indicates that ``A`` is a parent of ``B``.
    * The graph must not contain directed cycles.
    * The parents of a chance or decision node define the evidence available to its
    probability parameterization or policy.
    * Nodes are processed in topological order during :meth:`simulate`, so all
    parent values are available before a node is sampled.
    * Utility nodes must be leaves. Their parent values are passed to the assigned
    utility function, but utility nodes are not sampled from a probability
    distribution.

    Unlike a plain ``DAG``, this class uses mutually exclusive node roles:
    ``"chance"``, ``"decision"``, and ``"utility"``. These roles
    determine which parameterization is required during validation and simulation.
    Chance nodes have conditional probability parameterizations, decision nodes have policies, and utility nodes have
    functions of their parents. Nodes added without a role default to ``"chance"``, and assigning one of these roles
    with :meth:`with_role` replaces the node's previous one.

    Parameters
    ----------
    ebunch : input graph, optional
        Data used to initialize the graph, as accepted by :class:`pgmpy.base.DAG`. Nodes in the role
        arguments below must be in ``ebunch``.
    chance_nodes : iterable of hashable, optional
        Nodes to assign the ``"chance"`` role.
    decision_nodes : iterable of hashable, optional
        Nodes to assign the ``"decision"`` role.
    utility_nodes : iterable of hashable, optional
        Nodes to assign the ``"utility"`` role. Utility nodes must be leaves.
    latents : iterable of hashable, optional
        Nodes to assign the ``"latents"`` role. Latent nodes can only be either chance or decision nodes.
        They cannot be utility nodes.

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
    """

    _ALLOWED_NODE_ROLES = ("chance", "decision", "utility")

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        chance_nodes: Iterable[Hashable] | None = None,
        decision_nodes: Iterable[Hashable] | None = None,
        utility_nodes: Iterable[Hashable] | None = None,
        latents: Iterable[Hashable] | None = None,
    ):
        self._probabilities: dict[Hashable, BaseParameter] = {}
        self._policies: dict[Hashable, BaseParameter] = {}
        self._utilities: dict[Hashable, Callable[[pd.DataFrame], object]] = {}

        roles = {
            "chance": self._as_node_list(chance_nodes),
            "decision": self._as_node_list(decision_nodes),
            "utility": self._as_node_list(utility_nodes),
            "latents": self._as_node_list(latents),
        }

        chance, decision, utility = (set(roles[role]) for role in self._ALLOWED_NODE_ROLES)
        overlap = (chance & decision) | (chance & utility) | (decision & utility)
        if overlap:
            raise ValueError(
                f"Cannot assign more than one role among 'chance', 'decision', and 'utility' to nodes {overlap}."
            )

        super().__init__(ebunch=ebunch, roles=roles)

    @staticmethod
    def _as_node_list(nodes: Iterable[Hashable] | None):
        if nodes is None:
            return []
        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
            return [nodes]
        return list(nodes)

    def _node_roles(self, node: Hashable) -> list[str]:
        """Get the chance, decision, and utility roles of ``node``."""
        return [role for role in self._ALLOWED_NODE_ROLES if node in self.get_role(role)]

    def add_node(self, node_for_adding: Hashable, role: str | None = None, **attr):
        """Add a node with a role; new nodes default to ``"chance"``."""
        if role is not None and role not in self._ALLOWED_NODE_ROLES:
            raise ValueError(f"role must be one of {list(self._ALLOWED_NODE_ROLES)}, but got {role!r}.")

        previous_roles = self._node_roles(node_for_adding)
        super().add_node(node_for_adding, **attr)
        current_roles = self._node_roles(node_for_adding)

        if role is None:
            role = ([r for r in current_roles if r not in previous_roles] or current_roles or ["chance"])[0]
        elif previous_roles and previous_roles != [role]:
            warnings.warn(
                f"Overwriting role of node {node_for_adding!r} from {previous_roles} to {role!r}.",
                UserWarning,
            )

        for current_role in current_roles:
            if current_role != role:
                self.without_role(role=current_role, variables=[node_for_adding], inplace=True)
        if role not in current_roles:
            self.with_role(role=role, variables=[node_for_adding], inplace=True)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for item in nodes_for_adding:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
                node, node_attr = item
                self.add_node(node, **{**attr, **node_attr})
            else:
                self.add_node(item, **attr)

    def add_edge(self, u_of_edge: Hashable, v_of_edge: Hashable, **attr):
        self.add_nodes_from(node for node in (u_of_edge, v_of_edge) if node not in self)
        super().add_edge(u_of_edge, v_of_edge, **attr)

    def remove_node(self, n: Hashable):
        """Remove a node and any parameterization assigned to it."""
        super().remove_node(n)
        self._probabilities.pop(n, None)
        self._policies.pop(n, None)
        self._utilities.pop(n, None)

    def remove_nodes_from(self, nodes: Iterable[Hashable]):
        """Remove nodes and any parameterizations assigned to them."""
        for node in nodes:
            self.remove_node(node)

    def clear(self):
        """Remove all nodes, edges, and assigned parameterizations."""
        super().clear()
        self._probabilities.clear()
        self._policies.clear()
        self._utilities.clear()

    def copy(self):
        """Copy the diagram, its roles, and assigned parameterizations."""
        copied = super().copy()
        copied._probabilities = self._probabilities.copy()
        copied._policies = self._policies.copy()
        copied._utilities = self._utilities.copy()
        return copied

    @property
    def chance_nodes(self):
        """Get the nodes with the chance role, in graph order."""
        return self.get_role("chance")

    @property
    def decision_nodes(self):
        """Get the nodes with the decision role, in graph order."""
        return self.get_role("decision")

    @property
    def utility_nodes(self):
        """Get the nodes with the utility role, in graph order."""
        return self.get_role("utility")

    def check_model(self):
        """
        Validate graph roles, utility assignments, and conditional parameterizations.

        * Checks for acyclicity and every node has exactly one valid role.
        * Every utility node is a non-latent leaf.
        * Every decision node has a fitted policy.
        * checks that each parameterization targets the correct node and has evidence
          matching the node's graph parents.
        """
        self._check_cycles()

        for node in self.nodes():
            node_roles = self._node_roles(node)
            if len(node_roles) != 1:
                raise ValueError(
                    f"Node {node!r} must have exactly one of the roles {list(self._ALLOWED_NODE_ROLES)}, "
                    f"but has {node_roles}."
                )

        for node, utility in self._utilities.items():
            if node not in self.utility_nodes:
                raise ValueError(f"Utility assignment target {node!r} is not an existing utility node.")
            if not callable(utility):
                raise TypeError(f"Utility assigned to node {node!r} must be callable.")

        for node in self.utility_nodes:
            if self.out_degree(node):
                raise ValueError(f"Utility nodes must be leaves, but {node!r} has outgoing edges.")
            if node in self.latents:
                raise ValueError(f"Utility nodes cannot be latent, but {node!r} has the 'latents' role.")
            if node not in self._utilities:
                raise ValueError(f"No utility function assigned to utility node {node!r}.")

        for role, assignments, assignment_kind in (
            ("chance", self._probabilities, "probability parameterization"),
            ("decision", self._policies, "policy"),
        ):
            for node in self.get_role(role):
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
        if not self.has_node(node):
            raise ValueError(f"Chance node {node!r} is not in the diagram.")
        node_roles = self._node_roles(node)
        if node_roles != ["chance"]:
            raise ValueError(
                f"Cannot assign probabilities to {node!r}: node roles are {node_roles}, expected 'chance'."
            )
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
        if not self.has_node(node):
            raise ValueError(f"Decision node {node!r} is not in the diagram.")
        node_roles = self._node_roles(node)
        if node_roles != ["decision"]:
            raise ValueError(f"Cannot assign a policy to {node!r}: node roles are {node_roles}, expected 'decision'.")
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

        if not self.has_node(node):
            raise ValueError(f"Utility node {node!r} is not in the diagram.")
        node_roles = self._node_roles(node)
        if node_roles != ["utility"]:
            raise ValueError(
                f"Cannot assign a utility function to {node!r}: node roles are {node_roles}, expected 'utility'."
            )

        if not callable(utility):
            raise TypeError(f"Utility assigned to node {node!r} must be callable, got {type(utility).__name__}")

        self._utilities[node] = utility

    def simulate(
        self,
        n_samples: int = 1000,
        rng: int | np.random.Generator | None = None,
    ):
        """Generate samples from assigned chance distributions and decision policies.

        Nodes are sampled in topological order. Utility callables are then evaluated and their values are returned in
        columns named after the utility nodes. The row-wise sum of these columns is the total utility for each sample.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of rows to sample.
        rng : int or numpy.random.Generator, optional
            random number generator used for all draws.

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
        random_state = np.random.default_rng(rng)
        samples = pd.DataFrame(index=pd.RangeIndex(n_samples))
        decision_nodes = set(self.decision_nodes)
        utility_nodes = set(self.utility_nodes)
        for node in nx.topological_sort(self):
            if node in utility_nodes:
                continue

            assignments = self._policies if node in decision_nodes else self._probabilities
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
            samples[node] = values

        return samples.reindex(columns=list(self.nodes()))
