from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pgmpy.base import DAG


@dataclass
class NodeObject:
    """Read-only view of node metadata returned by :meth:`BayesianNetwork.get_node`."""

    node: Hashable
    parents: set[Hashable] = field(default_factory=set)
    children: set[Hashable] = field(default_factory=set)
    roles: Any = None
    local_model: Any = None
    estimator: Any = None
    data: Any = None

    def __str__(self) -> str:
        """Return a readable summary of node-level metadata."""
        return (
            "NodeObject("
            f"node={self.node!r}, "
            f"parents={sorted(self.parents, key=str)!r}, "
            f"children={sorted(self.children, key=str)!r}, "
            f"roles={sorted(self.roles)!r}, "
            f"local_model={type(self.local_model).__name__ if self.local_model is not None else None}, "
            f"estimator={type(self.estimator).__name__ if self.estimator is not None else None}, "
            f"data={'set' if self.data is not None else None}"
            ")"
        )


class BayesianNetwork(DAG):
    """Graph container for Bayesian network structure and CPD metadata.

    This class intentionally focuses on model-definition responsibilities only
    (nodes, edges, roles, latent/exposure/outcome annotations, and CPD storage).
    Learning, inference, prediction, and simulation should be handled by
    estimator/inference components that consume this model.
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_cpds(self, *cpds: Any) -> None:
        """Add CPDs to the model.

        CPD objects are expected to provide at least a `variable` attribute.
        If present, `variables` and `cardinality` are used to update
        `self.cardinalities`.
        """
        for cpd in cpds:
            self.cpds.append(cpd)
            if cpd.variable in self.nodes:
                self.nodes[cpd.variable]["local_model"] = cpd

    def get_cpds(self, node: Hashable | None = None) -> Any:
        """Return CPD(s) in the model or CPD associated with a node."""
        if node is None:
            return self.cpds

        if node not in self.nodes():
            raise ValueError("Node not present in the graph")

        for cpd in self.cpds:
            if cpd.variable == node:
                return cpd
        return None

    def get_node(self, node: Hashable) -> NodeObject:
        if node not in self.nodes():
            raise ValueError("Node not present in the graph")

        node_attrs = self.nodes[node]

        roles = self.nodes[node]["roles"]

        return NodeObject(
            node=node,
            parents=set(self.predecessors(node)),
            children=set(self.successors(node)),
            roles=roles,
            local_model=node_attrs.get("local_model"),
            estimator=node_attrs.get("estimator"),
            data=node_attrs.get("data"),
        )

    def check_model(self) -> bool:
        """Validate that each node has a CPD and CPDs are structurally consistent."""
        for node in self.nodes():
            cpd = self.get_cpds(node=node)
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")

            evidence = set(getattr(cpd, "get_evidence", lambda: [])())
            parents = set(self.get_parents(node))
            if evidence != parents:
                raise ValueError(f"CPD associated with {node} doesn't have proper parents associated with it.")

            if hasattr(cpd, "is_valid_cpd") and not cpd.is_valid_cpd():
                raise ValueError(f"Sum or integral of conditional probabilities for node {node} is not equal to 1.")

        return True
