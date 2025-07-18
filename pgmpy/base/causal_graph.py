#!/usr/bin/env python3

from typing import Dict, Hashable, Iterable, Optional, Set, Union
from copy import deepcopy

from pgmpy.base.DAG import DAG

ROLE_ALIASES = {
    "treatment": "exposure",
    "target": "outcome",
    "intervention": "exposure",
    "response": "outcome",
}


class CausalGraph:
    """
    A causal graphical model combining a directed acyclic graph (DAG)
    structure with metadata about the roles of variables in the causal
    system (e.g., exposure, outcome, adjustment set).

    Parameters
    ----------
    graph : DAG, list of edges, or any graph-like object
        The underlying graph structure.
    roles : dict, optional
        A dictionary mapping role names to sets of variables.
    **role_kwargs : keyword arguments
        Individual role assignments (e.g., exposure="X", outcome="Y").

    Examples
    --------
    >>> cg = CausalGraph(
    ...     graph=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
    ...     exposure="X",
    ...     outcome="Y"
    ... )
    >>> cg.get_role("exposure")
    {'X'}
    """

    def __init__(
        self,
        graph: Union[DAG, Iterable[tuple[Hashable, Hashable]], None] = None,
        roles: Optional[Dict[str, Union[Set[Hashable], Hashable]]] = None,
        **role_kwargs,
    ):
        if isinstance(graph, DAG):
            self._graph = graph.copy()
        else:
            self._graph = DAG(ebunch=graph)

        self._roles: Dict[str, Set[Hashable]] = {}

        if roles is not None:
            for role_name, role_vars in roles.items():
                role_name = self._resolve_role_alias(role_name)
                self._set_role_internal(role_name, role_vars)

        for role_name, role_vars in role_kwargs.items():
            role_name = self._resolve_role_alias(role_name)
            self._set_role_internal(role_name, role_vars)

    def _resolve_role_alias(self, role_name: str) -> str:
        return ROLE_ALIASES.get(role_name, role_name)

    def _set_role_internal(
        self, role_name: str, variables: Union[Set[Hashable], Hashable]
    ):
        if isinstance(variables, (set, list, tuple)):
            self._roles[role_name] = set(variables)
        else:
            self._roles[role_name] = {variables}

    def _validate_variables_exist(self, variables: Set[Hashable]):
        graph_nodes = set(self._graph.nodes())
        invalid_vars = variables - graph_nodes
        if invalid_vars:
            raise ValueError(f"Variables {invalid_vars} not found in the graph")

    def get_role(self, role_name: str) -> Set[Hashable]:
        """Get variables assigned to a role."""
        role_name = self._resolve_role_alias(role_name)
        return self._roles.get(role_name, set()).copy()

    def get_roles(self) -> Dict[str, Set[Hashable]]:
        """Get all role assignments."""
        return {role: vars_set.copy() for role, vars_set in self._roles.items()}

    def has_role(self, role_name: str) -> bool:
        """Check if a role is defined and non-empty."""
        role_name = self._resolve_role_alias(role_name)
        return role_name in self._roles and len(self._roles[role_name]) > 0

    def with_role(
        self, role_name: str, variables: Union[Set[Hashable], Hashable]
    ) -> "CausalGraph":
        """Return a new CausalGraph with the specified role assignment."""
        role_name = self._resolve_role_alias(role_name)

        if isinstance(variables, (set, list, tuple)):
            var_set = set(variables)
        else:
            var_set = {variables}

        self._validate_variables_exist(var_set)

        new_roles = self._roles.copy()
        new_roles[role_name] = var_set

        new_cg = CausalGraph(graph=self._graph)
        new_cg._roles = new_roles
        return new_cg

    def without_role(self, role_name: str) -> "CausalGraph":
        """Return a new CausalGraph with the specified role removed."""
        role_name = self._resolve_role_alias(role_name)

        new_roles = self._roles.copy()
        new_roles.pop(role_name, None)

        new_cg = CausalGraph(graph=self._graph)
        new_cg._roles = new_roles
        return new_cg

    def __getattr__(self, name):
        """Forward attribute access to the underlying DAG."""
        if hasattr(self._graph, name):
            return getattr(self._graph, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def validate_roles(self) -> bool:
        """Validate that all assigned role variables exist in the graph."""
        graph_nodes = set(self._graph.nodes())
        for role_name, variables in self._roles.items():
            invalid_vars = variables - graph_nodes
            if invalid_vars:
                raise ValueError(
                    f"Role '{role_name}' contains variables {invalid_vars} "
                    f"that don't exist in the graph"
                )
        return True

    def copy(self) -> "CausalGraph":
        """Return a deep copy of the CausalGraph."""
        new_cg = CausalGraph(graph=self._graph)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def __str__(self) -> str:
        return f"CausalGraph(nodes={list(self.nodes())}, edges={list(self.edges())}, roles={dict(self._roles)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check equality between CausalGraph instances."""
        if not isinstance(other, CausalGraph):
            return False

        if set(self.nodes()) != set(other.nodes()) or set(self.edges()) != set(
            other.edges()
        ):
            return False

        return self._roles == other._roles

    def __hash__(self) -> int:
        """Hash function for CausalGraph instances."""
        nodes_hash = hash(tuple(sorted(self.nodes(), key=str)))
        edges_hash = hash(tuple(sorted(self.edges(), key=str)))

        roles_items = []
        for role_name in sorted(self._roles.keys()):
            role_vars = tuple(sorted(self._roles[role_name], key=str))
            roles_items.append((role_name, role_vars))
        roles_hash = hash(tuple(roles_items))

        return hash((nodes_hash, edges_hash, roles_hash))

    def to_dag(self) -> DAG:
        """Return a copy of the underlying DAG."""
        return self._graph.copy()

    def is_valid_causal_structure(self) -> bool:
        """Validate that the causal structure makes sense."""
        self.validate_roles()

        for role in ["exposure", "outcome"]:
            if self.has_role(role):
                role_vars = self.get_role(role)
                if len(role_vars) > 1:
                    raise ValueError(
                        f"Role '{role}' should contain only one variable, "
                        f"but has {len(role_vars)}: {role_vars}"
                    )

        return True

    def with_nodes(self, nodes: Iterable[Hashable]) -> "CausalGraph":
        """Return a new CausalGraph with additional nodes."""
        new_dag = self._graph.copy()
        new_dag.add_nodes_from(nodes)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def with_edges(self, edges: Iterable[tuple[Hashable, Hashable]]) -> "CausalGraph":
        """Return a new CausalGraph with additional edges."""
        new_dag = self._graph.copy()
        new_dag.add_edges_from(edges)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def without_nodes(self, nodes: Iterable[Hashable]) -> "CausalGraph":
        """Return a new CausalGraph with specified nodes removed."""
        new_dag = self._graph.copy()
        new_dag.remove_nodes_from(nodes)

        nodes_to_remove = set(nodes)
        new_roles = {}
        for role_name, role_vars in self._roles.items():
            remaining_vars = role_vars - nodes_to_remove
            if remaining_vars:
                new_roles[role_name] = remaining_vars

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = new_roles
        return new_cg

    def without_edges(
        self, edges: Iterable[tuple[Hashable, Hashable]]
    ) -> "CausalGraph":
        """Return a new CausalGraph with specified edges removed."""
        new_dag = self._graph.copy()
        new_dag.remove_edges_from(edges)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg
