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
    A causal graphical model that extends a DAG with variable roles.

    This class represents a causal graphical model by combining a directed acyclic
    graph (DAG) structure with metadata about the roles of variables in the causal
    system (e.g., exposure, outcome, adjustment set).

    The class uses composition over inheritance, internally holding a DAG instance
    and forwarding relevant graph operations while managing variable roles.

    Parameters
    ----------
    graph : DAG, list of edges, or any graph-like object
        The underlying graph structure. Can be:
        - An existing DAG instance
        - A list of edges as tuples (u, v)
        - Any object that can be passed to DAG constructor

    roles : dict, optional
        A dictionary mapping role names to sets of variables.
        Example: {"adjustment_set": {"Z1", "Z2"}}

    **role_kwargs : keyword arguments
        Individual role assignments. Common roles include:
        - exposure: The treatment or intervention variable
        - outcome: The target or response variable
        - adjustment_set: Variables to adjust for

    Examples
    --------
    >>> # Create a simple causal graph
    >>> cg = CausalGraph(
    ...     graph=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
    ...     exposure="X",
    ...     outcome="Y"
    ... )

    >>> # Access the graph structure
    >>> cg.get_nodes()
    ['U', 'X', 'M', 'Y']

    >>> # Check variable roles
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
        """
        Get the set of variables assigned to a role.

        Parameters
        ----------
        role_name : str
            The name of the role to retrieve.

        Returns
        -------
        set
            Set of variables assigned to the role. Empty set if role doesn't exist.
        """
        role_name = self._resolve_role_alias(role_name)
        return self._roles.get(role_name, set()).copy()

    def get_roles(self) -> Dict[str, Set[Hashable]]:
        """
        Get all role assignments.

        Returns
        -------
        dict
            Dictionary mapping role names to sets of variables.
        """
        return {role: vars_set.copy() for role, vars_set in self._roles.items()}

    def has_role(self, role_name: str) -> bool:
        """
        Check if a role is defined and non-empty.

        Parameters
        ----------
        role_name : str
            The name of the role to check.

        Returns
        -------
        bool
            True if the role exists and has at least one variable assigned.
        """
        role_name = self._resolve_role_alias(role_name)
        return role_name in self._roles and len(self._roles[role_name]) > 0

    def with_role(
        self, role_name: str, variables: Union[Set[Hashable], Hashable]
    ) -> "CausalGraph":
        """
        Return a new CausalGraph with the specified role assignment.

        This method maintains immutability by returning a new instance.

        Parameters
        ----------
        role_name : str
            The name of the role to set.
        variables : set or hashable
            The variable(s) to assign to the role.

        Returns
        -------
        CausalGraph
            A new CausalGraph instance with the role assignment.
        """
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
        """
        Return a new CausalGraph with the specified role removed.

        Parameters
        ----------
        role_name : str
            The name of the role to remove.

        Returns
        -------
        CausalGraph
            A new CausalGraph instance without the specified role.
        """
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
        """
        Validate that all assigned role variables exist in the graph.

        Returns
        -------
        bool
            True if all role assignments are valid.

        Raises
        ------
        ValueError
            If any assigned variable doesn't exist in the graph.
        """
        graph_nodes = set(self._graph.nodes())
        for role_name, variables in self._roles.items():
            invalid_vars = variables - graph_nodes
            if invalid_vars:
                raise ValueError(
                    f"Role '{role_name}' contains variables {invalid_vars} "
                    f"that don't exist in the graph"
                )
        return True

    def get_variable_roles(self, variable: Hashable) -> Set[str]:
        """
        Get all roles assigned to a specific variable.

        Parameters
        ----------
        variable : hashable
            The variable to check.

        Returns
        -------
        set
            Set of role names that include this variable.
        """
        roles = set()
        for role_name, variables in self._roles.items():
            if variable in variables:
                roles.add(role_name)
        return roles

    def get_unassigned_variables(self) -> Set[Hashable]:
        """
        Get variables that are not assigned to any role.

        Returns
        -------
        set
            Set of variables not assigned to any role.
        """
        all_nodes = set(self._graph.nodes())
        assigned_vars = set()
        for variables in self._roles.values():
            assigned_vars.update(variables)
        return all_nodes - assigned_vars

    def copy(self) -> "CausalGraph":
        """Return a deep copy of the CausalGraph."""
        new_cg = CausalGraph(graph=self._graph)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def __str__(self) -> str:
        """String representation of the CausalGraph."""
        nodes_str = f"Nodes: {list(self.nodes())}"
        edges_str = f"Edges: {list(self.edges())}"
        roles_str = f"Roles: {dict(self._roles)}"
        return f"CausalGraph(\n  {nodes_str}\n  {edges_str}\n  {roles_str}\n)"

    def __repr__(self) -> str:
        """Detailed representation of the CausalGraph."""
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

    def summary(self) -> str:
        """
        Return a detailed summary of the CausalGraph.

        Returns
        -------
        str
            Multi-line summary including graph structure and roles.
        """
        lines = ["CausalGraph Summary:"]
        lines.append(f"  Nodes ({self.number_of_nodes()}): {list(self.nodes())}")
        lines.append(f"  Edges ({self.number_of_edges()}): {list(self.edges())}")

        if self._roles:
            lines.append("  Roles:")
            for role_name, variables in sorted(self._roles.items()):
                if len(variables) == 1:
                    lines.append(f"    {role_name}: {list(variables)[0]}")
                else:
                    lines.append(f"    {role_name}: {sorted(list(variables), key=str)}")
        else:
            lines.append("  Roles: None defined")

        unassigned = self.get_unassigned_variables()
        if unassigned:
            lines.append(f"  Unassigned variables: {sorted(list(unassigned), key=str)}")

        return "\n".join(lines)

    def to_dag(self) -> DAG:
        """
        Return a copy of the underlying DAG.

        This can be useful when you need to pass the graph structure
        to other pgmpy functions that expect a DAG.

        Returns
        -------
        DAG
            A copy of the underlying DAG structure.
        """
        return self._graph.copy()

    def is_valid_causal_structure(self) -> bool:
        """
        Validate that the causal structure makes sense.

        Performs basic validation checks:
        - All role variables exist in the graph
        - Exposure and outcome are single variables (if defined)
        - No cycles in the graph

        Returns
        -------
        bool
            True if the structure is valid.

        Raises
        ------
        ValueError
            If validation fails with details about the issue.
        """
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
        """
        Return a new CausalGraph with additional nodes.

        Parameters
        ----------
        nodes : iterable
            Nodes to add to the graph.

        Returns
        -------
        CausalGraph
            A new CausalGraph with the additional nodes.
        """
        new_dag = self._graph.copy()
        new_dag.add_nodes_from(nodes)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def with_edges(self, edges: Iterable[tuple[Hashable, Hashable]]) -> "CausalGraph":
        """
        Return a new CausalGraph with additional edges.

        Parameters
        ----------
        edges : iterable of tuples
            Edges to add to the graph as (u, v) tuples.

        Returns
        -------
        CausalGraph
            A new CausalGraph with the additional edges.
        """
        new_dag = self._graph.copy()
        new_dag.add_edges_from(edges)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg

    def without_nodes(self, nodes: Iterable[Hashable]) -> "CausalGraph":
        """
        Return a new CausalGraph with specified nodes removed.

        Also removes any role assignments involving the removed nodes.

        Parameters
        ----------
        nodes : iterable
            Nodes to remove from the graph.

        Returns
        -------
        CausalGraph
            A new CausalGraph with the nodes removed.
        """
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
        """
        Return a new CausalGraph with specified edges removed.

        Parameters
        ----------
        edges : iterable of tuples
            Edges to remove from the graph as (u, v) tuples.

        Returns
        -------
        CausalGraph
            A new CausalGraph with the edges removed.
        """
        new_dag = self._graph.copy()
        new_dag.remove_edges_from(edges)

        new_cg = CausalGraph(graph=new_dag)
        new_cg._roles = deepcopy(self._roles)
        return new_cg
