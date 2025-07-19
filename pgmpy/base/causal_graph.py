#!/usr/bin/env python3

from typing import Hashable


import networkx as nx


class _GraphRolesMixin:
    """Mixin class for handling roles in a causal graph."""

    def get_role(self, role: str):
        """Return list of nodes in graph G with a specific role.

        Parameters
        ----------
        role : str
            The role to match.

        Returns
        -------
        List of nodes with the specified role.
        """
        G = self
        n_w_role = [n for n, d in G.nodes(data=True) if d.get("role", None) == role]
        return n_w_role

    def get_roles(self):
        """Get list of all roles present in the graph.

        Returns
        -------
        List of str
            list of all roles defined in the graph.
        """
        roles = {d.get("role", None) for _, d in self.nodes(data=True)}
        roles.discard(None)  # remove "None"
        return list(roles)

    def has_role(self, role: str) -> bool:
        """Check if a role is defined and non-empty.

        Parameters
        ----------
        role : str
            The name of the role to check.

        Returns
        -------
        bool
            True if the role exists and has variables assigned, False otherwise.
        """
        return role in self.get_roles()

    def with_role(self, role: str, variables, inplace=False) -> "CausalGraph":
        """Return a new CausalGraph with the specified role assignment.

        Parameters
        ----------
        role : str
            The name of the role to assign, e.g., "exposure", "outcome".
        variables : set, list, or any iterable
            The variables to assign to the role.
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        CausalGraph
            A new CausalGraph instance with the specified role assigned,
            to the variables provided.
        """
        if not inplace:
            new_graph = self.copy()

        for var in variables:
            new_graph.add_node(var, role=role)

        return new_graph

    def without_role(self, role: str, inplace=False) -> "CausalGraph":
        """Return a new CausalGraph with the specified role removed.

        Parameters
        ----------
        role : str
            The name of the role to remove, e.g., "exposure", "outcome".
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        CausalGraph
            A new CausalGraph instance with the specified role removed
            from all nodes that had it.
        """
        if not inplace:
            new_graph = self.copy()

        for _, attr in new_graph.nodes(data=True):
            if attr.get("role") == role:
                attr.pop("role")
        return new_graph

    def is_valid_causal_structure(self) -> bool:
        """Validate that the causal structure makes sense."""
        has_exposure = self.has_role("exposure")
        has_outcome = self.has_role("outcome")
        valid = has_exposure and has_outcome

        problem_str = []
        if not has_exposure:
            problem_str.append("no 'exposure' role was defined")
        if not has_outcome:
            problem_str.append("no 'outcome' role was defined")
        problem_str = ", and ".join(problem_str)

        if not valid:
            raise ValueError(
                f"CausalGraph must have at least one 'exposure' and one 'outcome'"
                f"role defined, but {problem_str}."
            )
        return True


class CausalGraph(_GraphRolesMixin, nx.DiGraph):
    """A causal graphical model which manages variable roles explicitly.

    Typical roles are exposure, outcome, adjustment set, but this structure
    supports any roles.

    Extends ``networkx.DiGraph`` with additional functionality for
    managing variable roles and ensuring the integrity of causal
    relationships.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    roles : dict, optional (default: None)
        A dictionary mapping node names to their roles. The keys are node names
        and the values are role names (strings). If provided, this will
        automatically assign roles to the nodes in the graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    Examples
    --------
    >>> from pgmpy.base.causal_graph import CausalGraph
    >>>
    >>> cg = CausalGraph(
    ...     [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
    ...     roles={
    ...         "X": "exposure",
    ...         "Y": "outcome",
    ...     }
    ... )
    >>> cg.get_role("exposure")
    ['X']
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize a graph with edges, name, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph (optional, default: None)
            Data to initialize graph. If None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a 2D NumPy array, a
            SciPy sparse array, or a PyGraphviz graph.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.
        """
        if "roles" in attr:
            roles = attr.pop("roles")
            if not isinstance(roles, dict):
                raise TypeError("Roles must be provided as a dictionary.")
        super().__init__(incoming_graph_data, **attr)

        for node, role in roles.items():
            if not isinstance(node, Hashable):
                raise TypeError("Node names must be hashable.")
            if not isinstance(role, str):
                raise TypeError("Role names must be strings.")
            self.add_node(node, role=role)
