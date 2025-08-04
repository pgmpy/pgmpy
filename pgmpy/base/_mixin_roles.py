#!/usr/bin/env python3

__all__ = ["_GraphRolesMixin"]


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

    def get_role_dict(self):
        """Get dict of lists of roles preset in the graph.

        Returns
        -------
        Dict with str keys and values being list of nodes
            keys are roles present in the graph, and lists are nodes with that role
        """
        tpls = [(n, d.get("role", None)) for n, d in self.nodes(data=True)]
        r_dict = {r: [] for r in self.get_roles()}

        for n, r in tpls:
            if r is not None:
                r_dict[r].append(n)
        return r_dict

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

    def with_role(self, role: str, variables, inplace=False):
        """Return a new graph with the specified role assignment.

        Parameters
        ----------
        role : str
            The name of the role to assign, e.g., "exposure", "outcome".
        variables : str, set, list, or any iterable
            The variables to assign to the role.
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        graph of same type as self
            A new instance with the specified role assigned, to the variables provided.
        """
        if isinstance(variables, str):
            variables = {variables}

        if not inplace:
            new_graph = self.copy()
        else:
            new_graph = self

        for var in variables:
            if var not in new_graph:
                raise ValueError(f"Variable '{var}' not found in the graph.")
            else:
                new_graph.add_node(var, role=role)

        return new_graph

    def without_role(self, role: str, variables=None, inplace=False):
        """Return a new graph with the specified role removed.

        Parameters
        ----------
        role : str
            The name of the role to remove, e.g., "exposure", "outcome".
        variables : str, set, list, or iterable, default = all variables with the role
            The variables to remove the role from. If not provided,
            all variables with the specified role will have it removed.
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        graph of same type as self
            A new instance with the specified role removed from all nodes that had it.
        """
        if isinstance(variables, str):
            variables = {variables}

        if not inplace:
            new_graph = self.copy()
        else:
            new_graph = self

        for v, attr in new_graph.nodes(data=True):
            if variables is None or v in variables:
                if attr.get("role", None) == role:
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
                f"{type(self)} must have at least one 'exposure' and one 'outcome'"
                f"role defined, but {problem_str}."
            )
        return True
