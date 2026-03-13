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
        n_w_role = [n for n, d in G.nodes(data=True) if role in d.get("roles", set())]
        return n_w_role

    def get_roles(self):
        """Get list of all roles present in the graph.

        Returns
        -------
        List of str
            list of all roles defined in the graph.
        """
        roles = set()
        for _, d in self.nodes(data=True):
            roles.update(d.get("roles", set()))
        return list(roles)

    def get_role_dict(self):
        """Get dict of lists of roles preset in the graph.

        Returns
        -------
        Dict with str keys and values being list of nodes
            keys are roles present in the graph, and lists are nodes with that role
        """
        tpls = [(n, d.get("roles", set())) for n, d in self.nodes(data=True)]
        r_dict = {r: [] for r in self.get_roles()}

        for n, roles in tpls:
            for role in roles:
                r_dict[role].append(n)
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
            The name of the role to assign, e.g., "exposures", "outcomes".
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

        is_sem_graph = new_graph.__module__ == "pgmpy.models.SEM"
        if not is_sem_graph:
            for var in variables:
                if var not in new_graph:
                    raise ValueError(f"Variable '{var}' not found in the graph.")
                else:
                    existing_role = new_graph.nodes[var].get("roles", set())

                    existing_role.add(role)
                    new_graph.add_node(var, roles=existing_role)
        else:
            for var in variables:
                if var not in new_graph.graph:
                    raise ValueError(f"Variable '{var}' not found in the graph.")
                else:
                    existing_role = new_graph.nodes[var].get("roles", set())

                    existing_role.add(role)
                    new_graph.add_node(var, roles=existing_role)
        if not inplace:
            return new_graph

    def without_role(self, role: str, variables=None, inplace=False):
        """Return a new graph with the specified role removed.

        Parameters
        ----------
        role : str
            The name of the role to remove, e.g., "exposures", "outcomes".
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
                roles = attr.get("roles", set())
                if isinstance(roles, set) and role in roles:
                    roles.discard(role)
                    if len(roles) == 0:
                        attr.pop("roles")
                    else:
                        attr["roles"] = roles

        if not inplace:
            return new_graph

    def is_valid_causal_structure(self) -> bool:
        """Validate that the causal structure makes sense."""
        has_exposure = self.has_role("exposures")
        has_outcome = self.has_role("outcomes")
        valid = has_exposure and has_outcome

        problem_str = []
        if not has_exposure:
            problem_str.append("no 'exposures' role was defined")
        if not has_outcome:
            problem_str.append("no 'outcomes' role was defined")
        problem_str = ", and ".join(problem_str)

        if not valid:
            raise ValueError(
                f"{type(self)} must have at least one 'exposures' and one 'outcomes' "
                f"role defined, but {problem_str}."
            )
        return True

    @property
    def latents(self):
        """
        Returns the set of latent variables in the causal model.

        Property
        --------
        latents : set of nodes (default: empty set)
            A set of latent variables in the graph. These are not observed
            variables but are used to represent unobserved confounding or
            other latent structures.

        Examples
        --------
        Create a DAG with latents and check the latents value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], latents="a")
        >>> G.latents
        {'a'}
        """
        if self.has_role("latents"):
            return set(self.get_role("latents"))
        else:
            return set()

    @latents.setter
    def latents(self, variables):
        """
        Sets the latent variables in the model. If latents already exist, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of latent variables in the graph. These are not observed
            variables but are used to represent unobserved confounding or
            other latent structures.
        """
        if self.has_role("latents"):
            self.without_role(
                role="latents", variables=self.get_role("latents"), inplace=True
            )
        self.with_role(role="latents", variables=variables, inplace=True)

    @property
    def observed(self):
        """
        Returns the set of observed variables in the causal model.

        Property
        --------
        observed: set of nodes (default: empty set)
            A set of observed variables in the graph. These are the variables
            that can be measured directed and have data available for them.

        Examples
        --------
        Create a DAG with latents and check the observed value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], latents="a")
        >>> G.observed
        {'b'}
        """
        nodes = set(self.nodes())
        if self.has_role("latents"):
            return nodes - set(self.get_role("latents"))
        else:
            return nodes

    @property
    def exposures(self):
        """
        Returns the set of exposure variables in the causal model.

        Property
        --------
        exposures : set of nodes (default: empty set)
            A set of exposure variables in the graph. These are the variables
            that represent the treatment or intervention being studied in a
            causal analysis.

        Examples
        --------
        Create a DAG with exposures and check the exposures value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], exposures="a")
        >>> G.exposures
        {'a'}
        """
        if self.has_role("exposures"):
            return set(self.get_role("exposures"))
        else:
            return set()

    @exposures.setter
    def exposures(self, variables):
        """
        Sets the exposure variables in the model. If exposure variables are already defined, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of exposure variables in the graph. These are the variables that represent the treatment or
            intervention being studied in a causal analysis.
        """
        if self.has_role("exposures"):
            self.without_role(
                role="exposures", variables=self.get_role("exposures"), inplace=True
            )
        self.with_role(role="exposures", variables=variables, inplace=True)

    @property
    def outcomes(self):
        """
        Returns the set of outcome variables in the causal model.

        Property
        --------
        outcomes : set of nodes (default: empty set)
            A set of outcome variables in the graph. These are the variables
            that represent the response or dependent variables being studied
            in a causal analysis.

        Examples
        --------
        Create a DAG with outcomes and check the outcomes value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], outcomes="b")
        >>> G.outcomes
        {'b'}
        """
        if self.has_role("outcomes"):
            return set(self.get_role("outcomes"))
        else:
            return set()

    @outcomes.setter
    def outcomes(self, variables):
        """
        Sets the outcome variables in the model. If outcome variables are already defined, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of outcome variables in the graph. These are the variables
            that represent the response or dependent variables being studied
            in a causal analysis.
        """
        if self.has_role("outcomes"):
            self.without_role(
                role="outcomes", variables=self.get_role("outcomes"), inplace=True
            )
        self.with_role(role="outcomes", variables=variables, inplace=True)
