class SEMLISREL:
    """
    Base class for algebraic representation of Structural Equation Models(SEMs).
    """
    def __init__(self, str_model=None, params=None, fixed_params=None):
        pass

    def to_SEMGraph(self, params):
        """
        Sets the params in the graph structure. The model is defined as:

        eta = B eta + Gamma xi + zeta
        y = wedge_y eta + epsilon
        x = wedge_x xi + delta
        Psi = COV(eta)
        Phi = COV(xi)
        theta_e = COV(y)
        theta_{del} = COV(x)

        Parameters
        ----------
        params: dict
            A dictionary with the parameters to set to the model. Each array should assume that
            the variable lists eta, xi, y and x are sorted.

            Expected keys in `params`:
                1. B: array (m x m)
                2. gamma: array (m x n)
                3. wedge_y: array (p x m)
                4. wedge_x: array (q x n)
                5. phi: array (n x n)
                6. theta_e: array (p x p)
                7. theta_del: array (q x q)
                8. psi: array (m x m)
            m = len(self.eta)
            n = len(self.xi)
            p = len(self.y)
            q = len(self.x)

        Returns
        -------
        None

        Examples
        --------
        """
        expected_keys = {'B', 'gamma', 'wedge_y', 'wedge_x', 'phi', 'theta_e', 'theta_del', 'psi'}

        # Check if all parameters are present.
        missing_keys = expected_keys - set(params.keys())
        if missing_keys:
            warnings.warn("Key(s): {key} not found in params. Skipping setting it.".format(key=missing_keys))

        # Sort the variable names so that it gets correct values assigned
        eta = sorted(self.eta)
        xi = sorted(self.xi)
        y = sorted(self.y)
        x = sorted(self.x)

        # Set the values
        if 'B' not in missing_keys:
            for u, v in self.graph.subgraph(self.eta).edges:
                self.graph.edges[u, v]['weight'] = params['B'][eta.index(v), eta.index(u)]

        if 'gamma' not in missing_keys:
            for u, v in [(u, v) for v in self.eta for u in self.xi]:
                if (u, v) in self.graph.edges:
                    self.graph.edges[u, v]['weight'] = params['gamma'][eta.index(v), xi.index(u)]

        if 'wedge_y' not in missing_keys:
            for u, v in [(u, v) for v in self.y for u in self.eta]:
                if (u, v) in self.graph.edges:
                    self.graph.edges[u, v]['weight'] = params['wedge_y'][y.index(v), eta.index(u)]

        if 'wedge_x' not in missing_keys:
            for u, v in [(u, v) for v in x for u in xi]:
                if (u, v) in self.graph.edges:
                    self.graph.edges[u, v]['weight'] = params['wedge_x'][x.index(v), xi.index(u)]

        if 'phi' not in missing_keys:
            for index, node in enumerate(xi):
                self.err_graph.nodes[node]['var'] = params['phi'][index, index]
            for u, v in self.err_graph.subgraph(xi).edges:
                self.err_graph.edges[u, v]['weight'] = params['phi'][xi.index(v), xi.index(u)]

        if 'psi' not in missing_keys:
            for index, node in enumerate(eta):
                self.err_graph.nodes[node]['var'] = params['psi'][index, index]
            for u, v in self.err_graph.subgraph(eta).edges:
                self.err_graph.edges[u, v]['weight'] = params['psi'][eta.index(v), eta.index(u)]

        if 'theta_e' not in missing_keys:
            for index, node in enumerate(y):
                self.err_graph.nodes[node]['var'] = params['theta_e'][index, index]
            for u, v in self.err_graph.subgraph(y).edges:
                self.err_graph.edges[u, v]['weight'] = params['theta_e'][y.index(v), y.index(u)]

        if 'theta_del' not in missing_keys:
            for index, node in enumerate(x):
                self.err_graph.node[node]['var'] = params['theta_del'][index, index]
            for u, v in self.err_graph.subgraph(x).edges:
                self.err_graph.edges[u, v]['weight'] = params['theta_del'][x.index(v), x.index(u)]
