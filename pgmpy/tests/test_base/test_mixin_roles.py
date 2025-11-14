#!/usr/bin/env python3

from pgmpy.base import DAG  # if #2402 be merged, change DAG to _CoreGraph class.


class Test_GraphRolesMixin:
    def test_init(self):
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["A", "B"]}

        graph = DAG(  # if #2402 be merged, change DAG to _CoreGraph class.
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        assert graph.get_role_dict() == {
            "exposures": ["A"],
            "latents": ["D"],
            "outcomes": ["C"],
            "test_role": ["A", "B"],
        }

    def test_get_role(self):
        """Test the `get_role` method."""
        ...

    def test_get_roles(self):
        """Test the `get_roles` method."""
        ...

    def test_get_role_dict(self):
        """Test the `get_role_dict` method."""
        ...

    def test_has_role(self):
        """Test the `has_role` method."""
        ...

    def test_with_role(self):
        """Test the `with_role` method."""
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]

        graph = DAG()  # if #2402 be merged, change DAG to _CoreGraph class.
        graph.add_edges_from(ebunch=edges)
        graph.exposures = exposures
        graph.outcomes = outcomes
        graph.latents = latents
        graph.with_role("test_role", ["A", "B"], inplace=True)

        assert graph.get_role_dict() == {
            "exposures": ["A"],
            "latents": ["D"],
            "outcomes": ["C"],
            "test_role": ["A", "B"],
        }

    def test_without_role(self):
        """Test the `without_role` method."""
        ...

    def test_is_valid_causal_structure(self):
        """Test the `is_valid_causal_structure` method."""
        ...

    def test_latents(self):
        """Test the `latents` property getter."""
        ...

    def test_latents_setter(self):
        """Test the `latents` property setter."""
        ...

    def test_observed(self):
        """Test the `observed` property getter."""
        ...

    def test_exposures(self):
        """Test the `exposures` property getter."""
        ...

    def test_exposures_setter(self):
        """Test the `exposures` property setter."""
        ...

    def test_outcomes(self):
        """Test the `outcomes` property getter."""
        ...

    def test_outcomes_setter(self):
        """Test the `outcomes` property setter."""
        ...
