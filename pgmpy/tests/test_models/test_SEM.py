import os

import networkx as nx
import numpy as np
import numpy.testing as npt
import pytest

from pgmpy.models import SEM, SEMGraph


class TestSEM:
    def test_from_graph(self):
        self.demo = SEM.from_graph(
            ebunch=[
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "x3"),
                ("xi1", "eta1"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
                ("eta1", "y4"),
                ("eta1", "eta2"),
                ("xi1", "eta2"),
                ("eta2", "y5"),
                ("eta2", "y6"),
                ("eta2", "y7"),
                ("eta2", "y8"),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5"),
                ("y2", "y6"),
                ("y2", "y4"),
                ("y3", "y7"),
                ("y4", "y8"),
                ("y6", "y8"),
            ],
        )

        assert self.demo.latents == {"xi1", "eta1", "eta2"}
        assert self.demo.observed == {"x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"}
        assert sorted(self.demo.nodes()) == [
            "eta1",
            "eta2",
            "x1",
            "x2",
            "x3",
            "xi1",
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
        ]
        assert sorted(self.demo.edges()) == sorted(
            [
                ("eta1", "eta2"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
                ("eta1", "y4"),
                ("eta2", "y5"),
                ("eta2", "y6"),
                ("eta2", "y7"),
                ("eta2", "y8"),
                ("xi1", "eta1"),
                ("xi1", "eta2"),
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "x3"),
            ]
        )

        assert self.demo.edges[("xi1", "x1")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "x2")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "x3")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "eta1")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y1")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y2")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y3")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y4")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "eta2")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "eta2")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y5")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y6")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y7")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y8")] == {"weight": np.nan}

        npt.assert_equal(
            nx.to_numpy_array(
                self.demo.err_graph,
                nodelist=sorted(self.demo.err_graph.nodes()),
                weight=None,
            ),
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
        )

        for edge in self.demo.err_graph.edges():
            assert self.demo.err_graph.edges[edge] == {"weight": np.nan}
        for node in self.demo.err_graph.nodes():
            assert self.demo.err_graph.nodes[node] == {"weight": np.nan}

    def test_from_lavaan(self):
        model_str = """# %load model.lav
                       # measurement model
                         ind60 =~ x1 + x2 + x3
                         dem60 =~ y1 + y2 + y3 + y4
                         dem65 =~ y5 + y6 + y7 + y8
                       # regressions
                         dem60 ~ ind60
                         dem65 ~ ind60 + dem60
                       # residual correlations
                         y1 ~~ y5
                         y2 ~~ y4 + y6
                         y3 ~~ y7
                         y4 ~~ y8
                         y6 ~~ y8
                       """
        model_from_str = SEM.from_lavaan(string=model_str)

        with open("test_model.lav", "w") as f:
            f.write(model_str)
        model_from_file = SEM.from_lavaan(filename="test_model.lav")
        os.remove("test_model.lav")

        expected_edges = {
            ("ind60", "x1"),
            ("ind60", "x2"),
            ("ind60", "x3"),
            ("ind60", "dem60"),
            ("ind60", "dem65"),
            ("dem60", "dem65"),
            ("dem60", "y1"),
            ("dem60", "y2"),
            ("dem60", "y3"),
            ("dem60", "y4"),
            ("dem65", "y5"),
            ("dem65", "y6"),
            ("dem65", "y7"),
            ("dem65", "y8"),
        }

        # Undirected Graph, needs to handle when edges returned in reverse.
        expected_err_edges = {
            ("y1", "y5"),
            ("y5", "y1"),
            ("y2", "y6"),
            ("y6", "y2"),
            ("y2", "y4"),
            ("y4", "y2"),
            ("y3", "y7"),
            ("y7", "y3"),
            ("y4", "y8"),
            ("y8", "y4"),
            ("y6", "y8"),
            ("y8", "y6"),
        }

        expected_latents = {"dem60", "dem65", "ind60"}
        assert set(model_from_str.edges()) == expected_edges
        assert set(model_from_file.edges()) == expected_edges
        assert not set(model_from_str.err_graph.edges()) - expected_err_edges
        assert not set(model_from_file.err_graph.edges()) - expected_err_edges
        assert set(model_from_str.latents) == expected_latents
        assert set(model_from_file.latents) == expected_latents

    def test_from_lisrel(self):
        pass  # TODO: Add this test when done writing the tests for SEMAlg

    def test_from_ram(self):
        pass  # TODO: Add this.


class TestSEMGraph:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.demo = SEMGraph(
            ebunch=[
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "x3"),
                ("xi1", "eta1"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
                ("eta1", "y4"),
                ("eta1", "eta2"),
                ("xi1", "eta2"),
                ("eta2", "y5"),
                ("eta2", "y6"),
                ("eta2", "y7"),
                ("eta2", "y8"),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5"),
                ("y2", "y6"),
                ("y2", "y4"),
                ("y3", "y7"),
                ("y4", "y8"),
                ("y6", "y8"),
            ],
        )

        self.union = SEMGraph(
            ebunch=[
                ("yrsmill", "unionsen"),
                ("age", "laboract"),
                ("age", "deferenc"),
                ("deferenc", "laboract"),
                ("deferenc", "unionsen"),
                ("laboract", "unionsen"),
            ],
            latents=[],
            err_corr=[("yrsmill", "age")],
        )

        self.demo_params = SEMGraph(
            ebunch=[
                ("xi1", "x1", 0.4),
                ("xi1", "x2", 0.5),
                ("xi1", "x3", 0.6),
                ("xi1", "eta1", 0.3),
                ("eta1", "y1", 1.1),
                ("eta1", "y2", 1.2),
                ("eta1", "y3", 1.3),
                ("eta1", "y4", 1.4),
                ("eta1", "eta2", 0.1),
                ("xi1", "eta2", 0.2),
                ("eta2", "y5", 0.7),
                ("eta2", "y6", 0.8),
                ("eta2", "y7", 0.9),
                ("eta2", "y8", 1.0),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5", 1.5),
                ("y2", "y6", 1.6),
                ("y2", "y4", 1.9),
                ("y3", "y7", 1.7),
                ("y4", "y8", 1.8),
                ("y6", "y8", 2.0),
            ],
            err_var={
                "y1": 2.1,
                "y2": 2.2,
                "y3": 2.3,
                "y4": 2.4,
                "y5": 2.5,
                "y6": 2.6,
                "y7": 2.7,
                "y8": 2.8,
                "x1": 3.1,
                "x2": 3.2,
                "x3": 3.3,
                "eta1": 2.9,
                "eta2": 3.0,
                "xi1": 3.4,
            },
        )

        self.custom = SEMGraph(
            ebunch=[
                ("xi1", "eta1"),
                ("xi1", "y1"),
                ("xi1", "y4"),
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("y4", "y1"),
                ("y1", "eta2"),
                ("eta2", "y5"),
                ("y1", "eta1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[("y1", "y2"), ("y2", "y3")],
            err_var={},
        )

    def test_demo_init(self):
        assert self.demo.latents == {"xi1", "eta1", "eta2"}
        assert self.demo.observed == {"x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"}
        assert sorted(self.demo.nodes()) == [
            "eta1",
            "eta2",
            "x1",
            "x2",
            "x3",
            "xi1",
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
        ]
        assert sorted(self.demo.edges()) == sorted(
            [
                ("eta1", "eta2"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("eta1", "y3"),
                ("eta1", "y4"),
                ("eta2", "y5"),
                ("eta2", "y6"),
                ("eta2", "y7"),
                ("eta2", "y8"),
                ("xi1", "eta1"),
                ("xi1", "eta2"),
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "x3"),
            ]
        )

        assert self.demo.edges[("xi1", "x1")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "x2")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "x3")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "eta1")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y1")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y2")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y3")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "y4")] == {"weight": np.nan}
        assert self.demo.edges[("eta1", "eta2")] == {"weight": np.nan}
        assert self.demo.edges[("xi1", "eta2")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y5")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y6")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y7")] == {"weight": np.nan}
        assert self.demo.edges[("eta2", "y8")] == {"weight": np.nan}

        npt.assert_equal(
            nx.to_numpy_array(
                self.demo.err_graph,
                nodelist=sorted(self.demo.err_graph.nodes()),
                weight=None,
            ),
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
        )

        for edge in self.demo.err_graph.edges():
            assert self.demo.err_graph.edges[edge] == {"weight": np.nan}
        for node in self.demo.err_graph.nodes():
            assert self.demo.err_graph.nodes[node] == {"weight": np.nan}

    def test_union_init(self):
        assert self.union.latents == set()
        assert self.union.observed == {"yrsmill", "unionsen", "age", "laboract", "deferenc"}

        assert sorted(self.union.nodes()) == sorted(["yrsmill", "unionsen", "age", "laboract", "deferenc"])
        assert sorted(self.union.edges()) == sorted(
            [
                ("yrsmill", "unionsen"),
                ("age", "laboract"),
                ("age", "deferenc"),
                ("deferenc", "laboract"),
                ("deferenc", "unionsen"),
                ("laboract", "unionsen"),
            ]
        )

        assert self.union.edges[("yrsmill", "unionsen")] == {"weight": np.nan}
        assert self.union.edges[("age", "laboract")] == {"weight": np.nan}
        assert self.union.edges[("age", "deferenc")] == {"weight": np.nan}
        assert self.union.edges[("deferenc", "laboract")] == {"weight": np.nan}
        assert self.union.edges[("deferenc", "unionsen")] == {"weight": np.nan}
        assert self.union.edges[("laboract", "unionsen")] == {"weight": np.nan}

        npt.assert_equal(
            nx.to_numpy_array(
                self.union.err_graph,
                nodelist=sorted(self.union.err_graph.nodes()),
                weight=None,
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        for edge in self.union.err_graph.edges():
            assert self.union.err_graph.edges[edge] == {"weight": np.nan}
        for node in self.union.err_graph.nodes():
            assert self.union.err_graph.nodes[node] == {"weight": np.nan}

    def test_demo_param_init(self):
        assert self.demo_params.edges[("xi1", "x1")] == {"weight": 0.4}
        assert self.demo_params.edges[("xi1", "x2")] == {"weight": 0.5}
        assert self.demo_params.edges[("xi1", "x3")] == {"weight": 0.6}
        assert self.demo_params.edges[("xi1", "eta1")] == {"weight": 0.3}
        assert self.demo_params.edges[("eta1", "y1")] == {"weight": 1.1}
        assert self.demo_params.edges[("eta1", "y2")] == {"weight": 1.2}
        assert self.demo_params.edges[("eta1", "y3")] == {"weight": 1.3}
        assert self.demo_params.edges[("eta1", "y4")] == {"weight": 1.4}
        assert self.demo_params.edges[("eta1", "eta2")] == {"weight": 0.1}
        assert self.demo_params.edges[("xi1", "eta2")] == {"weight": 0.2}
        assert self.demo_params.edges[("eta2", "y5")] == {"weight": 0.7}
        assert self.demo_params.edges[("eta2", "y6")] == {"weight": 0.8}
        assert self.demo_params.edges[("eta2", "y7")] == {"weight": 0.9}
        assert self.demo_params.edges[("eta2", "y8")] == {"weight": 1.0}

        assert self.demo_params.err_graph.edges[("y1", "y5")] == {"weight": 1.5}
        assert self.demo_params.err_graph.edges[("y2", "y6")] == {"weight": 1.6}
        assert self.demo_params.err_graph.edges[("y2", "y4")] == {"weight": 1.9}
        assert self.demo_params.err_graph.edges[("y3", "y7")] == {"weight": 1.7}
        assert self.demo_params.err_graph.edges[("y4", "y8")] == {"weight": 1.8}
        assert self.demo_params.err_graph.edges[("y6", "y8")] == {"weight": 2.0}

        assert self.demo_params.err_graph.nodes["y1"] == {"weight": 2.1}
        assert self.demo_params.err_graph.nodes["y2"] == {"weight": 2.2}
        assert self.demo_params.err_graph.nodes["y3"] == {"weight": 2.3}
        assert self.demo_params.err_graph.nodes["y4"] == {"weight": 2.4}
        assert self.demo_params.err_graph.nodes["y5"] == {"weight": 2.5}
        assert self.demo_params.err_graph.nodes["y6"] == {"weight": 2.6}
        assert self.demo_params.err_graph.nodes["y7"] == {"weight": 2.7}
        assert self.demo_params.err_graph.nodes["y8"] == {"weight": 2.8}
        assert self.demo_params.err_graph.nodes["x1"] == {"weight": 3.1}
        assert self.demo_params.err_graph.nodes["x2"] == {"weight": 3.2}
        assert self.demo_params.err_graph.nodes["x3"] == {"weight": 3.3}
        assert self.demo_params.err_graph.nodes["eta1"] == {"weight": 2.9}
        assert self.demo_params.err_graph.nodes["eta2"] == {"weight": 3.0}

    def test_get_full_graph_struct(self):
        full_struct = self.union._get_full_graph_struct()
        assert not (
            set(full_struct.nodes())
            - {
                "yrsmill",
                "unionsen",
                "age",
                "laboract",
                "deferenc",
                ".yrsmill",
                ".unionsen",
                ".age",
                ".laboract",
                ".deferenc",
                "..ageyrsmill",
                "..yrsmillage",
            }
        )
        assert not (
            set(full_struct.edges())
            - {
                ("yrsmill", "unionsen"),
                ("age", "laboract"),
                ("age", "deferenc"),
                ("deferenc", "laboract"),
                ("deferenc", "unionsen"),
                ("laboract", "unionsen"),
                (".yrsmill", "yrsmill"),
                (".unionsen", "unionsen"),
                (".age", "age"),
                (".laboract", "laboract"),
                (".deferenc", "deferenc"),
                ("..ageyrsmill", ".age"),
                ("..ageyrsmill", ".yrsmill"),
                ("..yrsmillage", ".age"),
                ("..yrsmillage", ".yrsmill"),
            }
        )

    def test_active_trail_nodes(self):
        demo_nodes = ["x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
        for node in demo_nodes:
            assert self.demo.active_trail_nodes(node, struct="full")[node] == set(demo_nodes)

        union_nodes = self.union.nodes()
        active_trails = self.union.active_trail_nodes(list(union_nodes), struct="full")
        for node in union_nodes:
            assert active_trails[node] == set(union_nodes)

        assert self.union.active_trail_nodes("age", observed=["laboract", "deferenc", "unionsen"])["age"] == {
            "age",
            "yrsmill",
        }

    def test_to_lisrel(self):
        demo = SEMGraph(
            ebunch=[
                ("xi1", "x1", 1.000),
                ("xi1", "x2", 2.180),
                ("xi1", "x3", 1.819),
                ("xi1", "eta1", 1.483),
                ("eta1", "y1", 1.000),
                ("eta1", "y2", 1.257),
                ("eta1", "y3", 1.058),
                ("eta1", "y4", 1.265),
                ("eta1", "eta2", 0.837),
                ("xi1", "eta2", 0.572),
                ("eta2", "y5", 1.000),
                ("eta2", "y6", 1.186),
                ("eta2", "y7", 1.280),
                ("eta2", "y8", 1.266),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5", 0.624),
                ("y2", "y6", 2.153),
                ("y2", "y4", 1.313),
                ("y3", "y7", 0.795),
                ("y4", "y8", 0.348),
                ("y6", "y8", 1.356),
            ],
            err_var={
                "x1": 0.082,
                "x2": 0.120,
                "x3": 0.467,
                "y1": 1.891,
                "y2": 7.373,
                "y3": 5.067,
                "y4": 3.148,
                "y5": 2.351,
                "y6": 4.954,
                "y7": 3.431,
                "y8": 3.254,
                "xi1": 0.448,
                "eta1": 3.956,
                "eta2": 0.172,
            },
        )
        demo_lisrel = demo.to_lisrel()

        indexing = []
        vars_ordered = [
            "y1",
            "y2",
            "y3",
            "y4",
            "y5",
            "y6",
            "y7",
            "y8",
            "x1",
            "x2",
            "x3",
            "xi1",
            "eta1",
            "eta2",
        ]
        for var in vars_ordered:
            indexing.append(demo_lisrel.eta.index(var))

        eta_reorder = [demo_lisrel.eta[i] for i in indexing]

        B_reorder = demo_lisrel.B[indexing, :][:, indexing]
        B_fixed_reorder = demo_lisrel.B_fixed_mask[indexing, :][:, indexing]

        zeta_reorder = demo_lisrel.zeta[indexing, :][:, indexing]
        zeta_fixed_reorder = demo_lisrel.zeta_fixed_mask[indexing, :][:, indexing]

        assert vars_ordered == eta_reorder
        npt.assert_array_equal(
            B_reorder,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                ]
            ),
        )

        npt.assert_array_equal(
            zeta_reorder,
            np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        )

        npt.assert_array_equal(
            B_fixed_reorder,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.000, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.257, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.058, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.265, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.000],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.186],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.280],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.266],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.000, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.180, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.819, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.483, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.572, 0.837, 0],
                ]
            ),
        )

        npt.assert_array_equal(
            zeta_fixed_reorder,
            np.array(
                [
                    [1.891, 0, 0, 0, 0.624, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 7.373, 0, 1.313, 0, 2.153, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 5.067, 0, 0, 0, 0.795, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1.313, 0, 3.148, 0, 0, 0, 0.348, 0, 0, 0, 0, 0, 0],
                    [0.624, 0, 0, 0, 2.351, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2.153, 0, 0, 0, 4.954, 0, 1.356, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.795, 0, 0, 0, 3.431, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.348, 0, 1.356, 0, 3.254, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.082, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.120, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.467, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.448, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.956, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.172],
                ]
            ),
        )

        npt.assert_array_equal(
            demo_lisrel.wedge_y,
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                ]
            ),
        )

    def test_to_from_lisrel(self):
        demo_lisrel = self.demo.to_lisrel()
        union_lisrel = self.union.to_lisrel()
        demo_params_lisrel = self.demo_params.to_lisrel()
        custom_lisrel = self.custom.to_lisrel()

        demo_graph = demo_lisrel.to_SEMGraph()
        union_graph = union_lisrel.to_SEMGraph()
        demo_params_graph = demo_params_lisrel.to_SEMGraph()
        custom_graph = custom_lisrel.to_SEMGraph()

        # Test demo
        assert set(self.demo.nodes()) == set(demo_graph.nodes())
        assert set(self.demo.edges()) == set(demo_graph.edges())

        assert set(self.demo.err_graph.nodes()) == set(demo_graph.err_graph.nodes())
        npt.assert_array_equal(
            nx.to_numpy_array(self.demo.err_graph, nodelist=sorted(self.demo.err_graph.nodes())),
            nx.to_numpy_array(demo_graph.err_graph, nodelist=sorted(demo_graph.err_graph.nodes())),
        )

        assert set(self.demo.full_graph_struct.nodes()) == set(demo_graph.full_graph_struct.nodes())
        assert set(self.demo.full_graph_struct.edges()) == set(demo_graph.full_graph_struct.edges())

        assert self.demo.latents == demo_graph.latents
        assert self.demo.observed == demo_graph.observed

        # Test union
        assert set(self.union.nodes()) == set(union_graph.nodes())
        assert set(self.union.edges()) == set(union_graph.edges())

        assert set(self.union.err_graph.nodes()) == set(union_graph.err_graph.nodes())
        npt.assert_array_equal(
            nx.to_numpy_array(self.union.err_graph, nodelist=sorted(self.union.err_graph.nodes())),
            nx.to_numpy_array(union_graph.err_graph, nodelist=sorted(union_graph.err_graph.nodes())),
        )

        assert set(self.union.full_graph_struct.nodes()) == set(union_graph.full_graph_struct.nodes())
        assert set(self.union.full_graph_struct.edges()) == set(union_graph.full_graph_struct.edges())

        assert self.union.latents == union_graph.latents
        assert self.union.observed == union_graph.observed

        # Test demo_params
        assert set(self.demo_params.nodes()) == set(demo_params_graph.nodes())
        assert set(self.demo_params.edges()) == set(demo_params_graph.edges())

        assert set(self.demo_params.err_graph.nodes()) == set(demo_params_graph.err_graph.nodes())
        npt.assert_array_equal(
            nx.to_numpy_array(
                self.demo_params.err_graph,
                nodelist=sorted(self.demo_params.err_graph.nodes()),
                weight=None,
            ),
            nx.to_numpy_array(
                demo_graph.err_graph,
                nodelist=sorted(demo_params_graph.err_graph.nodes()),
                weight=None,
            ),
        )

        assert set(self.demo_params.full_graph_struct.nodes()) == set(demo_params_graph.full_graph_struct.nodes())
        assert set(self.demo_params.full_graph_struct.edges()) == set(demo_params_graph.full_graph_struct.edges())

        assert self.demo_params.latents == demo_params_graph.latents
        assert self.demo_params.observed == demo_params_graph.observed

        # Test demo
        assert set(self.custom.nodes()) == set(custom_graph.nodes())
        assert set(self.custom.edges()) == set(custom_graph.edges())

        assert set(self.custom.err_graph.nodes()) == set(custom_graph.err_graph.nodes())
        npt.assert_array_equal(
            nx.to_numpy_array(self.custom.err_graph, nodelist=sorted(self.custom.err_graph.nodes())),
            nx.to_numpy_array(custom_graph.err_graph, nodelist=sorted(custom_graph.err_graph.nodes())),
        )

        assert set(self.custom.full_graph_struct.nodes()) == set(custom_graph.full_graph_struct.nodes())
        assert set(self.custom.full_graph_struct.edges()) == set(custom_graph.full_graph_struct.edges())

        assert self.custom.latents == custom_graph.latents
        assert self.custom.observed == custom_graph.observed

    def test_graph_methods(self):
        model = SEMGraph(
            ebunch=[
                ("xi1", "x1"),
                ("xi1", "x2"),
                ("xi1", "eta1"),
                ("eta1", "y1"),
                ("eta1", "y2"),
                ("x1", "y1"),
                ("y2", "y3"),
            ],
            latents=["xi1", "eta1"],
            err_corr=[("y1", "y3")],
        )
        assert sorted(map(sorted, model.moralize(graph="non_error").edges())) == [
            ["eta1", "x1"],
            ["eta1", "xi1"],
            ["eta1", "y1"],
            ["eta1", "y2"],
            ["x1", "xi1"],
            ["x1", "y1"],
            ["x2", "xi1"],
            ["y2", "y3"],
        ]
        assert model.active_trail_nodes("x1", struct="non_error") == {"x1": {"x1", "x2", "y1", "y2", "y3"}}
        assert model._nearest_separator(model.full_graph_struct, "x2", "y3") == {"y2"}

        measurement = SEMGraph(
            ebunch=[("xi1", "x1"), ("xi1", "x2"), ("xi1", "eta1"), ("eta1", "y1"), ("eta1", "y2")],
            latents=["xi1", "eta1"],
            err_corr=[("y1", "y2")],
        )
        var_names, edges_masks, _ = measurement.to_standard_lisrel()
        assert {k: sorted(v) for k, v in var_names.items()} == {
            "eta": ["eta1"],
            "xi": ["xi1"],
            "y": ["y1", "y2"],
            "x": ["x1", "x2"],
        }
        assert {k: int(v.sum()) for k, v in edges_masks.items()} == {
            "B": 0,
            "gamma": 1,
            "wedge_y": 2,
            "wedge_x": 2,
            "phi": 1,
            "theta_e": 4,
            "theta_del": 2,
            "psi": 1,
        }


class TestSEMAlg:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.demo = SEMGraph(
            ebunch=[
                ("xi1", "x1", 1.000),
                ("xi1", "x2", 2.180),
                ("xi1", "x3", 1.819),
                ("xi1", "eta1", 1.483),
                ("eta1", "y1", 1.000),
                ("eta1", "y2", 1.257),
                ("eta1", "y3", 1.058),
                ("eta1", "y4", 1.265),
                ("eta1", "eta2", 0.837),
                ("xi1", "eta2", 0.572),
                ("eta2", "y5", 1.000),
                ("eta2", "y6", 1.186),
                ("eta2", "y7", 1.280),
                ("eta2", "y8", 1.266),
            ],
            latents=["xi1", "eta1", "eta2"],
            err_corr=[
                ("y1", "y5", 0.624),
                ("y2", "y6", 2.153),
                ("y2", "y4", 1.313),
                ("y3", "y7", 0.795),
                ("y4", "y8", 0.348),
                ("y6", "y8", 1.356),
            ],
            err_var={
                "x1": 0.082,
                "x2": 0.120,
                "x3": 0.467,
                "y1": 1.891,
                "y2": 7.373,
                "y3": 5.067,
                "y4": 3.148,
                "y5": 2.351,
                "y6": 4.954,
                "y7": 3.431,
                "y8": 3.254,
                "xi1": 0.448,
                "eta1": 3.956,
                "eta2": 0.172,
            },
        )
        self.demo_lisrel = self.demo.to_lisrel()

        self.small_model = SEM.from_graph(ebunch=[("X", "Y", 0.3)], latents=[], err_var={"X": 0.1, "Y": 0.1})
        self.small_model_lisrel = self.small_model.to_lisrel()

    def test_generate_samples(self):
        self.small_model_lisrel.generate_samples(n_samples=100)
        self.demo_lisrel.generate_samples(n_samples=100)


class TestSEMGraphRoles:
    ebunch = [("z", "x"), ("x", "y", 0.5), ("l", "y")]

    def test_roles_init(self):
        model = SEMGraph(
            ebunch=self.ebunch,
            latents=["l"],
            exposures={"x"},
            outcomes={"y"},
            roles={"adjustment": "z"},
        )
        assert model.latents == {"l"}
        assert model.observed == {"x", "y", "z"}
        assert model.exposures == {"x"}
        assert model.outcomes == {"y"}
        assert model.get_role("adjustment") == ["z"]
        assert model.nodes["x"]["roles"] == {"exposures"}

        with pytest.raises(ValueError):
            SEMGraph(ebunch=self.ebunch, exposures={"missing"})
        with pytest.raises(TypeError):
            SEMGraph(ebunch=self.ebunch, roles=["z"])

    @pytest.mark.parametrize("cls", [SEMGraph, SEM.from_graph])
    def test_with_role_not_inplace(self, cls):
        model = cls(ebunch=self.ebunch, latents=["l"], err_corr=[("x", "y", 0.2)], err_var={"x": 1.0})
        new_model = model.with_role("exposures", "x", inplace=False)

        assert type(new_model) is type(model)
        assert new_model.exposures == {"x"}
        assert model.exposures == set()
        assert new_model.latents == {"l"}
        assert new_model.edges[("x", "y")] == {"weight": 0.5}
        assert new_model.err_graph.edges[("x", "y")] == {"weight": 0.2}
        assert new_model.err_graph.nodes["x"] == {"weight": 1.0}
        assert set(new_model.full_graph_struct.edges()) == set(model.full_graph_struct.edges())

        new_model.without_role("exposures", "x", inplace=True)
        assert new_model.exposures == set()
