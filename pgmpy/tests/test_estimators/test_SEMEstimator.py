import unittest

import numpy as np
import pandas as pd
import torch

from pgmpy import config
from pgmpy.estimators import IVEstimator, SEMEstimator
from pgmpy.models import SEM, SEMGraph


class TestSEMEstimator(unittest.TestCase):
    def setUp(self):
        self.custom = SEMGraph(
            ebunch=[("a", "b"), ("b", "c")], latents=[], err_corr=[], err_var={}
        )
        a = np.random.randn(10**3)
        b = a + np.random.normal(loc=0, scale=0.1, size=10**3)
        c = b + np.random.normal(loc=0, scale=0.2, size=10**3)
        self.custom_data = pd.DataFrame({"a": a, "b": b, "c": c})
        self.custom_data -= self.custom_data.mean(axis=0)
        self.custom_lisrel = self.custom.to_lisrel()

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
                ("y3", "y7"),
                ("y4", "y8"),
                ("y2", "y4"),
                ("y6", "y8"),
            ],
            err_var={},
        )
        self.demo_lisrel = self.demo.to_lisrel()

        self.demo_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/democracy1989a.csv",
            index_col=0,
            header=0,
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
            err_var={},
        )
        self.union_lisrel = self.union.to_lisrel()

        self.union_data = pd.read_csv(
            "pgmpy/tests/test_estimators/testdata/union1989b.csv", index_col=0, header=0
        )

    @unittest.skipIf(config.BACKEND == "numpy", "backend is numpy")
    def test_get_init_values(self):
        demo_estimator = SEMEstimator(self.demo)
        for method in ["random", "std"]:
            B_init, zeta_init = demo_estimator.get_init_values(
                data=self.demo_data, method=method
            )

            demo_lisrel = self.demo.to_lisrel()
            m = len(demo_lisrel.eta)
            self.assertEqual(B_init.shape, (m, m))
            self.assertEqual(zeta_init.shape, (m, m))

            union_estimator = SEMEstimator(self.union)
            B_init, zeta_init = union_estimator.get_init_values(
                data=self.union_data, method=method
            )
            union_lisrel = self.union.to_lisrel()
            m = len(union_lisrel.eta)
            self.assertEqual(B_init.shape, (m, m))
            self.assertEqual(zeta_init.shape, (m, m))

    @unittest.skip
    def test_demo_estimator_random_init(self):
        estimator = SEMEstimator(self.demo)
        summary = estimator.fit(self.demo_data, method="ml")

    @unittest.skip
    def test_union_estimator_random_init(self):
        estimator = SEMEstimator(self.union_lisrel)
        summary = estimator.fit(
            self.union_data, method="ml", opt="adam", max_iter=10**6, exit_delta=1e-1
        )

    @unittest.skip
    def test_custom_estimator_random_init(self):
        estimator = SEMEstimator(self.custom_lisrel)
        summary = estimator.fit(
            self.custom_data, method="ml", max_iter=10**6, opt="adam"
        )
        summary = estimator.fit(
            self.custom_data, method="uls", max_iter=10**6, opt="adam"
        )
        summary = estimator.fit(
            self.custom_data,
            method="gls",
            max_iter=10**6,
            opt="adam",
            W=np.ones((3, 3)),
        )

    @unittest.skip
    def test_union_estimator_std_init(self):
        estimator = SEMEstimator(self.union_lisrel)
        summary = estimator.fit(
            self.union_data,
            method="ml",
            opt="adam",
            init_values="std",
            max_iter=10**6,
            exit_delta=1e-1,
        )

    @unittest.skip
    def test_custom_estimator_std_init(self):
        estimator = SEMEstimator(self.custom_lisrel)
        summary = estimator.fit(
            self.custom_data,
            method="ml",
            init_values="std",
            max_iter=10**6,
            opt="adam",
        )


class TestIVEstimator(unittest.TestCase):
    def setUp(self):
        self.model = SEM.from_graph(
            ebunch=[
                ("Z1", "X", 1.0),
                ("Z2", "X", 1.0),
                ("Z2", "W", 1.0),
                ("W", "U", 1.0),
                ("U", "X", 1.0),
                ("U", "Y", 1.0),
                ("X", "Y", 1.0),
            ],
            latents=["U"],
            err_var={"Z1": 1, "Z2": 1, "W": 1, "X": 1, "U": 1, "Y": 1},
        )
        self.generated_data = self.model.to_lisrel().generate_samples(100000)

    def test_fit(self):
        estimator = IVEstimator(self.model)
        param, summary = estimator.fit(X="X", Y="Y", data=self.generated_data)
        self.assertTrue((param - 1) < 0.027)
