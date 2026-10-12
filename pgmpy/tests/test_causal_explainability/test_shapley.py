import unittest

from pgmpy.causal_explainability._shapley import ShapleyEngine


class TestShapleyEngineExact(unittest.TestCase):
    def test_additive_game(self):
        weights = {0: 3.0, 1: 7.0, 2: 5.0}

        def value_fn(coalition):
            return sum(weights[i] for i in coalition)

        engine = ShapleyEngine(n_players=3, value_function=value_fn, method="exact")
        result = engine.compute()
        for i in range(3):
            self.assertAlmostEqual(result[i], weights[i], places=10)

    def test_efficiency_axiom(self):
        def value_fn(coalition):
            return len(coalition) ** 2

        engine = ShapleyEngine(n_players=4, value_function=value_fn, method="exact")
        result = engine.compute()
        total = sum(result.values())
        grand_value = value_fn(frozenset(range(4)))
        self.assertAlmostEqual(total, grand_value, places=10)

    def test_symmetry_axiom(self):
        def value_fn(coalition):
            return float(len(coalition))

        engine = ShapleyEngine(n_players=3, value_function=value_fn, method="exact")
        result = engine.compute()
        self.assertAlmostEqual(result[0], result[1], places=10)
        self.assertAlmostEqual(result[1], result[2], places=10)

    def test_null_player(self):
        def value_fn(coalition):
            return sum(1.0 for i in coalition if i != 2)

        engine = ShapleyEngine(n_players=3, value_function=value_fn, method="exact")
        result = engine.compute()
        self.assertAlmostEqual(result[2], 0.0, places=10)

    def test_single_player(self):
        def value_fn(coalition):
            return 10.0 if coalition else 0.0

        engine = ShapleyEngine(n_players=1, value_function=value_fn, method="exact")
        result = engine.compute()
        self.assertAlmostEqual(result[0], 10.0, places=10)

    def test_caching(self):
        call_count = [0]

        def value_fn(coalition):
            call_count[0] += 1
            return float(len(coalition))

        engine = ShapleyEngine(n_players=3, value_function=value_fn, method="exact")
        engine.compute()
        self.assertLessEqual(call_count[0], 2**3)


class TestShapleyEngineSampling(unittest.TestCase):
    def test_additive_game_sampling_converges(self):
        weights = {0: 3.0, 1: 7.0, 2: 5.0}

        def value_fn(coalition):
            return sum(weights[i] for i in coalition)

        engine = ShapleyEngine(n_players=3, value_function=value_fn, method="sampling")
        result = engine.compute(n_permutations=5000, seed=42)
        for i in range(3):
            self.assertAlmostEqual(result[i], weights[i], places=1)

    def test_efficiency_sampling(self):
        def value_fn(coalition):
            return len(coalition) ** 2

        engine = ShapleyEngine(n_players=4, value_function=value_fn, method="sampling")
        result = engine.compute(n_permutations=5000, seed=42)
        total = sum(result.values())
        grand_value = value_fn(frozenset(range(4)))
        self.assertAlmostEqual(total, grand_value, places=0)


class TestShapleyEngineCausalOrdering(unittest.TestCase):
    def test_causal_ordering_changes_values(self):
        def value_fn(coalition):
            if 0 in coalition and 1 in coalition:
                return 10.0
            elif 0 in coalition:
                return 6.0
            elif 1 in coalition:
                return 2.0
            return 0.0

        engine_sym = ShapleyEngine(n_players=2, value_function=value_fn, method="exact")
        result_sym = engine_sym.compute()

        engine_asym = ShapleyEngine(n_players=2, value_function=value_fn, causal_ordering=[{0}, {1}], method="exact")
        result_asym = engine_asym.compute()

        self.assertAlmostEqual(sum(result_sym.values()), 10.0, places=10)
        self.assertAlmostEqual(sum(result_asym.values()), 10.0, places=10)
        self.assertNotAlmostEqual(result_sym[0], result_asym[0], places=5)

    def test_causal_ordering_efficiency(self):
        def value_fn(coalition):
            return sum(i + 1 for i in coalition) + len(coalition) ** 2

        engine = ShapleyEngine(n_players=3, value_function=value_fn, causal_ordering=[{0}, {1}, {2}], method="exact")
        result = engine.compute()
        total = sum(result.values())
        grand_value = value_fn(frozenset(range(3)))
        self.assertAlmostEqual(total, grand_value, places=10)

    def test_auto_method_selection(self):
        def value_fn(coalition):
            return float(len(coalition))

        engine = ShapleyEngine(n_players=20, value_function=value_fn, method="auto")
        self.assertEqual(engine._resolved_method, "sampling")

        engine2 = ShapleyEngine(n_players=10, value_function=value_fn, method="auto")
        self.assertEqual(engine2._resolved_method, "exact")
