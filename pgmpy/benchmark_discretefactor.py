import time
import tracemalloc
import sys
import numpy as np
from functools import partial
from timeit import repeat
import logging
from pympler import asizeof  # for deep memory inspection
from pgmpy.factors.discrete import DiscreteFactor as DF_SLOTS
from pgmpy.inference import VariableElimination
from pgmpy.utils import get_example_model

logging.getLogger("pgmpy").setLevel(logging.ERROR)

print("\n=== pgmpy Realistic Benchmark Suite ===\n")


# Helper: Non-slotted version
class BaseFactorNoSlots:
    def __init__(self):
        self.variables = None
        self.cardinality = None
        self.dtype = None
        self.state_names = None


class StateNameMixinNoSlots:
    def __init__(self):
        self.name_to_no = None
        self.no_to_name = None


class DiscreteFactorNoSlots(BaseFactorNoSlots, StateNameMixinNoSlots):
    def __init__(self, variables, cardinality, values, state_names=None):
        BaseFactorNoSlots.__init__(self)
        StateNameMixinNoSlots.__init__(self)
        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.values = np.array(values).reshape(tuple(self.cardinality))
        self.state_names = state_names or {
            var: list(range(card)) for var, card in zip(variables, cardinality)
        }
        self.name_to_no = {
            var: {name: i for i, name in enumerate(states)}
            for var, states in self.state_names.items()
        }
        self.no_to_name = {
            var: {i: name for i, name in enumerate(states)}
            for var, states in self.state_names.items()
        }

    def reduce(self, var_vals, inplace=False):
        slicer = [slice(None)] * len(self.variables)
        for var, val in var_vals:
            slicer[self.variables.index(var)] = val
        reduced_values = self.values[tuple(slicer)]
        reduced_vars = [v for v, _ in var_vals]
        new_variables = [v for v in self.variables if v not in reduced_vars]
        new_cardinality = [
            self.cardinality[i]
            for i, v in enumerate(self.variables)
            if v not in reduced_vars
        ]
        return DiscreteFactorNoSlots(new_variables, new_cardinality, reduced_values)

    def marginalize(self, vars_to_marginalize, inplace=False):
        axes = tuple(self.variables.index(var) for var in vars_to_marginalize)
        marginalized_values = self.values.sum(axis=axes)
        new_variables = [v for v in self.variables if v not in vars_to_marginalize]
        new_cardinality = [
            self.cardinality[i]
            for i, v in enumerate(self.variables)
            if v not in vars_to_marginalize
        ]
        return DiscreteFactorNoSlots(
            new_variables, new_cardinality, marginalized_values
        )

    def __mul__(self, other):
        all_vars = sorted(set(self.variables + other.variables))
        all_card = [
            (
                self.cardinality[self.variables.index(v)]
                if v in self.variables
                else other.cardinality[other.variables.index(v)]
            )
            for v in all_vars
        ]

        self_val = self.values
        other_val = other.values

        for v in all_vars:
            if v not in self.variables:
                self_val = np.expand_dims(self_val, axis=0)
            if v not in other.variables:
                other_val = np.expand_dims(other_val, axis=0)

        result = self_val * other_val
        return DiscreteFactorNoSlots(all_vars, all_card, result)


# -----------------------------
# Benchmark Function
# -----------------------------
def benchmark(label, DF):
    print(f"\n=== {label} ===")

    # 1. Attribute Benchmark
    print("\n--- Attribute Set/Get Benchmark ---")

    def get_set_delete(factor):
        factor.values = np.ones((2, 2))
        _ = factor.values

    factor = DF(["X", "Y"], [2, 2], np.random.rand(4))
    attr_time = min(repeat(partial(get_set_delete, factor), number=100_000))
    print(f"Set/Get/Delete time (100K ops): {attr_time:.6f} sec")

    # 2. Instance Creation + Memory
    print("\n--- DiscreteFactor Instance Creation Benchmark ---")

    def benchmark_creation(n=100_000):
        tracemalloc.start()
        start = time.perf_counter()
        factors = [DF(["X", "Y"], [2, 2], np.random.rand(4)) for _ in range(n)]
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Creation time: {end - start:.4f} sec")
        print(f"Peak memory: {peak / 1024:.2f} KB")

    benchmark_creation()

    # 3. Core Ops
    print("\n--- Inference & Factor Ops Benchmark ---")
    f1 = DF([f"X_{i}" for i in range(10)], [2] * 10, np.arange(2**10) / (2**10))

    def time_op(label, fn, repeat_count=1):
        start = time.perf_counter()
        for _ in range(repeat_count):
            fn()
        end = time.perf_counter()
        print(f"{label}: {end - start:.6f} sec")

    try:
        time_op("product", lambda: f1 * f1)
        time_op("reduce", lambda: f1.reduce([("X_1", 1)], inplace=False))
        time_op("marginalize", lambda: f1.marginalize(["X_3", "X_4"], inplace=False))
    except Exception as e:
        print(f"[Skipped core ops due to: {e}]")

    # 4. Full model inference (slotted version only)
    if DF is DF_SLOTS:
        print("\n--- Full Model Simulation ---")
        model = get_example_model("munin1")
        infer = VariableElimination(model)
        time_op(
            "inference.query",
            lambda: infer.query(variables=["R_APB_QUAL_MUPDUR"], show_progress=False),
        )
        time_op("simulate(1000)", lambda: model.simulate(1000, show_progress=False))

    # 5. Memory and Attribute Size
    print("\n--- Memory and Attribute Analysis ---")
    df = DF(["X", "Y"], [2, 2], [0.1, 0.2, 0.3, 0.4])
    print(f"asizeof.asizeof (deep): {asizeof.asizeof(df)} bytes")

    # Attribute inspection
    if hasattr(df, "__dict__"):
        print(f"Uses __dict__:(non-slotted)")
        print(f"Number of attributes: {len(df.__dict__)}")
        print("Attributes:", list(df.__dict__.keys()))
    else:
        print(f"Uses __dict__:(slotted)")
        slot_attrs = set()
        for cls in df.__class__.mro():
            if "__slots__" in cls.__dict__:
                slots = cls.__dict__["__slots__"]
                if isinstance(slots, str):
                    slot_attrs.add(slots)
                else:
                    slot_attrs.update(slots)
        print(f"Number of __slots__ attributes: {len(slot_attrs)}")
        print("Slots:", slot_attrs)


# -----------------------------
# Run Benchmarks
# -----------------------------
benchmark("Benchmark With __slots__ (Real pgmpy)", DF_SLOTS)
benchmark("Benchmark Without __slots__ (Simulated Class)", DiscreteFactorNoSlots)
