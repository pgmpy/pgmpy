import os
import time

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.utils import get_example_model

# os.environ["NUMEXPR_MAX_THREADS"] = "1"


def time_op(name, fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} sec")
    return result


# -----------------------------
# 1. Microbenchmarks on core ops
# -----------------------------
print("\n--- DiscreteFactor micro-benchmarks ---")

f1 = DiscreteFactor([f"X_{i}" for i in range(10)], [2] * 10, np.arange(2**10) / (2**10))

time_op("product", lambda: f1 * f1)
time_op("reduce", lambda: f1.reduce([("X_1", 1)], inplace=False))
time_op("marginalize", lambda: f1.marginalize(["X_3", "X_4"], inplace=False))

# -----------------------------
# 2. Inference benchmark
# -----------------------------
print("\n--- Inference benchmark (Variable Elimination) ---")

model = get_example_model("munin1")
infer = VariableElimination(model)

print("Running inference.query...")
time_op(
    "inference.query",
    lambda: infer.query(variables=["R_APB_QUAL_MUPDUR"], show_progress=False),
)

print("Simulation Benchmark")
# Simulation
time_op("Simulation", lambda: model.simulate(int(1e3), show_progress=False))
