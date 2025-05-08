import time
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.utils import get_example_model

def time_op(name, fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    print(f"{name}: {end - start:.6f} sec")
    return result

print("\n--- DiscreteFactor micro-benchmarks ---")
f1 = DiscreteFactor(["X"], [2], [0.1, 0.9])
f2 = DiscreteFactor(["Y"], [2], [0.3, 0.7])
f3 = DiscreteFactor(["X", "Y"], [2, 2], [0.2, 0.8, 0.6, 0.4])

time_op("product", lambda: f1 * f3)
time_op("reduce", lambda: f3.reduce([("Y", 1)], inplace=False))
time_op("marginalize", lambda: f3.marginalize(["Y"], inplace=False))
time_op("normalize", lambda: f3.normalize(inplace=False))

print("\n--- Inference benchmark (Variable Elimination) ---")
model = get_example_model("asia")
infer = VariableElimination(model)
time_op("inference.query", lambda: infer.query(variables=["either"]))
