import sys

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

# 1. Setup a simple Linear Gaussian Bayesian Network: X -> Y -> Z
model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
cpd_x = LinearGaussianCPD("X", [0], 1)
cpd_y = LinearGaussianCPD("Y", [0, 2], 1, ["X"])
# Z = 3*Y + 0
cpd_z = LinearGaussianCPD("Z", [0, 3], 1, ["Y"])
model.add_cpds(cpd_x, cpd_y, cpd_z)

print("Original Model Valid?", model.check_model())

try:
    # 2. Attempt formal intervention on Y: do(Y)
    # The base DAG.do() removes the incoming edges but blindly copies the CPDs
    intervened_model = model.do(["Y"])

    # 3. Check if the returned object is structurally valid
    intervened_model.check_model()

except ValueError as e:
    print(f"\n[Structural Validity Error] {e}")
    print(
        "Explanation: The base DAG.do() severed the edges, but didn't update or marginalize "
        "the LinearGaussianCPD parameters. The model is essentially corrupted and cannot be used."
    )
    sys.exit(0)

print("\n[!] Unexpectedly succeeded.")
