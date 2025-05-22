from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd

# Step 1: Create sample data
data = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 1, 0, 1]})

# Step 2: Define model structure
model = BayesianModel([("A", "B")])  # A → B

# Step 3: Fit model with data
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Step 4: Check if CPDs are populated
cpds = model.get_cpds()
print(f"Number of CPDs: {len(cpds)}\n")

# Step 5: Print the CPDs
for cpd in cpds:
    print(cpd)

# Step 6: Validate the model
print("\nModel check:", model.check_model())
