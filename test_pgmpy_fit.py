from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1]
})

# Define model: A → B
model = DiscreteBayesianNetwork([('A', 'B')])

# Fit model
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Print CPDs
cpds = model.get_cpds()
print(f"Number of CPDs: {len(cpds)}")
for cpd in cpds:
    print(cpd)
