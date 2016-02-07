"""
===================================================
Price Predicting using pgmpy and BayesianModel
===================================================
For simplicity we will consider that the price of the house depends only on Area, Location, Furnishing,
Crime Rate and Distance from the airport.  And also we will consider that all of these are discrete variables.
"""
from pgmpy.models import BayesianModel
import pandas as pd
import numpy as np
data = pd.DataFrame(raw_data, columns=['A', 'C', 'D', 'L', 'F', 'P'])
"""
A-AREA
C-CRIME RATE
D-DISTANCE
L-LOCATION
F-FURNISHING
P-PRICE
"""
data_train = data[:data.shape[0] * 0.75]
model = BayesianModel([('F', 'P'), ('A', 'P'), ('L', 'P'), ('C', 'L'), ('D', 'L')])#Different relation between attributes
model.fit(data_train)#fit method adds a Conditional Probability Distribution (CPD) to each of the node in the model
model.get_cpds()
#Let’s say the probability of getting an unfurnished home is equal to getting a furnished house
from pgmpy.factors import TabularCPD
f_cpd = TabularCPD('F', 2, [[0.5], [0.5]])
model.remove_cpd('F')
model.add_cpd(f_cpd)
model.check_model()
#do some reasoning on our model to verify if our intuitution for the model was correct or not
from pgmpy.Inference import VariableElimination
model = VariableElimination(model)
# Returns a probability distribution over variables A and B.
model.query(variables=['A', 'B'])
model.predict(data[0.75 * data.shape[0] : data.shape[0]])


