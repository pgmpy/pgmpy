from pgmpy.factors.discrete import DiscreteFactor
import numpy as np

# Test case 1: Basic 2x2 factor
print("Test Case 1: Basic 2x2 factor")
factor1 = DiscreteFactor(['Nags', 'Ankur'], [2, 2], np.ones(4))
print(factor1)
print("\n" + "="*50 + "\n")

# Test case 2: Different cardinalities
print("Test Case 2: Different cardinalities")
factor2 = DiscreteFactor(['Bob', 'Oggy'], [2, 3], np.ones(6))
print(factor2)
print("\n" + "="*50 + "\n")

# Test case 3: Different values
print("Test Case 3: Different values")
factor3 = DiscreteFactor(['A', 'B'], [2, 2], [0.1, 0.2, 0.3, 0.4])
print(factor3)
print("\n" + "="*50 + "\n")

# Test case 4: Single variable
print("Test Case 4: Single variable")
factor4 = DiscreteFactor(['A'], [3], [0.1, 0.2, 0.3])
print(factor4)
print("\n" + "="*50 + "\n")

# Test case 5: Three variables
print("Test Case 5: Three variables")
factor5 = DiscreteFactor(['A', 'B', 'C'], [2, 2, 2], np.ones(8))
print(factor5)
print("\n" + "="*50 + "\n") 