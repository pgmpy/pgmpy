import pandas as pd
from pgmpy.inference.igci import IGCI

# Dummy data
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 6, 8, 10]
})

igci = IGCI()
igci.fit(data)
direction, score = igci.estimate_direction('X', 'Y')

print("Direction:", direction)
print("Score:", score)
