#This module initializes flags for optional dependencies
try:
    import pandas
    HAS_PANDAS = True
except:
    HAS_PANDAS = False
