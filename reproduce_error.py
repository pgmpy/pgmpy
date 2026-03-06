import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import pgmpy.extern.tabulate...")
    from pgmpy.extern import tabulate
    print("Successfully imported tabulate.")
    
    print("Attempting to import pgmpy.factors.discrete.CPD...")
    from pgmpy.factors.discrete import CPD
    print("Successfully imported CPD.")
except Exception as e:
    import traceback
    traceback.print_exc()
