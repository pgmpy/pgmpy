import os
import traceback

import pandas as pd

from pgmpy.base import DAG

# UPDATE: Only import load_dataset and DATASETS.
# load_abalone and load_sachs were removed.
from pgmpy.datasets import DATASETS, load_dataset

print("--- 1. TESTING REGISTRY ---")
datasets_list = DATASETS.list_all()
print(f"Available datasets: {datasets_list}")
assert "abalone" in datasets_list
assert "sachs" in datasets_list
print("Registry test PASSED")
print("-" * 30 + "\n")


print("--- 2. TESTING GENERIC LOADER: load_dataset('sachs') ---")
try:
    # This remains the same
    sachs_df, sachs_gt = load_dataset("sachs")

    print("Sachs Data (Head):")
    print(sachs_df.head())
    print(f"\nSachs Ground Truth Edges ({len(sachs_gt.edges())} total):")
    print(sachs_gt.edges())

    print(f"\nData type correct: {isinstance(sachs_df, pd.DataFrame)}")
    print(f"Ground truth type correct: {isinstance(sachs_gt, DAG)}")

    assert isinstance(sachs_df, pd.DataFrame)
    assert isinstance(sachs_gt, DAG)
    assert len(sachs_gt.edges()) > 15
    print(" Generic 'sachs' test PASSED")

except Exception:
    print(f"'sachs' test FAILED: {traceback.format_exc()}")
print("-" * 30 + "\n")


print("--- 3. TESTING SPECIFIC VARIANT: load_dataset('abalone') ---")
try:
    # UPDATE: Changed load_abalone(...) to load_dataset("abalone", ...)
    abalone_df, abalone_gt = load_dataset(
        "abalone", variant="continuous", load_ground_truth=True
    )

    print("Abalone Data (Head):")
    print(abalone_df.head())
    print(f"\nAbalone Ground Truth Edges ({len(abalone_gt.edges())} total):")
    print(abalone_gt.edges())

    print(f"\nData type correct: {isinstance(abalone_df, pd.DataFrame)}")
    print(f"Ground truth type correct: {isinstance(abalone_gt, DAG)}")

    assert isinstance(abalone_df, pd.DataFrame)
    assert isinstance(abalone_gt, DAG)
    assert len(abalone_gt.edges()) > 5
    print(" Specific 'abalone' variant test PASSED")

except Exception:
    print(f" 'abalone' test FAILED: {traceback.format_exc()}")
print("-" * 30 + "\n")

print("--- 4. TESTING CACHING ---")
print("Run this script again. If the downloads are instant, caching is working!")
print(
    f"Check folder: {os.environ.get('PGMPY_DATA_HOME', os.path.join(os.path.expanduser('~'), '.pgmpy', 'data'))}"
)
print("\n ALL TESTS COMPLETED ")
