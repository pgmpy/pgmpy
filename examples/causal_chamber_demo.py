"""
Demo script showing how to use CausalChamber datasets in pgmpy.

This demonstrates loading the CausalChamber datasets and accessing
their data and ground truth causal structures.
"""

from pgmpy.datasets import list_datasets, load_dataset

# List all CausalChamber datasets
print("Available CausalChamber datasets:")
all_datasets = list_datasets()
causal_chamber_datasets = [d for d in all_datasets if "light_tunnel" in d]
for name in causal_chamber_datasets:
    print(f"  - {name}")

# Load a dataset
print("\n" + "=" * 60)
print("Loading light_tunnel_palette dataset...")
print("=" * 60)
dataset = load_dataset("light_tunnel_palette")

# Access the data
print(f"\nDataset name: {dataset.name}")
print(f"Data shape: {dataset.data.shape}")
print(f"Number of variables: {dataset.tags['n_variables']}")
print(f"Number of samples: {dataset.tags['n_samples']}")

print("\nFirst few columns:")
print(dataset.data.columns[:10].tolist())

print("\nFirst few rows:")
print(dataset.data.head(3))

# Access ground truth
print("\n" + "=" * 60)
print("Ground Truth Causal Graph")
print("=" * 60)
ground_truth = dataset.ground_truth
print(f"Number of nodes: {len(ground_truth.nodes())}")
print(f"Number of edges: {len(ground_truth.edges())}")

print("\nSample nodes:")
print(list(ground_truth.nodes())[:10])

print("\nSample edges (first 10):")
for i, (parent, child) in enumerate(ground_truth.edges()):
    if i >= 10:
        break
    print(f"  {parent} → {child}")

# You can use this data for causal discovery evaluation
print("\n" + "=" * 60)
print("Usage Example")
print("=" * 60)
print(
    """
# Use for structure learning evaluation
from pgmpy.estimators import PC

# Learn structure
pc = PC(dataset.data)
estimated_model = pc.estimate()

# Compare with ground truth
from pgmpy.metrics import StructureScore
score = StructureScore(estimated_model, dataset.ground_truth)
print(f"Structural Hamming Distance: {score.shd()}")
"""
)
