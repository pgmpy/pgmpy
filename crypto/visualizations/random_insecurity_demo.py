import random
import secrets
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rand_vs_secure(num_samples=1000, seed=42):
    """
    Visualizes the difference between Python's `random` and `secrets` modules.

    Args:
        num_samples (int): Number of samples to generate from each PRNG.
        seed (int): Seed value to make Python `random` predictable.

    Returns:
        None. Displays histogram comparison.
    """