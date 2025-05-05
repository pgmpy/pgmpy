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
    random.seed(seed)
    rand_samples = [random.randint(0, 100) for _ in range(num_samples)]
    secure_samples = [secrets.randbelow(101) for _ in range(num_samples)]

    plt.figure(figsize=(12, 6))
    sns.histplot(rand_samples, kde=True, stat="density", label="random (insecure)", color="red")
    sns.histplot(secure_samples, kde=True, stat="density", label="secrets (secure)", color="green")
    plt.title("Python `random` vs. `secrets` – Distribution Comparison")
    plt.xlabel("Generated Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()