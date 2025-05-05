# The Illusion of Randomness in Python's `random`: A Measure-Theoretic Warning

## Introduction

> "If you think Python's `random.random()` gives you unpredictability, think again."

This notebook educates mathematically mature readers — particularly university-level students in mathematics or computer science — about the **cryptographic insecurity** of Python's built-in `random` module. Unlike true randomness, Python's generator is **predictable**, **stateful**, and **unsuitable** for any task requiring cryptographic or information-theoretic security.

We will explore the vulnerabilities from a **rigorous, measure-theoretic** standpoint using concepts from:

- Topology and σ-algebras
- Measurable functions
- Pushforward measures
- Lebesgue integrals
- Probability measures and pseudorandomness
## I. What is Randomness, Mathematically?

A **probability space** $(\Omega, \mathcal{F}, P)$ consists of:

- $\Omega$: sample space  
- $\mathcal{F}$: a $\sigma$-algebra of measurable sets  
- $P$: a probability measure $P : \mathcal{F} \to [0, 1]$ with $P(\Omega) = 1$

A **random variable** $X : \Omega \to \mathbb{R}$ is a **measurable function**:  
$X^{-1}(B) \in \mathcal{F} \quad \text{for all } B \in \mathcal{B}(\mathbb{R})$

> If we generate numbers using Python’s `random`, is it truly measurable in this rigorous sense?

---

## II. What Makes a PRNG Cryptographically Secure?

> A PRNG is **cryptographically secure** if its output is **indistinguishable** from uniform by any efficient adversary.

**Definition (CSPRNG):**  
A PRNG $G : \{0, 1\}^s \to \{0, 1\}^n$ is cryptographically secure if for every polynomial-time distinguisher $A$,

$$
\left| \Pr[A(G(s)) = 1] - \Pr[A(U_n) = 1] \right| < \varepsilon(n)
$$

where $U_n \sim \text{Uniform}(\{0,1\}^n)$ and $\varepsilon$ is negligible.

> Python’s `random` module fails this test. Why? Because it is **stateful** and easily reverse-engineered
## III. Attack Demonstration: Predicting Python's `random`

```python
import random

random.seed(42)
for _ in range(5):
    print(random.random())
```
Given a few output values, one can easily brute-force the seed.
This shows the PRNG is not forward secure.
## IV. Measure-Theoretic Analysis

Let \( f : \Omega \to \mathbb{R} \) be a random number generator. In true randomness, \( f \) must be:

- a **measurable function**
- pushing forward the uniform measure
- **indistinguishable** from \( U(0,1) \)

But with Python's `random.random()`, the pushforward measure \( f_*(\mu) \)  
becomes **concentrated** on a low-dimensional subspace due to finite state space.
## V. Visualization: Histogram Divergence from Uniform

```python
import matplotlib.pyplot as plt
import random
import secrets

# Generate samples
random.seed(42)
rand_samples = [random.random() for _ in range(10000)]
secure_samples = [secrets.randbelow(10000)/10000 for _ in range(10000)]

# Plot histograms
plt.hist(rand_samples, bins=50, alpha=0.5, label='random.random')
plt.hist(secure_samples, bins=50, alpha=0.5, label='secrets.randbelow')
plt.title("Distribution of PRNG outputs")
plt.legend()
plt.show()
```

## VI. Statistical Tests (Basic Entropy Check)

```python
import math
from collections import Counter

def shannon_entropy(bits):
    freq = Counter(bits)
    total = len(bits)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())

bits = ''.join(f'{int(x*10000):014b}' for x in rand_sample)
entropy = shannon_entropy(bits)
print(f"Entropy per bit: {entropy / len(bits):.5f}")
```
## VII. Secure Alternatives

```python
import secrets
print(secrets.token_hex(16))  # Cryptographically secure
```
## VIII. Summary and Takeaway

> **Measure theory doesn’t lie.** `random.random()` is not measurable from a security perspective.

- Not cryptographically secure  
- Predictable from outputs  
- Fails indistinguishability  
- Unsafe for keys, tokens, salts, sessions  

**Application:**
- `secrets` for all security-related randomness  
- `random` only for simulations and games
