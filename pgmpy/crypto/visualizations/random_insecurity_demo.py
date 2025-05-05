import random
import numpy as np
from scipy.special import erfc

# Constants
BITSTREAM_LENGTH = 2000
BLOCK_SIZE = 100
ALPHA = 0.01

def generate_bitstream_random(length=BITSTREAM_LENGTH):
    return [1 if random.random() > 0.5 else 0 for _ in range(length)]

def block_frequency_test(bits, M=BLOCK_SIZE, alpha=ALPHA):
    N = len(bits) // M
    proportions = []
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        pi = sum(block) / M
        proportions.append(pi)
    chi_squared = sum([(pi - 0.5)**2 for pi in proportions]) * 4 * M
    p_value = erfc(np.sqrt(chi_squared / 2))
    bias = np.mean(proportions) - 0.5
    verdict = '✅ PASSES (sufficiently random)' if p_value >= alpha else '❌ FAILS (bias detected)'
    return {
        "bitstream": bits,
        "p_value": p_value,
        "proportions": proportions,
        "chi_squared": chi_squared,
        "bias": bias,
        "mean_proportion": np.mean(proportions),
        "verdict": verdict,
        "N": N,
        "block_size": M
    }

def nist_mathematical_intuition(results):
    explanation = f"""
📐 Mathematically Rigorous Explanation of the NIST Block Frequency Test
=======================================================================

Set-up:
-------
Let S be a finite bitstring of length n, i.e., S ∈ {{0,1}}^n, where n = {BITSTREAM_LENGTH}.
Define a block size M = {BLOCK_SIZE}, and partition S into N = n / M = {results['N']} blocks.

Let B_i ∈ {{0,1}}^M be the i-th block, and define π_i as the empirical frequency of 1s in B_i:
  π_i = (1 / M) * Σ_{{j=1}}^M B_i[j]

Assumption (Null Hypothesis H₀):
--------------------------------
Under the assumption that bits in S are i.i.d. Bernoulli(p=1/2), each π_i is approximately normal
with mean μ = 0.5 and variance σ² = 1 / (4M), by the Central Limit Theorem.

So, for all i, π_i ∼ N(0.5, 1/(4M))

Statistical Deviation:
----------------------
The cumulative deviation from 0.5 over N blocks is captured by the test statistic:

  χ² = 4M * Σ_{{i=1}}^N (π_i - 0.5)²

This test statistic approximates a chi-squared distribution with N degrees of freedom,
under the null hypothesis of uniform randomness.

p-value:
--------
We transform χ² into a p-value using the complementary error function (erfc):

  p = erfc(√(χ² / 2))

Interpretation:
---------------
- If p ≥ α = {ALPHA}, then the deviations are consistent with random chance → PASS.
- If p < α, the bias is statistically significant → FAIL.

Prime/Entropy Degradation View (optional intuition):
----------------------------------------------------
In information-theoretic terms, a uniformly random bitstring has maximum entropy:
  H(S) = n bits

Any statistical deviation implies information redundancy (bias), which leaks structure.
Analogously, PRNGs like Python’s `random()` suffer entropy degradation akin to
the loss of prime unpredictability in weak pseudoprimes or congruence-structured sets.

Result:
-------
Mean proportion of 1s:      {results['mean_proportion']:.4f}
Bias from 0.5:              {results['bias']:.4f}
Chi-squared (χ²):           {results['chi_squared']:.4f}
p-value:                    {results['p_value']:.6f}
Verdict:                    {results['verdict']}

Test It Yourself:
-----------------
You can rerun this code to generate another bitstream using Python's `random()` and
observe how often the generator fails under the NIST model of randomness.
"""
    print(explanation)

if __name__ == "__main__":
    bits = generate_bitstream_random()
    results = block_frequency_test(bits)
    nist_mathematical_intuition(results)
