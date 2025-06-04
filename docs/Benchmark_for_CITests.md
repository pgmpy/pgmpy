# Conditional Independence (CI) Tests Benchmark Documentation

## Overview

This benchmark framework evaluates the performance of various Conditional Independence (CI) tests implemented in `pgmpy.estimators.CITests`. CI tests can perform differently depending on the underlying data-generating mechanism (DGM), sample size, variable types, effect size, and conditioning set complexity. The benchmark helps users and developers:

- Compare CI tests under standardized, reproducible settings.
- Select the best CI test for their dataset.
- Contribute new tests or data-generating mechanisms for further comparison.

Benchmark results are outputted to data files, which can be used to generate plots for documentation or further analysis.

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Clone the [pgmpy](https://github.com/pgmpy/pgmpy) repository.
- Go through the Documentation.

### Deployment and Testing

If you want to install `pgmpy` in editable mode:
```bash
pip install -e .[tests]
```

---

## Usage

### Running the Benchmark

1. Make sure `CI_benchmark.py` is present in your repository.
2. Run the script from the root directory:
   ```bash
   python CI_benchmark.py
   ```
3. The script will:
   - Run each CI test on each DGM for various sample sizes, noise levels, and effect sizes.
   - Output detailed and summary CSV files (`ci_benchmark_raw_result.csv`, `ci_benchmark_summaries.csv`).

### Custom Data-Generating Mechanisms

You can add your own DGM by:

1. Defining a function with the signature:
   ```python
   def pgmpy_custom_dgm(n_samples, effect_size=1.0, noise_std=1.0, seed=None):
       # return a pandas.DataFrame with columns ['X', 'Y', 'Z1', ...]
   ```
2. Registering it in the DGM registry (`data_generator_mechanism.py`).
3. (Optional) Running the benchmark script with your DGM included in the loop (see `CI_benchmark.py` for details).

---

## Understanding the Output

When you run the benchmark, you get two main files:

- **`ci_benchmark_raw_result.csv`**: All individual benchmark runs.
- **`ci_benchmark_summaries.csv`**: Aggregated summary statistics.

### Output Columns

#### `ci_benchmark_raw_result.csv`

| Column         | Description                                                        |
|----------------|--------------------------------------------------------------------|
| dgm            | Data Generating Mechanism used                                     |
| sample_size    | Number of samples                                                  |
| noise_level    | Standard deviation of noise                                        |
| effect_type    | "null" (independence) or "alt" (dependence)                        |
| effect_size    | Numeric effect size (0 = null, >0 = alt)                           |
| repeat         | Repetition index (for averaging)                                   |
| ci_test        | CI test used (e.g., pearsonr, gcm, pillai)                        |
| rejected_null  | Was null hypothesis rejected? (`True`/`False`/`None`)              |

#### `ci_benchmark_summaries.csv`

| Column         | Description                                                        |
|----------------|--------------------------------------------------------------------|
| dgm            | Data Generating Mechanism used                                     |
| sample_size    | Number of samples                                                  |
| noise_level    | Standard deviation of noise                                        |
| ci_test        | CI test used                                                       |
| type1_error    | False positive rate (rejecting null when true)                     |
| type2_error    | False negative rate (failing to reject alt when true)              |
| power          | 1 - type2_error                                                    |
| N_null         | Number of null runs                                                |
| N_alt          | Number of alt runs                                                 |


## Customizing the Benchmark

### Adding a New Data-Generating Mechanism (DGM)

1. Open `dgm.py`.
2. Define your function (see template above).
3. Add it to the DGM registry, for example:
   ```python
   DGP_REGISTRY["my_custom"] = my_custom_dgm
   ```
4. Add your DGM to the main DGM list in the benchmark script if you want it in official results.

### Adding a New CI Test

1. Implement your test as a function compatible with `CITests`.
2. Register it in the `ci_tests` dictionary in the benchmark script.
3. Add it to the appropriate DGM’s test list in `DGM_TO_CITESTS`.

---

## Plotting and Visualization

You can create plots from the summary CSV using pandas, matplotlib, or seaborn.

For Example : Check out the pgmpy/plots.
---

## Contribution Guidelines

- Please add tests for any new functionality.
- Follow the code style used in pgmpy.
- Document any new DGMs or CI tests in this file.

---

## References

- [pgmpy documentation](https://pgmpy.org/)
- [pgmpy/pgmpy#2150](https://github.com/pgmpy/pgmpy/issues/2150)
- Relevant academic papers (Referred as needed).

---
