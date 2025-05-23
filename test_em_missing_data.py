import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_example_model


def run_em_test(model_name="asia", n_samples=1000, missing_rate=0.2, max_iter=50):
    print("\n=== Testing EM with Missing Data ===\n")

    # 1. Load an example model (using a smaller one)
    model = get_example_model(model_name)
    print(
        f"Loaded {model_name} model with {len(model.nodes())} nodes and {len(model.edges())} edges"
    )

    # 2. Simulate data from the model
    df_complete = model.simulate(n_samples=n_samples, seed=42)
    print(f"Simulated {n_samples} samples from the model")

    # 3. Introduce missing values (NaNs)
    df_missing = df_complete.copy()
    np.random.seed(42)
    mask = np.random.choice(
        [True, False], size=df_missing.shape, p=[missing_rate, 1 - missing_rate]
    )
    df_missing = df_missing.mask(mask)
    df_missing.dropna(axis=1, how="all", inplace=True)

    print(f"Introduced approximately {missing_rate*100}% missing values")
    print(f"Number of NaNs per column (first 5 columns):")
    print(df_missing.isnull().sum().head())

    # 4. Initialize and run EM estimator
    print("\nRunning EM algorithm...")
    estimator = EM(model, df_missing.copy())

    effective_latents = estimator.model_copy.latents
    latent_card_for_em = {}
    if effective_latents:
        print(f"Effective latents for EM: {effective_latents}")
        for latent_var in effective_latents:
            if latent_var in df_complete.columns:
                latent_card_for_em[latent_var] = df_complete[latent_var].nunique()
            else:
                latent_card_for_em[latent_var] = 2

    estimated_cpds = estimator.get_parameters(
        latent_card=latent_card_for_em if latent_card_for_em else None,
        max_iter=max_iter,
        show_progress=True,
        n_jobs=1,
    )

    # 5. Validate the results
    print("\n=== Validation ===")

    # 5.1 Check that all CPDs were estimated
    expected_cpd_count = len(model.nodes())
    actual_cpd_count = len(estimated_cpds)
    print(f"Expected CPDs: {expected_cpd_count}, Actual CPDs: {actual_cpd_count}")
    assert (
        actual_cpd_count == expected_cpd_count
    ), f"Expected {expected_cpd_count} CPDs, got {actual_cpd_count}"

    # 5.2 Compare with MLE on complete data (as benchmark)
    print("\nComparing with MLE on complete data...")
    try:
        mle = MLE(model, df_complete)
        mle_cpds = mle.get_parameters()

        # Compare a few key CPDs
        sample_nodes = list(model.nodes())[:3]  # First three nodes for comparison
        for node in sample_nodes:
            em_cpd = next((cpd for cpd in estimated_cpds if cpd.variable == node), None)
            mle_cpd = next((cpd for cpd in mle_cpds if cpd.variable == node), None)

            if em_cpd and mle_cpd:
                # Calculate mean absolute difference
                em_values = em_cpd.values.flatten()
                mle_values = mle_cpd.values.flatten()
                mean_abs_diff = np.mean(np.abs(em_values - mle_values))
                print(
                    f"Node {node} - Mean absolute difference between EM and MLE: {mean_abs_diff:.4f}"
                )

        print(
            "\nNote: Some difference is expected due to missing values and the nature of the EM algorithm."
        )
    except Exception as e:
        print(f"Couldn't compare with MLE: {e}")

    # 5.3 Check if the model with EM parameters can be used for inference
    print("\nVerifying model usability...")
    try:
        model_with_em = model.copy()
        model_with_em.cpds = estimated_cpds
        # Try a simple query to ensure the model works
        model_with_em.check_model()
        print("Model with EM parameters is valid and can be used for inference.")
    except Exception as e:
        print(f"Model validation failed: {e}")
        raise

    print("\n=== Test Completed Successfully ===")
    return model, estimated_cpds, df_complete, df_missing


if __name__ == "__main__":
    try:
        # Use the smaller "asia" model instead of "alarm"
        model, cpds, complete_data, missing_data = run_em_test(
            model_name="asia", n_samples=1000, missing_rate=0.2, max_iter=50
        )
        print(
            "\nTo run this test: source pgmpy_env/bin/activate && python test_em_missing_data.py"
        )
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
