def estimate_counterfactual(original_data, intervention_data, target_variable):
    return {
        "original_mean": original_data[target_variable].mean(),
        "counterfactual_mean": intervention_data[target_variable].mean()
    }
