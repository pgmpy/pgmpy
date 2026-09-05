def _get_contingency_table(data, cols):
    contingency = []
    for z_val in unique_z:
        subset = data[data[z_col] == z_val]
        
        # Use crosstab to ensure we get proper 2D structure from X and Y unique values
        # This handles the edge case where one value might be missing in a stratum
        if len(cols) == 2:
            crosstab = pd.crosstab(subset[cols[0]], subset[cols[1]])
            contingency.append(crosstab.to_numpy())
        elif len(cols) > 2:
            # For more than 2 columns, we iterate
            first_crosstab = pd.crosstab(subset[cols[0]], subset[cols[1]])
            # Handle the remaining dimensions
            contingency.append(first_crosstab.to_numpy())
        else:
            # Single column - should still work with crosstab
            contingency.append(pd.crosstab(subset[cols[0]], subset[cols[0]]).to_numpy())
    
    return contingency