import numpy as np
import pandas as pd
from scipy import stats


#margin of error: t-distribution, CL 95%, confidence intervals: mean +/- margin of error
def t_margin_of_error(data, confidence_level=0.95):
    data = np.asarray(data, dtype=float)

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional array.")

    if data.size < 2 or not np.all(np.isfinite(data)):
        return np.nan

    n = data.size
    std_err = stats.sem(data)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)

    return t_critical * std_err


# 2-sample t-test: group 1 vs group 2; independent (two sample), assumes equal variance (by default), two-tailed
def two_sample_t_test(column1, column2):
    try:
        column1 = np.asarray(column1, dtype=float)
        column2 = np.asarray(column2, dtype=float)
    except Exception as ex:
        print(f'failed to convert input to float: {ex}')
        return np.NaN

    if np.isnan(column1).all() or np.isnan(column2).all():
        print("Warning: at least one of the groups contains only missing values (np.NaNs)")
        return np.nan

    if np.isnan(column1).any() or np.isnan(column2).any():
        print("Warning: at least one of the groups contains missing values (np.NaNs)")
        column1 = column1[~np.isnan(column1)]
        column2 = column2[~np.isnan(column2)]

    try:
        t_stat, p_value = stats.ttest_ind(column1, column2)
        return p_value
    except Exception as ex:
        print(f'failed to calculate the p-value: {ex}')
        return np.NaN


#1-sample t-test: reference value vs group; independent (two sample), assumes equal variance (by default), two-tailed
def one_sample_t_test(reference, column):
    try:
        reference = float(reference)
        column = np.asarray(column, dtype=float)
    except Exception as ex:
        print(f'Failed to convert input to float: {ex}')
        return np.nan

    if np.isnan(column).all() or np.isnan(reference):
        print("Warning: the group contains only missing values (np.NaNs) or the reference is NaN")
        return np.nan

    if np.isnan(column).any():
        print("Warning: the group contains missing values (np.NaNs)")
        column = column[~np.isnan(column)]

    try:
        t_stat, p_value = stats.ttest_1samp(column, popmean=reference)
        return p_value
    except Exception as ex:
        print(f'Failed to calculate the p-value: {ex}')
        return np.nan


#paired t-test: before vs after; independent (two sample), assumes equal variance (by default), two-tailed
def paired_t_test(column1, column2):
    try:
        column1 = np.asarray(column1, dtype=float)
        column2 = np.asarray(column2, dtype=float)
    except Exception as ex:
        print(f'Failed to convert input to float: {ex}')
        return np.nan

    if len(column1) != len(column2):
        print("Warning: the groups have different lengths")
        return np.nan

    if np.isnan(column1).all() or np.isnan(column2).all():
        print("Warning: at least one group contains only missing values (NaNs)")
        return np.nan

    # Remove pairs where either observation is NaN
    valid_positions = ~np.isnan(column1) & ~np.isnan(column2)
    if not valid_positions.all():
        print("Warning: missing values detected; removing incomplete pairs")
    column1 = column1[valid_positions]
    column2 = column2[valid_positions]

    if len(column1) < 2:
        print("Warning: fewer than two complete pairs")
        return np.nan

    try:
        t_stat, p_value = stats.ttest_rel(column1, column2)
        return p_value
    except Exception as ex:
        print(f'Failed to calculate the p-value: {ex}')
        return np.nan

# median
def median(data):
    try:
        numeric_dataset = pd.to_numeric(pd.Series(data), errors='coerce').dropna().to_numpy()
    except Exception as ex:
        raise ValueError(f'Failed to convert input to float: {ex}')

    if len(numeric_dataset) == 0:
        raise ValueError("No valid numerical values found.")

    return np.median(numeric_dataset)


#iqr-based outlie removal
def iqr_outlier_removal(data, factor= 1.5):
    # Convert to numeric and remove invalid values
    data = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

    # Calculate IQR bounds
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    # Remove outliers
    return data[(data >= lower) & (data <= upper)]


#permutation test for K and alpha parameters
def permutation_test_median(data, category_column, reference_group, tested_parameter, **kwargs):
    # check the inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Input data must be a pandas DataFrame.')
    if not isinstance(category_column, str):
        raise TypeError('category_column must be a string.')
    if category_column not in data.columns:
        raise ValueError(f'Column "{category_column}" not found in DataFrame.')
    if not isinstance(reference_group, str):
        raise TypeError('reference_group must be a string.')
    if reference_group not in data[category_column].unique():
        raise ValueError(f'Reference group "{reference_group}" not found in "{category_column}".')
    if not isinstance(tested_parameter, str):
        raise TypeError('tested_parameter must be a string.')
    if tested_parameter not in data.columns:
        raise ValueError(f'Column "{tested_parameter}" not found in DataFrame.')

    # import the lib
    from scipy.stats import permutation_test

    # split the control from the dataset
    control_data = data.loc[data[category_column] == reference_group, tested_parameter]
    if control_data.empty:
        raise ValueError(f'Reference group "{reference_group}" contains no valid datapoints.')
    data = data.loc[data[category_column] != reference_group, [tested_parameter, category_column]]

    # iterate over groups and perform the permutation test agains the control data
    import time
    groups = []
    p_values = []
    for group in data[category_column].unique():
        try:
            start_time = time.perf_counter()
            tested_data = data.loc[data[category_column] == group, tested_parameter]
            result = permutation_test((control_data, tested_data),
                                      statistic=lambda x, y: np.median(x) - np.median(y),
                                      permutation_type=kwargs.get('permutation_type', 'independent'),
                                      alternative=kwargs.get('alternative', 'two-sided'),
                                      n_resamples=kwargs.get('n_resamples', 100000),
                                      random_state=kwargs.get('random_state', 42))
            groups.append(group)
            p_values.append(result.pvalue)

            duration = (time.perf_counter() - start_time)/60
            print(f'Group: {group} | p-value: {result.pvalue} | Duration: {duration:.2f} minutes')

        except Exception as ex:
            print(f'failed to calculate the p-value for group {group}: {ex}')
            groups.append(group)
            p_values.append(np.nan)

    significance = pd.DataFrame({'group': groups,
                                 'p_value': p_values})

    print(f"Permutation test: {kwargs.get('permutation_type', 'independent')}, "
          f"{kwargs.get('alternative', 'two-sided')}, "
          f"number of resamples: {kwargs.get('n_resamples', 100000)}")

    # assign the significance symbols
    significance = significance.assign(significance_symbol=np.where(significance.p_value < 0.001, '***',
                                                                    np.where(significance.p_value < 0.01, '**',
                                                                             np.where(significance.p_value < 0.05, '*',
                                                                                      ''))))

    ##export (avod re-runs of the permutation test function)
    if kwargs.get('export', False):
        from pathlib import Path

        path = kwargs.get('path')

        if not path:
            raise ValueError("Export path must be provided.")

        output_path = Path(path)

        if not output_path.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {output_path.parent}")

        significance.to_csv(output_path, index=False, encoding='utf-8')

    return significance