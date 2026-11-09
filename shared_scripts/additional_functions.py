import numpy as np
import pandas as pd
from scipy import stats


#extract significance symbols from the significance dataset
def extract_significance_symbols(**kwargs):
    from pathlib import Path

    # get data
    input_pathway = kwargs.get('path', None)
    if input_pathway is None:
        raise ValueError("path to a permutation test data not provided")
    if not Path(input_pathway).is_file():
        raise FileNotFoundError(f"File does not exist: {input_pathway}")
    if Path(input_pathway).suffix.lower() != ".csv":
        raise ValueError(f"Input must be a CSV file")

    data = pd.read_csv(
        input_pathway).dropna()  # group/strain with no statistical significance, against the control group, dropped

    if data.group.duplicated().any():
        print("WARNING: Duplicated group")

    sig_dict = dict(zip(data.group, data.significance_symbol))

    return sig_dict