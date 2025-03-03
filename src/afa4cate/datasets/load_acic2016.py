"""
Code for obtaining the ACIC2016 dataset. Adapted from the CATENets library: https://github.com/AliciaCurth/CATENets.
"""
import glob

# stdlib
import random
from pathlib import Path
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from afa4cate.datasets.network import download_if_needed

np.random.seed(0)
random.seed(0)

FILE_ID = "0B7pG5PPgj6A3N09ibmFwNWE1djA"
PREPROCESSED_FILE_ID = "1iOfEAk402o3jYBs2Prfiz6oaailwWcR5"

NUMERIC_COLS = [
    0,
    3,
    4,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    24,
    25,
    30,
    31,
    32,
    33,
    39,
    40,
    41,
    53,
    54,
]
N_NUM_COLS = len(NUMERIC_COLS)


def get_acic_covariates(
    fn_csv: Path, keep_categorical: bool = False, preprocessed: bool = True
) -> np.ndarray:
    X = pd.read_csv(fn_csv)
    if not keep_categorical:
        X = X.drop(columns=["x_2", "x_21", "x_24"])
    else:
        # encode categorical features
        feature_list = []
        for cols_ in X.columns:
            if type(X.loc[X.index[0], cols_]) not in [np.int64, np.float64]:

                enc = OneHotEncoder(drop="first")

                enc.fit(np.array(X[[cols_]]).reshape((-1, 1)))

                for k in range(len(list(enc.get_feature_names()))):
                    X[cols_ + list(enc.get_feature_names())[k]] = enc.transform(
                        np.array(X[[cols_]]).reshape((-1, 1))
                    ).toarray()[:, k]

                feature_list.append(cols_)

        X.drop(feature_list, axis=1, inplace=True)

    if preprocessed:
        X_t = X.values
    else:
        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)
    return X_t


def load(
    data_path: Path,
    preprocessed: bool = True,
    original_acic_outcomes: bool = False,
    **kwargs: Any,
) -> Tuple:
    """
    ACIC2016 dataset dataloader.
        - Download the dataset if needed.
        - Load the dataset.
        - Preprocess the data.
        - Return train/test split.

    Parameters
    ----------
    data_path: Path
        Path to the CSV. If it is missing, it will be downloaded.
    preprocessed: bool
        Switch between the raw and preprocessed versions of the dataset.
    original_acic_outcomes: bool
        Switch between new simulations (Inductive bias paper) and original acic outcomes

    Returns
    -------
    train_x: array or pd.DataFrame
        Features in training data.
    train_t: array or pd.DataFrame
        Treatments in training data.
    train_y: array or pd.DataFrame
        Observed outcomes in training data.
    train_potential_y: array or pd.DataFrame
        Potential outcomes in training data.
    test_x: array or pd.DataFrame
        Features in testing data.
    test_potential_y: array or pd.DataFrame
        Potential outcomes in testing data.
    """
    if preprocessed:
        csv = data_path / "x_trans.csv"

        download_if_needed(csv, file_id=PREPROCESSED_FILE_ID)
    else:
        arch = data_path / "data_cf_all.tar.gz"

        download_if_needed(
            arch, file_id=FILE_ID, unarchive=True, unarchive_folder=data_path
        )

        csv = data_path / "data_cf_all/x.csv"

    return get_acic_covariates(
        csv, keep_categorical=False, preprocessed=preprocessed
    )