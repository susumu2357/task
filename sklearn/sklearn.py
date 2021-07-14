import argparse
import csv
import json
import os
import shutil
import sys
import time
from io import StringIO

import joblib
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

label = 'default'

variables = [x for x in data_df.columns if x != 'uuid' and x != label]

numeric_cols = [
    'account_amount_added_12_24m',
    'account_days_in_dc_12_24m',
    'account_days_in_rem_12_24m',
    'account_days_in_term_12_24m',
    'account_incoming_debt_vs_paid_0_24m',
    'age',
    'avg_payment_span_0_12m',
    'avg_payment_span_0_3m',
    'max_paid_inv_0_12m',
    'max_paid_inv_0_24m',
    'num_active_div_by_paid_inv_0_12m',
    'num_active_inv',
    'num_arch_dc_0_12m',
    'num_arch_dc_12_24m',
    'num_arch_ok_0_12m',
    'num_arch_ok_12_24m',
    'num_arch_rem_0_12m',
    'num_arch_written_off_0_12m',
    'num_arch_written_off_12_24m',
    'num_unpaid_bills',
    'recovery_debt',
    'sum_capital_paid_account_0_12m',
    'sum_capital_paid_account_12_24m',
    'sum_paid_inv_0_12m',
    'time_hours'    
]

categorical_cols = [x for x in variables if x not in numeric_cols]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    raw_data = [
        pd.read_csv(
            file,
            header=None,
            names=feature_columns_names + [label_column],
            dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
        )
        for file in input_files
    ]
    concat_data = pd.concat(raw_data)

    # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
    concat_data.drop(label_column, axis=1, inplace=True)

    # This section is adapted from the scikit-learn example of using preprocessing pipelines:
    #
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    #
    # We will train our classifier with the following features:
    # Numeric Features:
    # - length:  Longest shell measurement
    # - diameter: Diameter perpendicular to length
    # - height:  Height with meat in shell
    # - whole_weight: Weight of whole abalone
    # - shucked_weight: Weight of meat
    # - viscera_weight: Gut weight (after bleeding)
    # - shell_weight: Weight after being dried
    # Categorical Features:
    # - sex: categories encoded as strings {'M', 'F', 'I'} where 'I' is Infant
    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="category")),
        ]
    )

    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")