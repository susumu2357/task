# Based on https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/sklearn_abalone_featurizer.py

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
from sklearn.model_selection import train_test_split

label_column = 'default'

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

#     Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()
    
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train) if file.endswith('csv')]
    
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
        )
        for file in input_files
    ]
    concat_data = pd.concat(raw_data)
    concat_data.drop(label_column, axis=1, inplace=True)
        
    categorical_cols = [col for col in concat_data.columns if col not in numeric_cols]

#     class_0 = concat_data[concat_data[label_column] == 0]
#     class_1 = concat_data[concat_data[label_column] == 1]
#     down_class_0 = class_0.sample(4*len(class_1), random_state=42) # class_0 : class_1 = 4 : 1
    
#     train_data = pd.concat([down_class_0, class_1])

    train_data = concat_data

    numeric_transformer = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()) # NaN marking.

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1),
        OneHotEncoder(handle_unknown="ignore"),
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )

    preprocessor.fit(train_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")

    
def input_fn(input_data, content_type):
    """Parse input data payload"""
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        return df
    elif content_type == "application/json":
        try:
            df = pd.read_json(StringIO(data), orient='records')
        except:
            df = pd.read_json(StringIO(data), orient='records', typ='series')
#         df = pd.read_json(StringIO(input_data), orient='records')
        print(df.shape) # print the shape for logging purpose
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data"""
    variables = [col for col in input_data.columns if col != label_column]
    features = model.transform(input_data[variables])
    features = features.toarray()
    print(f'feature shape: {features.shape}') # print the shape for logging purpose

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor