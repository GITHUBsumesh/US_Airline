import os
import sys
import numpy as np
import pandas as pd
from typing import Dict
"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "arr_del15"
TARGET_COLUMN = TARGET_COLUMN.strip().replace('"', '')
PIPELINE_NAME: str = "US_Airline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "cleaned_flights.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"
DROP_COLUMNS= ["carrier_name", "airport_name"] # type: ignore
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "US_Airline"
DATA_INGESTION_DATABASE_NAME: str = "US_Airline"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME: Dict[str, str] = {
    "linear": "linear_preprocessor.pkl",
    "ridge": "ridge_preprocessor.pkl",
    "lasso": "lasso_preprocessor.pkl",
    "knn": "knn_preprocessor.pkl",
    "svr": "svr_preprocessor.pkl",
    "dnn": "dnn_preprocessor.pkl",
    "rf": "rf_preprocessor.pkl",
    "extratrees": "extratrees_preprocessor.pkl",
    "xgb": "xgb_preprocessor.pkl",
    "lgbm": "lgbm_preprocessor.pkl",
    "gbr": "gbr_preprocessor.pkl",
    "catboost": "catboost_preprocessor.pkl",
}

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_objects"
OHE_MODELS = ["linear", "ridge", "lasso", "knn", "svr", "dnn"]
ORDINAL_MODELS = ["rf", "extratrees", "xgb", "lgbm", "gbr"]
RAW_MODELS = ["catboost"]
SKIP_SCALING_MODELS = ["rf", "extratrees", "catboost"]
ALL_MODELS = OHE_MODELS + ORDINAL_MODELS + RAW_MODELS


## simple imputer to replace nan values (numerical )
DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "strategy": "mean",
}
## KNN imputer to replace nan values (categorical )
DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "strategy": "most_frequent",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "usairline"