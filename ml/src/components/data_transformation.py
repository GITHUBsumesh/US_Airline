import sys
import os,joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import Dict
from ml.src.utils.path_config import ML_ROOT

from src.constants.training_pipeline import TARGET_COLUMN
from src.constants.training_pipeline import DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS,DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import AirLineException 
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data,save_object,read_yaml_file, load_numpy_array_data

# Define model types
OHE_MODELS = ["linear", "ridge", "lasso", "knn", "svr", "dnn"]
LABEL_MODELS = ["rf", "extratrees", "xgb", "lgbm", "gbr"]
RAW_MODELS = ["catboost"]
SKIP_SCALING_MODELS = ["rf", "extratrees", "catboost"]


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise AirLineException(e,sys)
        

    @staticmethod
    def get_imputer():
        return SimpleImputer(strategy="mean")
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AirLineException(e, sys)

    # def get_column_transformer(self, df: pd.DataFrame, model_name: str, is_train: bool = True) -> ColumnTransformer:
    #     """
    #     Builds a ColumnTransformer that:
    #     - One-hot encodes 'carrier'
    #     - Label encodes 'origin' and 'dest'
    #     - Imputes and scales numerical columns
    #     - fit_transform on train and only transform on test.
    #     Applies label encoding in-place to df.
    #     """
    #     import joblib

    #     transformers = []

    #     # Clean column names and string values to avoid whitespace issues
    #     df.columns = df.columns.str.strip()
    #     for col in df.select_dtypes(include="object").columns:
    #         df[col] = df[col].astype(str).str.strip()
    #     # Numeric columns
    #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    #     # Remove target if it exists
    #     if TARGET_COLUMN in numeric_cols:
    #         numeric_cols.remove(TARGET_COLUMN)

    #     num_pipeline_steps = [("imputer", SimpleImputer(**DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS))]
    #     if model_name not in SKIP_SCALING_MODELS:
    #         num_pipeline_steps.append(("scaler", StandardScaler()))
    #     transformers.append(("num", Pipeline(steps=num_pipeline_steps), numeric_cols))

    #     # One-Hot Encode 'carrier'
    #     if "carrier" in df.columns:
    #         cat_pipeline = Pipeline(steps=[
    #             ("imputer", SimpleImputer(**DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS)),
    #             ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    #         ])
    #         transformers.append(("carrier_ohe", cat_pipeline, ["carrier"]))

    #     # Label encode 'origin' and 'dest' (in-place, must be done outside ColumnTransformer)
    #     for col in ["origin", "dest"]:
    #         if col in df.columns:
    #             label_path = os.path.join("artifacts/label_encoders", f"{col}_{model_name}.pkl")
    #             os.makedirs(os.path.dirname(label_path), exist_ok=True)

    #             if is_train:
    #                 le = LabelEncoder()
    #                 df[col] = le.fit_transform(df[col].astype(str))
    #                 joblib.dump(le, label_path)
    #             else:
    #                 le = joblib.load(label_path)
    #                 df[col] = le.transform(df[col].astype(str))

    #     return ColumnTransformer(transformers=transformers)
    
    def get_column_transformer(self, df: pd.DataFrame, model_name: str, is_train: bool = True) -> ColumnTransformer:
        """
        Builds a ColumnTransformer that:
        - One-hot encodes 'carrier'
        - Label encodes 'origin' and 'dest'
        - Imputes and scales numerical columns
        - fit_transform on train and only transform on test.
        Applies label encoding in-place to df.
        """
        import joblib

        transformers = []

        # Clean column names and string values to avoid whitespace issues
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if TARGET_COLUMN in numeric_cols:
            numeric_cols.remove(TARGET_COLUMN)

        # Numeric pipeline
        num_pipeline_steps = [("imputer", SimpleImputer(**DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS))]
        if model_name not in SKIP_SCALING_MODELS:
            num_pipeline_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_pipeline_steps), numeric_cols))

        # One-Hot Encode 'carrier'
        if "carrier" in df.columns:
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(**DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS)),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("carrier_ohe", cat_pipeline, ["carrier"]))

        # Label encode 'origin' and 'dest' (in-place)
        for col in ["origin", "dest"]:
            if col in df.columns:
                label_path = os.path.join("artifacts/label_encoders", f"{col}_{model_name}.pkl")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    joblib.dump(le, label_path)
                else:
                    le = joblib.load(label_path)
                    df[col] = le.transform(df[col].astype(str))

        return ColumnTransformer(transformers=transformers)

    def build_model_preprocessor(self, df: pd.DataFrame, model_name: str, is_train: bool = True) -> Pipeline:
        if model_name in RAW_MODELS:
            return None

        column_transformer = self.get_column_transformer(df, model_name, is_train=is_train)
        return Pipeline([("preprocessor", column_transformer)])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation process")

        train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
        test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

        # Clean column names
        train_df.columns = train_df.columns.str.strip().str.replace('"', '')
        # print(f"Train DataFrame columns: {train_df.columns.tolist()}")
        test_df.columns = test_df.columns.str.strip().str.replace('"', '')

        input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
        input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
        target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)
        target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

        all_models = OHE_MODELS + LABEL_MODELS + RAW_MODELS
        transformed_objects: Dict[str, str] = {}

        for model in all_models:
            logging.info(f"Processing for model: {model}")
            model_train_df = input_feature_train_df.copy()
            model_test_df = input_feature_test_df.copy()

             # Build and fit transformer on training data
            preprocessor = self.build_model_preprocessor(model_train_df, model, is_train=True)
            if preprocessor is None:
                logging.info(f"Skipping preprocessing for raw model: {model}")
                continue

            preprocessor_object = preprocessor.fit(model_train_df)

            # Apply same label encoding to test data
            self.build_model_preprocessor(model_test_df, model, is_train=False)

            transformed_train = preprocessor_object.transform(model_train_df)
            transformed_test = preprocessor_object.transform(model_test_df)

            train_arr = np.c_[transformed_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test, np.array(target_feature_test_df)]

            # Save transformed numpy arrays only once
            if model == "linear":  # save under one model only
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            
            # self.load_transformed_test_data()
            preprocessor_path = self.data_transformation_config.transformed_object_file_paths[model]
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            save_object(preprocessor_path, preprocessor_object)
            transformed_objects[model] = preprocessor_path

        return DataTransformationArtifact(
            transformed_object_file_paths=transformed_objects,
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
        )
    
    