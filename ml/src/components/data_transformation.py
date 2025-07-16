import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict
from ml.src.utils.path_config import ML_ROOT

from src.constants.training_pipeline import TARGET_COLUMN,ALL_MODELS,ORDINAL_MODELS,OHE_MODELS,SKIP_SCALING_MODELS,RAW_MODELS
from src.constants.training_pipeline import DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS,DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import AirLineException 
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data,save_object


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


    def get_column_transformer(self, df: pd.DataFrame, model_name: str) -> ColumnTransformer:
        """
        Builds a ColumnTransformer that:
        - One-hot encodes 'carrier'
        - Ordinal encodes 'airport'
        - Imputes and scales numerical columns
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

        # Ordinal Encode 'airport'
        if 'airport' in df.columns:
            ordinal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(**DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS)),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            transformers.append(("airport_ordinal", ordinal_pipeline, ["airport"]))

        return ColumnTransformer(transformers=transformers)

    def build_model_preprocessor(self, df: pd.DataFrame, model_name: str) -> Pipeline:
        if model_name in RAW_MODELS:
            return None

        column_transformer = self.get_column_transformer(df, model_name)
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

        all_models = OHE_MODELS + ORDINAL_MODELS + RAW_MODELS
        transformed_objects: Dict[str, str] = {}

        for model in ALL_MODELS:
            logging.info(f"Processing for model: {model}")
            model_train_df = input_feature_train_df.copy()
            model_test_df = input_feature_test_df.copy()

             # Build and fit transformer on training data
            preprocessor = self.build_model_preprocessor(model_train_df, model)
            if preprocessor is None:
                logging.info(f"Skipping preprocessing for raw model: {model}")
                continue

            preprocessor_object = preprocessor.fit(model_train_df)

            transformed_train = preprocessor_object.transform(model_train_df)
            transformed_test = preprocessor_object.transform(model_test_df)

            train_arr = np.c_[transformed_train, train_df[TARGET_COLUMN].values]
            test_arr = np.c_[transformed_test, test_df[TARGET_COLUMN].values]


            # Save transformed numpy arrays only once
            # if model == "linear":  # save under one model only
            #     save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            #     save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            model_train_path = self.data_transformation_config.transformed_train_file_paths[model]
            model_test_path = self.data_transformation_config.transformed_test_file_paths[model]

            os.makedirs(os.path.dirname(model_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(model_test_path), exist_ok=True)

            save_numpy_array_data(model_train_path, train_arr)
            save_numpy_array_data(model_test_path, test_arr)

            # self.load_transformed_test_data()
            preprocessor_path = self.data_transformation_config.transformed_object_file_paths[model]
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            save_object(preprocessor_path, preprocessor_object)
            transformed_objects[model] = preprocessor_path

        return DataTransformationArtifact(
            transformed_object_file_paths=transformed_objects,
            transformed_train_file_paths=self.data_transformation_config.transformed_train_file_paths,
            transformed_test_file_paths=self.data_transformation_config.transformed_test_file_paths,
        )
    
    