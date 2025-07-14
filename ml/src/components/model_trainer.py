import os
import sys
from ml.src.utils.path_config import ML_ROOT
import pandas as pd
from src.exception.exception import AirLineException 
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
import yaml

from src.constants.training_pipeline import MODEL_FILE_NAME
from src.utils.ml_utils.model.estimator import AirLineModel
from src.utils.main_utils.utils import save_object,load_object
from src.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.metric.regression_metric import get_regression_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from typing import Dict
import mlflow
from urllib.parse import urlparse

import dagshub
tracking_uri = mlflow.get_tracking_uri()
parsed_uri = urlparse(tracking_uri)
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_REPO_OWNER_NAME = os.getenv("DAGSHUB_REPO_OWNER_NAME")
dagshub.init(repo_owner=DAGSHUB_REPO_NAME, repo_name=DAGSHUB_REPO_OWNER_NAME, mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise AirLineException(e, sys)

    def build_dnn(self, input_dim=None, units=128, dropout_rate=0.3, learning_rate=0.001):
        def create_model():
            model = Sequential()
            model.add(Dense(units, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units // 2, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        return KerasRegressor(
            model=create_model,
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping]
        )


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1]
            )
            input_dim = x_train.shape[1]
            models: Dict[str, object] = {
                "linear": LinearRegression(),
                "ridge": Ridge(),
                "lasso": Lasso(),
                "knn": KNeighborsRegressor(),
                "svr": SVR(),
                "dnn": self.build_dnn(input_dim=input_dim),
                "rf": RandomForestRegressor(verbose=1, random_state=42),
                "extratrees": ExtraTreesRegressor(random_state=42),
                "xgb": XGBRegressor(verbosity=1, random_state=42),
                "lgbm": LGBMRegressor(random_state=42),
                "gbr": GradientBoostingRegressor(random_state=42),
                "catboost": CatBoostRegressor(verbose=0, random_state=42)
            }

            params: Dict[str, Dict] = {
                "ridge": {"alpha": [0.01, 0.1, 1.0]},
                "lasso": {"alpha": [0.01, 0.1, 1.0]},
                "knn": {"n_neighbors": [3, 5, 7]},
                "svr": {"C": [1, 10], "gamma": ['scale', 'auto']},
                "dnn": {
                    "model__units": [64, 128],
                    "model__dropout_rate": [0.2, 0.3],
                    "epochs": [50, 100],
                    "batch_size": [32, 64]
                },
                "rf": {"n_estimators": [100, 200]},
                "extratrees": {"n_estimators": [100, 200]},
                "xgb": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]},
                "lgbm": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]},
                "gbr": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]},
                "catboost": {"iterations": [100, 200], "learning_rate": [0.01, 0.1]}
            }
            pathName=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            model_report: dict = evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
                pathName=pathName
            )

            os.makedirs(f"{pathName}/trained_models", exist_ok=True)

            for model_name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                metrics = get_regression_score(y_test, y_pred)
                logging.info(f"Model: {model_name}, Metrics: {metrics}")
                with open(f"{pathName}/trained_models/{model_name}_metrics.yaml", "w") as file:
                    yaml.dump(metrics, file)

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Best Model: {best_model_name}, Train Metrics: {train_metric}, Test Metrics: {test_metric}")
            
            self.track_mlflow(best_model, test_metric, best_model_name)
            # self.track_mlflow(best_model, train_metric)
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_paths[best_model_name])

            # os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            airline_model = AirLineModel(preprocessor=preprocessor, model=best_model)
            save_object(os.path.join(
            self.model_trainer_config.trained_model_file_path, 
            MODEL_FILE_NAME
        ), airline_model)
            save_object("final_model/model.pkl",best_model)
            save_object("final_model/preprocessor.pkl", preprocessor)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            raise AirLineException(e, sys)

           
    def track_mlflow(self, best_model, regressionmetric, best_model_name):

        mse = regressionmetric.mean_squared_error
        mae = regressionmetric.mean_absolute_error
        rmse = regressionmetric.root_mean_squared_error
        r2 = regressionmetric.r2_score

        with mlflow.start_run():
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("model_name", best_model_name)
            model_path = os.path.normpath("model")
            mlflow.sklearn.log_model(best_model,"model")

            # if "dagshub" in parsed_uri.netloc or parsed_uri.scheme == "file":
            #     # ✅ Skip registry for DagsHub or local file-based tracking
            #     mlflow.sklearn.log_model(best_model, "model")
            # else:
            #     # ✅ Safe to register
            #     mlflow.sklearn.log_model(best_model, "model", registered_model_name="AirlineModel")

    