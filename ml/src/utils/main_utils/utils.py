import yaml
from ml.src.utils.path_config import ML_ROOT

from src.exception.exception import AirLineException
from src.logging.logger import logging
import os,sys
import numpy as np
#import dill
import traceback
import pickle
from typing import Dict
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,RandomizedSearchCV
import time

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AirLineException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise AirLineException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AirLineException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise AirLineException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise AirLineException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise AirLineException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, object], param: Dict[str, Dict],pathName:str) -> Dict[str, float]:
    report = {}
    cv_results = {}
    for model_name, model in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}")
            start_time = time.time()
            if model_name == "catboost":
                train_pool = Pool(X_train, y_train)
                test_pool = Pool(X_test, y_test)
                model.fit(train_pool, verbose=False)

            elif model_name == "dnn":
                rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                for train_index, val_index in rkf.split(X_train):
                    model.fit(X_train[train_index], y_train[train_index],
                              validation_data=(X_train[val_index], y_train[val_index]),
                              epochs=10, callbacks=[], verbose=0)

            elif model_name in param:
                rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                grid_search = RandomizedSearchCV(model, param[model_name],
                                                 cv=rkf, n_jobs=-1, n_iter=10,
                                                 verbose=1 if model_name in ["xgb", "rf"] else 0,
                                                 scoring='r2', return_train_score=True)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                model.fit(X_train, y_train)
                cv_results[model_name] = grid_search.cv_results_
            else:
                model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            duration = time.time() - start_time
            logging.info(f"Completed model: {model_name} in {duration:.2f} seconds with R2: {test_score:.4f}")

            report[model_name] = test_score

        except Exception as e:
            logging.error(f"Failed to evaluate {model_name}: {str(e)} {traceback.format_exc()}")
            continue

    # Save CV report
    # os.makedirs("model_trainer", exist_ok=True)
    os.makedirs(pathName, exist_ok=True)
    with open(f"{pathName}/cvreport.yaml", "w") as file:
        yaml.dump({k: {key: val.tolist() if hasattr(val, 'tolist') else val for key, val in v.items()} for k, v in cv_results.items()}, file)

    return report