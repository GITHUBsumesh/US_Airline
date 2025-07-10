from src.exception.exception import AirLineException
from src.logging.logger import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import sys
from src.entity.artifact_entity import RegressionMetricArtifact

def get_regression_score(y_true, y_pred):
    try:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        regression_metric = RegressionMetricArtifact(
            r2_score=r2,
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse
        )
        return regression_metric
    except Exception as e:
        raise AirLineException(e, sys)
