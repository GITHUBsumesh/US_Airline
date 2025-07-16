from ml.src.utils.path_config import ML_ROOT
import pandas as pd
from src.constants.training_pipeline import SAVED_MODEL_DIR

import os
import sys

from src.exception.exception import AirLineException
from src.logging.logger import logging
from sklearn.pipeline import Pipeline
import numpy as np
class AirLineModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise AirLineException(e,sys)
        
    def predict(self, x):
        try:

            if not isinstance(x, pd.DataFrame):
                raise ValueError("Expected pandas DataFrame but received different type.")

            x.columns = x.columns.str.strip()
            for col in x.select_dtypes(include='object').columns:
                x[col] = x[col].astype(str).str.strip()
            print(self.model)
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)

            return y_hat

        except Exception as e:
            raise AirLineException(e, sys)
