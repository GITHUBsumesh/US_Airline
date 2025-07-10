from src.constants.training_pipeline import SAVED_MODEL_DIR

import os
import sys

from src.exception.exception import AirLineException
from src.logging.logger import logging

class AirLineModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise AirLineException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise AirLineException(e,sys)