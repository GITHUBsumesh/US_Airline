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

            # Clean input
            x.columns = x.columns.str.strip()
            for col in x.select_dtypes(include='object').columns:
                x[col] = x[col].astype(str).str.strip()
            print(self.model)
            # --- Conditional pipeline usage ---
            # if self.model_name == "linear":
            #     # Model already contains preprocessor inside pipeline
            #     y_hat = self.model.predict(x)
            # else:
            # Apply preprocessor manually
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)

            return y_hat

        except Exception as e:
            raise AirLineException(e, sys)
        
            # print("Expected columns:", list(self.preprocessor.feature_names_in_))
            # print("Actual columns in test data:", list(x.columns))
            # print(x[1:5])
            # print("Missing columns:", set(self.preprocessor.feature_names_in_) - set(x.columns))
            # x_transform = self.preprocessor.transform(x)
            # # print(x_transform[1:5])
            # columns = self.preprocessor.get_feature_names_out()
            # print("Transformed columns:", columns)
            # print("Transformed data shape:", x_transform.shape)
            # print("Transformed x columns:", len(x.columns))
            # print("Transformed x_transform columns:", x_transform)
            # df_transformed = pd.DataFrame(x_transform, columns=columns)
            # assert isinstance(df_transformed, pd.DataFrame) and not df_transformed.empty, "Transformed data is empty or not a DataFrame"
            # y_hat = self.model.predict(df_transformed)
    # def predict(self, x):
    #     try:
    #         # Ensure input is a DataFrame
    #         # if isinstance(x, np.ndarray):
    #         #     raise ValueError("Expected pandas DataFrame but received numpy array.")
    #         print(self.model)
    #         expected_columns = list(self.preprocessor.feature_names_in_)
    #         # print("Expected columns:", expected_columns)
    #         # print("Input columns:", x.columns.tolist())
    #         # print("Input data shape:", len(x.columns))
    #         x = pd.DataFrame(x, columns=expected_columns)
    #         x.columns = x.columns.str.strip()
    #         for col in x.select_dtypes(include='object').columns:
    #             x[col] = x[col].astype(str).str.strip()

    #         # Build pipeline dynamically
    #         pipe = Pipeline([
    #             ('preprocessor', self.preprocessor),
    #             ('model', self.model)
    #         ])

    #         y_hat = pipe.predict(x)
    #         return y_hat

    #     except Exception as e:
    #         raise AirLineException(e, sys)

    # def predict(self, x):
    #     try:
    #         # ✅ Ensure x is a DataFrame
    #         if not isinstance(x, pd.DataFrame):
    #             raise ValueError("Input must be a pandas DataFrame")

    #         # ✅ Strip column names and string values
    #         x.columns = x.columns.str.strip()
    #         for col in x.select_dtypes(include='object').columns:
    #             x[col] = x[col].astype(str).str.strip()

    #         # ✅ Validate columns BEFORE transformation
    #         expected_cols = list(self.preprocessor.feature_names_in_)
    #         missing_cols = set(expected_cols) - set(x.columns)
    #         if missing_cols:
    #             raise ValueError(f"Missing required columns: {missing_cols}")

    #         # ✅ Apply transformation
    #         x_transform = self.preprocessor.transform(x)

    #         # ✅ Optional: create transformed DataFrame (for debugging)
    #         try:
    #             feature_names = self.preprocessor.get_feature_names_out()
    #             df_transformed = pd.DataFrame(x_transform, columns=feature_names)
    #         except Exception:
    #             # Fallback: if get_feature_names_out fails (e.g. pipeline without names), skip
    #             df_transformed = pd.DataFrame(x_transform)

    #         # ✅ Final model prediction
    #         y_hat = self.model.predict(df_transformed)
    #         return y_hat

    #     except Exception as e:
    #         raise AirLineException(e, sys)


    # def predict(self, x: pd.DataFrame):
    #     try:
    #         if not isinstance(x, pd.DataFrame):
    #             raise AirLineException("Input must be a pandas DataFrame", sys)

    #         x.columns = x.columns.str.strip()
    #         for col in x.select_dtypes(include='object').columns:
    #             x[col] = x[col].astype(str).str.strip()

    #         expected_raw_cols = set(self.preprocessor.feature_names_in_)
    #         actual_raw_cols = set(x.columns)
    #         missing_raw = expected_raw_cols - actual_raw_cols
    #         print("Expected columns:", list(expected_raw_cols))
    #         print("Actual columns in test data:", list(expected_raw_cols))
    #         print("Missing columns:", missing_raw)
    #         if missing_raw:
    #             raise AirLineException(f"Missing raw input columns: {missing_raw}", sys)

    #         # ✅ Transform
    #         x_transform = self.preprocessor.transform(x)

    #         # 🧪 Optional: wrap with column names
    #         try:
    #             feature_names = self.preprocessor.get_feature_names_out()
    #             x_transform = pd.DataFrame(x_transform, columns=feature_names)
    #         except:
    #             pass

    #         # ✅ Predict
    #         return self.model.predict(x_transform)

    #     except Exception as e:
    #         raise AirLineException(e, sys)

    # def predict(self, x: pd.DataFrame):
    #     try:
    #         # Strip and sanitize
    #         x.columns = x.columns.str.strip()
    #         for col in x.select_dtypes(include='object').columns:
    #             x[col] = x[col].astype(str).str.strip()

    #         # 🔒 Validate column names BEFORE transformation
    #         if not isinstance(x, pd.DataFrame):
    #             raise AirLineException("Input must be a pandas DataFrame", sys)

    #         expected_cols = set(self.preprocessor.feature_names_in_)
    #         actual_cols = set(x.columns)
    #         missing = expected_cols - actual_cols
    #         if missing:
    #             raise AirLineException(f"Missing raw input columns: {missing}", sys)

    #         # ✅ Transform after validation
    #         x_transformed = self.preprocessor.transform(x)

    #         # ✅ Wrap as DataFrame (optional)
    #         try:
    #             columns = self.preprocessor.get_feature_names_out()
    #             x_transformed = pd.DataFrame(x_transformed, columns=columns)
    #         except:
    #             pass

    #         return self.model.predict(x_transformed)

    #     except Exception as e:
    #         raise AirLineException(e, sys)
