import pandas as pd
import os
from backend.models.prediction import Prediction
from backend.utils.ml_path_importer import add_ml_path
add_ml_path()
from ml.src.utils.main_utils.utils import load_object
from ml.src.utils.ml_utils.model.estimator import AirLineModel
from sqlalchemy.orm import Session  # ✅ Use sync Session

# Get path to project root (US_Airline/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# File paths
preprocessor_path = os.path.join(BASE_DIR, "ml", "final_model", "preprocessor.pkl")
model_path = os.path.join(BASE_DIR, "ml", "final_model", "model.pkl")


def run_prediction(df: pd.DataFrame, session: Session):
    # Load artifacts
    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)

    # Inference
    airline_model = AirLineModel(preprocessor=preprocessor, model=model)
    y_pred = airline_model.predict(df)
    df['predicted_column'] = y_pred

    # Store predictions
    for _, row in df.iterrows():
        prediction = Prediction(
            predictedValue=int(row['predicted_column']),
            inputFeatures=row.drop('predicted_column').to_dict()
        )
        session.add(prediction)

    session.commit()

    # Return HTML table
    html_table = df.to_html(classes="table table-striped", index=False)
    return df, html_table
