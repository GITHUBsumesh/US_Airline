from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import sys
import os

from backend.db.sync_db import SessionLocal  # ✅ Use sync DB session
from backend.services.predict_service import run_prediction

router = APIRouter()
# templates = Jinja2Templates(directory="templates")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.post("/", response_class=HTMLResponse)
def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)
        assert isinstance(df, pd.DataFrame)

        # Use sync DB session
        with SessionLocal() as session:
            df, html_table = run_prediction(df, session)

        # Save predictions to CSV
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # Return predictions as HTML table
        return templates.TemplateResponse("table.html", {
            "request": request,
            "table": html_table
        })

    except Exception as e:
        print(f"[PREDICT ROUTE ERROR] {e}", file=sys.stderr)
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })
