from fastapi import APIRouter, Response
import sys
from backend.services.train_service import run_training_pipeline

router = APIRouter()

@router.get("/")
def train_model():
    try:
        message = run_training_pipeline()
        return Response(content=message, status_code=200)
    except Exception as e:
        print(f"[TRAIN ROUTE ERROR] {e}", file=sys.stderr)
        return Response(content=str(e), status_code=500)