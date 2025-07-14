from fastapi import FastAPI
from ml.src.utils.path_config import ML_ROOT

from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from backend.routes import predict, train
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml")))

app = FastAPI(title="Flight Delay Prediction API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template rendering (Jinja2)
templates = Jinja2Templates(directory="templates")

# Register prediction route
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(train.router, prefix="/api/train", tags=["Training"])