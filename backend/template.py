import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "backend"

list_of_files = [
    "app.py",
    "db.py",
    "__init__.py",

    "models/__init__.py",
    "models/prediction.py",

    "schemas/__init__.py",
    "schemas/prediction.py",

    "routes/__init__.py",
    "routes/predict.py",
    "routes/train.py",

    "services/__init__.py",
    "services/predict_service.py",
    "services/train_service.py",

    "utils/__init__.py",
    "utils/logger.py",

    "templates/table.html",
    "requirements.txt",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
