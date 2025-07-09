import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name="datascience"

list_of_files=[
    "src/__init__.py",
    "src/components/__init__.py",
    "src/cloud/__init__.py",
    "src/utils/__init__.py",
    "src/utils/main_utils/__init__.py",
    "src/utils/ml_utils/__init__.py",
    "src/utils/ml_utils/metric/__init__.py ",
    "src/utils/ml_utils/model/__init__.py ",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/exception/exception.py",
    "src/exception/__init__.py",
    "src/logging/__init__.py",
    "src/logging/logger.py",
    "src/pipeine/__init__.py",
    "src/pipeine/training_pipeline.py",
    "src/constants/__init__.py",
    "src/constants/training_pipeline/__init__.py",
    "data_scheme/scheme.yml",
    "push_data.py",
]


for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file : {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
            

