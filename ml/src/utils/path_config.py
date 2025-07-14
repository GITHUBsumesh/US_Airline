# ml/src/utils/path_config.py

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get project root path from .env
ML_ROOT = os.getenv("ML_ROOT")

if not ML_ROOT:
    raise EnvironmentError("ML_ROOT is not set in the .env file.")

# Set working directory to the root of the project
os.chdir(ML_ROOT)

# Ensure root is in sys.path for imports like `from src...`
if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)
