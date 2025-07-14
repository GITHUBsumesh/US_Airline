import sys
import os

def add_ml_path():
    ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml"))
    if ml_path not in sys.path:
        sys.path.append(ml_path)
