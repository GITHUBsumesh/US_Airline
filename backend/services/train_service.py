import sys
from backend.utils.ml_path_importer import add_ml_path
add_ml_path()
from ml.src.pipeline.training_pipeline import TrainingPipeline

def run_training_pipeline() -> str:
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return "✅ Model training completed."
    except Exception as e:
        print(f"[TRAINING SERVICE ERROR] {e}", file=sys.stderr)
        raise RuntimeError(f"❌ Training failed: {str(e)}")