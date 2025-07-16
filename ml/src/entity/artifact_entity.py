from dataclasses import dataclass
from typing import Dict

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_paths: Dict[str, str]
    transformed_train_file_paths: Dict[str, str]
    transformed_test_file_paths: Dict[str, str]


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    
@dataclass
class RegressionMetricArtifact:
    r2_score: float
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
