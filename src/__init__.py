"""
Text-to-SQL System

Production-grade Text-to-SQL system using fine-tuned transformer models.
"""

try:
    from src.data_preparation import SpiderDataProcessor, WikiSQLDataProcessor
    from src.model_training import (
        ModelConfig,
        TrainingConfig,
        Text2SQLTrainer,
        CustomTrainingLoop,
        get_gpu_memory_info
    )
except ImportError:
    # Dependencies missing (Lightweight mode)
    SpiderDataProcessor = None
    WikiSQLDataProcessor = None
    ModelConfig = None
    TrainingConfig = None
try:
    from src.model_inference import (
        Text2SQLInference,
        GenerationConfig,
        PredictionResult,
        SchemaHelper,
        load_model
    )
    from src.utils import (
        normalize_sql_query,
        extract_sql_keywords,
        calculate_query_complexity,
        serialize_schema_for_prompt,
        create_model_input
    )
except ImportError:
    Text2SQLInference = None
    GenerationConfig = None

__version__ = "1.0.0"
__all__ = [
    # Data Processing
    "SpiderDataProcessor",
    "WikiSQLDataProcessor",
    # Training
    "ModelConfig",
    "TrainingConfig",
    "Text2SQLTrainer",
    "CustomTrainingLoop",
    "get_gpu_memory_info",
    # Inference
    "Text2SQLInference",
    "GenerationConfig",
    "PredictionResult",
    "SchemaHelper",
    "load_model",
    # Utilities
    "normalize_sql_query",
    "extract_sql_keywords",
    "calculate_query_complexity",
    "serialize_schema_for_prompt",
    "create_model_input"
]
