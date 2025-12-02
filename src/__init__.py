"""Story Generation System - A modern story generation system using transformer-based language models."""

__version__ = "0.1.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from src.models.story_generator import StoryGenerator, StoryGeneratorConfig
from src.data.dataset import StoryDataset
from src.evaluation.metrics import StoryEvaluationMetrics
from src.utils.config import StoryGenerationConfig

__all__ = [
    "StoryGenerator",
    "StoryGeneratorConfig", 
    "StoryDataset",
    "StoryEvaluationMetrics",
    "StoryGenerationConfig",
]
