"""Configuration management for story generation system."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the language model."""
    
    name: str = "gpt2"
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 2
    do_sample: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    data_path: Optional[str] = None
    max_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    num_samples: int = 100
    metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "bleu", "rouge", "bert_score", "diversity", "coherence"
    ])
    batch_size: int = 8


@dataclass
class GenerationConfig:
    """Configuration for story generation."""
    
    max_length: int = 200
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 2
    num_return_sequences: int = 1
    do_sample: bool = True


@dataclass
class SystemConfig:
    """System-level configuration."""
    
    seed: int = 42
    device: Optional[str] = None
    mixed_precision: bool = False
    num_gpus: int = 1
    log_level: str = "INFO"


@dataclass
class StoryGenerationConfig:
    """Main configuration class for the story generation system."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate splits sum to 1.0
        total_split = (
            self.data.train_split + 
            self.data.val_split + 
            self.data.test_split
        )
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Data splits must sum to 1.0, got {total_split}"
            )
        
        # Validate temperature
        if self.generation.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Validate top_p
        if not 0 < self.generation.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        
        # Validate top_k
        if self.generation.top_k <= 0:
            raise ValueError("top_k must be positive")
    
    def to_dict(self) -> Dict[str, Dict[str, Union[str, int, float, bool, List]]]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature,
                "top_k": self.model.top_k,
                "top_p": self.model.top_p,
                "repetition_penalty": self.model.repetition_penalty,
                "no_repeat_ngram_size": self.model.no_repeat_ngram_size,
                "do_sample": self.model.do_sample,
            },
            "data": {
                "data_path": self.data.data_path,
                "max_length": self.data.max_length,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "test_split": self.data.test_split,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_grad_norm": self.training.max_grad_norm,
                "save_steps": self.training.save_steps,
                "eval_steps": self.training.eval_steps,
                "logging_steps": self.training.logging_steps,
                "output_dir": self.training.output_dir,
                "save_total_limit": self.training.save_total_limit,
                "load_best_model_at_end": self.training.load_best_model_at_end,
                "metric_for_best_model": self.training.metric_for_best_model,
                "greater_is_better": self.training.greater_is_better,
            },
            "evaluation": {
                "num_samples": self.evaluation.num_samples,
                "metrics": self.evaluation.metrics,
                "batch_size": self.evaluation.batch_size,
            },
            "generation": {
                "max_length": self.generation.max_length,
                "temperature": self.generation.temperature,
                "top_k": self.generation.top_k,
                "top_p": self.generation.top_p,
                "repetition_penalty": self.generation.repetition_penalty,
                "no_repeat_ngram_size": self.generation.no_repeat_ngram_size,
                "num_return_sequences": self.generation.num_return_sequences,
                "do_sample": self.generation.do_sample,
            },
            "system": {
                "seed": self.system.seed,
                "device": self.system.device,
                "mixed_precision": self.system.mixed_precision,
                "num_gpus": self.system.num_gpus,
                "log_level": self.system.log_level,
            },
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save the configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "StoryGenerationConfig":
        """Load configuration from file.
        
        Args:
            path: Path to the configuration file.
            
        Returns:
            Loaded configuration object.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Create configuration object
        config = cls()
        
        # Update with loaded values
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if "evaluation" in config_dict:
            for key, value in config_dict["evaluation"].items():
                if hasattr(config.evaluation, key):
                    setattr(config.evaluation, key, value)
        
        if "generation" in config_dict:
            for key, value in config_dict["generation"].items():
                if hasattr(config.generation, key):
                    setattr(config.generation, key, value)
        
        if "system" in config_dict:
            for key, value in config_dict["system"].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    @classmethod
    def from_omegaconf(cls, config: DictConfig) -> "StoryGenerationConfig":
        """Create configuration from OmegaConf DictConfig.
        
        Args:
            config: OmegaConf configuration object.
            
        Returns:
            StoryGenerationConfig object.
        """
        # Convert OmegaConf to dict
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # Create new config object
        story_config = cls()
        
        # Update with OmegaConf values
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(story_config.model, key):
                    setattr(story_config.model, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(story_config.data, key):
                    setattr(story_config.data, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(story_config.training, key):
                    setattr(story_config.training, key, value)
        
        if "evaluation" in config_dict:
            for key, value in config_dict["evaluation"].items():
                if hasattr(story_config.evaluation, key):
                    setattr(story_config.evaluation, key, value)
        
        if "generation" in config_dict:
            for key, value in config_dict["generation"].items():
                if hasattr(story_config.generation, key):
                    setattr(story_config.generation, key, value)
        
        if "system" in config_dict:
            for key, value in config_dict["system"].items():
                if hasattr(story_config.system, key):
                    setattr(story_config.system, key, value)
        
        return story_config


def create_default_config() -> StoryGenerationConfig:
    """Create a default configuration.
    
    Returns:
        Default StoryGenerationConfig object.
    """
    return StoryGenerationConfig()


def create_config_from_dict(config_dict: Dict) -> StoryGenerationConfig:
    """Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary.
        
    Returns:
        StoryGenerationConfig object.
    """
    config = StoryGenerationConfig()
    
    # Update with provided values
    for section_name, section_config in config_dict.items():
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            for key, value in section_config.items():
                if hasattr(section, key):
                    setattr(section, key, value)
    
    return config
