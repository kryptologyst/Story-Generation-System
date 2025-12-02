"""Training script for story generation models."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src.data.dataset import StoryDataset
from src.utils.config import StoryGenerationConfig

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def prepare_model_and_tokenizer(
    model_name: str,
    device: Optional[torch.device] = None,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Prepare model and tokenizer for training.
    
    Args:
        model_name: Name of the pre-trained model.
        device: Device to load the model on.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if device:
        model = model.to(device)
    
    logger.info(f"Model loaded on device: {model.device}")
    return model, tokenizer


def prepare_dataset(
    data_path: Optional[str],
    tokenizer: GPT2Tokenizer,
    config: StoryGenerationConfig,
) -> StoryDataset:
    """Prepare dataset for training.
    
    Args:
        data_path: Path to the dataset.
        tokenizer: Tokenizer for preprocessing.
        config: Configuration object.
        
    Returns:
        Prepared dataset.
    """
    logger.info("Preparing dataset...")
    
    dataset = StoryDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )
    
    logger.info(f"Dataset prepared with {len(dataset.get_dataset()['train'])} training samples")
    return dataset


def train_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    dataset: StoryDataset,
    config: StoryGenerationConfig,
) -> None:
    """Train the model.
    
    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        dataset: Training dataset.
        config: Training configuration.
    """
    logger.info("Starting model training...")
    
    # Prepare datasets
    train_dataset = dataset.get_dataset()["train"]
    eval_dataset = dataset.get_dataset()["validation"]
    
    # Preprocess datasets
    train_dataset = train_dataset.map(
        dataset.preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    eval_dataset = eval_dataset.map(
        dataset.preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is not a masked language model
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.data.batch_size,
        per_device_eval_batch_size=config.data.batch_size,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        logging_dir=f"{config.training.output_dir}/logs",
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        save_total_limit=config.training.save_total_limit,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        learning_rate=config.training.learning_rate,
        fp16=config.system.mixed_precision and torch.cuda.is_available(),
        dataloader_num_workers=config.data.num_workers,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)
    
    logger.info(f"Training completed. Model saved to {config.training.output_dir}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train story generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = StoryGenerationConfig.load(config_path)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = StoryGenerationConfig()
    
    # Override with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model_name:
        config.model.name = args.model_name
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.seed:
        config.system.seed = args.seed
    
    # Set up logging
    setup_logging(config.system.log_level)
    
    # Set random seed
    torch.manual_seed(config.system.seed)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(
        config.model.name,
        device=device,
    )
    
    # Prepare dataset
    dataset = prepare_dataset(
        config.data.data_path,
        tokenizer,
        config,
    )
    
    # Train model
    train_model(model, tokenizer, dataset, config)


if __name__ == "__main__":
    main()
