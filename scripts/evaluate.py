"""Evaluation script for story generation models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data.dataset import StoryDataset
from src.evaluation.metrics import StoryEvaluationMetrics
from src.models.story_generator import StoryGenerator
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


def evaluate_model(
    model_path: str,
    test_dataset: StoryDataset,
    config: StoryGenerationConfig,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model.
        test_dataset: Test dataset.
        config: Configuration object.
        num_samples: Number of samples to evaluate (None for all).
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating model at {model_path}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get test data
    test_data = test_dataset.get_dataset()["test"]
    prompts = test_data["prompt"]
    references = test_data["story"]
    
    if num_samples:
        prompts = prompts[:num_samples]
        references = references[:num_samples]
    
    logger.info(f"Evaluating on {len(prompts)} samples")
    
    # Generate stories
    generator = StoryGenerator(
        model_name=model_path,
        device=device,
        seed=config.system.seed,
    )
    
    generated_stories = []
    for prompt in prompts:
        story = generator.generate_story(
            prompt,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
            no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
        )
        generated_stories.append(story)
    
    # Evaluate
    evaluator = StoryEvaluationMetrics(tokenizer=tokenizer, device=device)
    results = evaluator.evaluate_stories(
        generated_stories,
        references,
        model=model,
    )
    
    return results


def evaluate_pretrained_model(
    model_name: str,
    test_dataset: StoryDataset,
    config: StoryGenerationConfig,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a pre-trained model.
    
    Args:
        model_name: Name of the pre-trained model.
        test_dataset: Test dataset.
        config: Configuration object.
        num_samples: Number of samples to evaluate (None for all).
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating pre-trained model: {model_name}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get test data
    test_data = test_dataset.get_dataset()["test"]
    prompts = test_data["prompt"]
    references = test_data["story"]
    
    if num_samples:
        prompts = prompts[:num_samples]
        references = references[:num_samples]
    
    logger.info(f"Evaluating on {len(prompts)} samples")
    
    # Generate stories
    generator = StoryGenerator(
        model_name=model_name,
        device=device,
        seed=config.system.seed,
    )
    
    generated_stories = []
    for prompt in prompts:
        story = generator.generate_story(
            prompt,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
            no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
        )
        generated_stories.append(story)
    
    # Evaluate
    evaluator = StoryEvaluationMetrics(tokenizer=tokenizer, device=device)
    results = evaluator.evaluate_stories(
        generated_stories,
        references,
        model=model,
    )
    
    return results


def compare_models(
    model_paths: List[str],
    test_dataset: StoryDataset,
    config: StoryGenerationConfig,
    num_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models.
    
    Args:
        model_paths: List of model paths to compare.
        test_dataset: Test dataset.
        config: Configuration object.
        num_samples: Number of samples to evaluate (None for all).
        
    Returns:
        Dictionary mapping model names to evaluation results.
    """
    results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).name
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            if Path(model_path).exists():
                model_results = evaluate_model(
                    model_path,
                    test_dataset,
                    config,
                    num_samples,
                )
            else:
                # Treat as pre-trained model name
                model_results = evaluate_pretrained_model(
                    model_path,
                    test_dataset,
                    config,
                    num_samples,
                )
            
            results[model_name] = model_results
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary of evaluation results.
    """
    if not results:
        logger.warning("No results to display")
        return
    
    # Get all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Print header
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Print table header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        header += f"{metric:<15}"
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, model_results in results.items():
        row = f"{model_name:<20}"
        for metric in all_metrics:
            value = model_results.get(metric, "N/A")
            if isinstance(value, float):
                row += f"{value:<15.4f}"
            else:
                row += f"{str(value):<15}"
        print(row)
    
    print("=" * 80)


def save_results(
    results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Save evaluation results to file.
    
    Args:
        results: Evaluation results.
        output_path: Path to save results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate story generation models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model (or model name for pre-trained)",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        help="Multiple model paths to compare",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to test dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results",
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
    if args.num_samples:
        config.evaluation.num_samples = args.num_samples
    if args.seed:
        config.system.seed = args.seed
    
    # Set up logging
    setup_logging(config.system.log_level)
    
    # Prepare dataset
    dataset = StoryDataset(
        data_path=config.data.data_path,
        tokenizer=None,  # Will be loaded with model
        max_length=config.data.max_length,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )
    
    # Evaluate models
    if args.model_paths:
        # Compare multiple models
        results = compare_models(
            args.model_paths,
            dataset,
            config,
            args.num_samples,
        )
    elif args.model_path:
        # Evaluate single model
        if Path(args.model_path).exists():
            results = {
                Path(args.model_path).name: evaluate_model(
                    args.model_path,
                    dataset,
                    config,
                    args.num_samples,
                )
            }
        else:
            results = {
                args.model_path: evaluate_pretrained_model(
                    args.model_path,
                    dataset,
                    config,
                    args.num_samples,
                )
            }
    else:
        # Evaluate default models
        default_models = ["gpt2", "gpt2-medium"]
        results = {}
        for model_name in default_models:
            results[model_name] = evaluate_pretrained_model(
                model_name,
                dataset,
                config,
                args.num_samples,
            )
    
    # Print and save results
    print_results(results)
    
    if args.output_file:
        save_results(results, args.output_file)


if __name__ == "__main__":
    main()
