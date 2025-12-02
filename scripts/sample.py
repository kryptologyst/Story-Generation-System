"""Sampling utilities for story generation."""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.models.story_generator import StoryGenerator, StoryGeneratorConfig
from src.utils.config import StoryGenerationConfig

logger = logging.getLogger(__name__)


def sample_stories(
    config: StoryGenerationConfig,
    prompts: List[str],
    output_dir: Optional[str] = None,
    num_samples_per_prompt: int = 5,
) -> Dict[str, List[str]]:
    """Generate multiple story samples for given prompts.
    
    Args:
        config: Configuration object.
        prompts: List of prompts to generate stories from.
        output_dir: Directory to save generated stories.
        num_samples_per_prompt: Number of samples per prompt.
        
    Returns:
        Dictionary mapping prompts to generated stories.
    """
    # Create story generator
    generator_config = StoryGeneratorConfig(
        model_name=config.model.name,
        max_length=config.generation.max_length,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        repetition_penalty=config.generation.repetition_penalty,
        no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
        seed=config.system.seed,
    )
    
    generator = StoryGenerator(
        model_name=generator_config.model_name,
        seed=generator_config.seed,
    )
    
    results = {}
    
    for prompt in prompts:
        logger.info(f"Generating stories for prompt: {prompt[:50]}...")
        
        # Generate multiple stories for this prompt
        stories = generator.generate_multiple_stories(
            prompt,
            num_stories=num_samples_per_prompt,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
            no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
        )
        
        results[prompt] = stories
        
        # Save individual stories if output directory is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename from prompt
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            
            for i, story in enumerate(stories):
                story_file = output_path / f"{safe_prompt}_story_{i:02d}.txt"
                with open(story_file, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generated Story:\n{story}\n")
    
    return results


def interactive_generation(config: StoryGenerationConfig) -> None:
    """Interactive story generation session.
    
    Args:
        config: Configuration object.
    """
    # Create story generator
    generator_config = StoryGeneratorConfig(
        model_name=config.model.name,
        max_length=config.generation.max_length,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        repetition_penalty=config.generation.repetition_penalty,
        no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
        seed=config.system.seed,
    )
    
    generator = StoryGenerator(
        model_name=generator_config.model_name,
        seed=generator_config.seed,
    )
    
    print("Story Generation Interactive Mode")
    print("=" * 40)
    print("Enter prompts to generate stories. Type 'quit' to exit.")
    print("Type 'continue' to continue the last generated story.")
    print()
    
    last_story = None
    
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            
            if prompt.lower() == "quit":
                break
            
            if prompt.lower() == "continue" and last_story:
                print("Continuing last story...")
                story = generator.continue_story(
                    last_story,
                    continuation_length=100,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    top_p=config.generation.top_p,
                )
            elif prompt:
                story = generator.generate_story(
                    prompt,
                    max_length=config.generation.max_length,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    top_p=config.generation.top_p,
                    repetition_penalty=config.generation.repetition_penalty,
                    no_repeat_ngram_size=config.generation.no_repeat_ngram_size,
                )
            else:
                print("Please enter a valid prompt.")
                continue
            
            print("\n" + "=" * 60)
            print("GENERATED STORY:")
            print("=" * 60)
            print(story)
            print("=" * 60)
            print()
            
            last_story = story
            
            # Ask if user wants to save the story
            save = input("Save this story? (y/n): ").strip().lower()
            if save == "y":
                output_dir = Path("assets/generated")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"story_{random.randint(1000, 9999)}.txt"
                filepath = output_dir / filename
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generated Story:\n{story}\n")
                
                print(f"Story saved to {filepath}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            print(f"Error: {e}")


def batch_generation(
    config: StoryGenerationConfig,
    input_file: str,
    output_file: str,
    num_samples_per_prompt: int = 3,
) -> None:
    """Generate stories from prompts in a file.
    
    Args:
        config: Configuration object.
        input_file: Path to file containing prompts (one per line).
        output_file: Path to save generated stories.
        num_samples_per_prompt: Number of samples per prompt.
    """
    # Read prompts
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Generate stories
    results = sample_stories(
        config,
        prompts,
        num_samples_per_prompt=num_samples_per_prompt,
    )
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt, stories in results.items():
            f.write(f"PROMPT: {prompt}\n")
            f.write("=" * 80 + "\n")
            
            for i, story in enumerate(stories):
                f.write(f"\nSTORY {i + 1}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{story}\n")
                f.write("-" * 40 + "\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"Generated stories saved to {output_file}")


def main() -> None:
    """Main function for sampling script."""
    parser = argparse.ArgumentParser(description="Generate story samples")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch", "sample"],
        default="interactive",
        help="Generation mode",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Prompts for story generation",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file with prompts (for batch mode)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for generated stories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/generated",
        help="Output directory for generated stories",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per prompt",
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
    
    # Override seed if provided
    config.system.seed = args.seed
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if args.mode == "interactive":
        interactive_generation(config)
    
    elif args.mode == "batch":
        if not args.input_file:
            raise ValueError("Input file required for batch mode")
        
        output_file = args.output_file or "generated_stories.txt"
        batch_generation(
            config,
            args.input_file,
            output_file,
            num_samples_per_prompt=args.num_samples,
        )
    
    elif args.mode == "sample":
        if not args.prompts:
            # Use default prompts
            prompts = [
                "Once upon a time in a faraway kingdom, there was a princess who",
                "In a small town by the sea, an old lighthouse keeper",
                "The last robot on Earth",
                "A detective walked into a coffee shop",
                "The time traveler arrived in the year 1920",
            ]
        else:
            prompts = args.prompts
        
        results = sample_stories(
            config,
            prompts,
            output_dir=args.output_dir,
            num_samples_per_prompt=args.num_samples,
        )
        
        # Print results
        for prompt, stories in results.items():
            print(f"\nPROMPT: {prompt}")
            print("=" * 80)
            for i, story in enumerate(stories):
                print(f"\nSTORY {i + 1}:")
                print("-" * 40)
                print(story)
                print("-" * 40)


if __name__ == "__main__":
    main()
