#!/usr/bin/env python3
"""Example usage of the Story Generation System.

This script demonstrates the basic usage of the modernized story generation system.
"""

import logging
from pathlib import Path

from src.models.story_generator import StoryGenerator, StoryGeneratorConfig
from src.data.dataset import StoryDataset, create_sample_data
from src.evaluation.metrics import StoryEvaluationMetrics
from src.utils.config import StoryGenerationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("=" * 60)
    print("STORY GENERATION SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # 1. Create sample data
    print("\n1. Creating sample dataset...")
    data_path = "data/sample_stories.jsonl"
    create_sample_data(data_path, num_samples=50)
    print(f"Sample data created at: {data_path}")
    
    # 2. Load configuration
    print("\n2. Loading configuration...")
    config = StoryGenerationConfig()
    print(f"Model: {config.model.name}")
    print(f"Max length: {config.generation.max_length}")
    print(f"Temperature: {config.generation.temperature}")
    
    # 3. Create story generator
    print("\n3. Initializing story generator...")
    generator_config = StoryGeneratorConfig(
        model_name=config.model.name,
        seed=config.system.seed,
    )
    generator = StoryGenerator(
        model_name=generator_config.model_name,
        seed=generator_config.seed,
    )
    print("Story generator initialized successfully!")
    
    # 4. Generate stories
    print("\n4. Generating stories...")
    prompts = [
        "Once upon a time in a faraway kingdom, there was a princess who",
        "In a small town by the sea, an old lighthouse keeper",
        "The last robot on Earth",
    ]
    
    generated_stories = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        
        story = generator.generate_story(
            prompt,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
        )
        
        generated_stories.append(story)
        print(f"Generated story:\n{story}")
        print("-" * 40)
    
    # 5. Load dataset for evaluation
    print("\n5. Loading dataset for evaluation...")
    dataset = StoryDataset(data_path=data_path)
    test_stories = dataset.get_stories("test")[:3]  # Get 3 reference stories
    
    # 6. Evaluate generated stories
    print("\n6. Evaluating generated stories...")
    evaluator = StoryEvaluationMetrics()
    
    # Use generated stories and reference stories for evaluation
    evaluation_results = evaluator.evaluate_stories(
        generated_stories,
        test_stories,
    )
    
    print("Evaluation Results:")
    print(f"  Average word count: {evaluation_results['avg_word_count']:.1f}")
    print(f"  BLEU score: {evaluation_results.get('bleu', 'N/A')}")
    print(f"  ROUGE-1: {evaluation_results.get('rouge_rouge1', 'N/A')}")
    print(f"  ROUGE-2: {evaluation_results.get('rouge_rouge2', 'N/A')}")
    print(f"  ROUGE-L: {evaluation_results.get('rouge_rougeL', 'N/A')}")
    print(f"  Diversity (Distinct-1): {evaluation_results['distinct_1']:.4f}")
    print(f"  Coherence: {evaluation_results['coherence']:.4f}")
    
    # 7. Generate multiple stories from one prompt
    print("\n7. Generating multiple stories from one prompt...")
    prompt = "A detective walked into a coffee shop"
    multiple_stories = generator.generate_multiple_stories(
        prompt,
        num_stories=3,
        temperature=0.8,
    )
    
    print(f"Prompt: {prompt}")
    for i, story in enumerate(multiple_stories, 1):
        print(f"\nStory {i}:")
        print(story)
        print("-" * 40)
    
    # 8. Continue an existing story
    print("\n8. Continuing an existing story...")
    existing_story = "Once upon a time, there was a brave knight who lived in a castle."
    continued_story = generator.continue_story(
        existing_story,
        continuation_length=100,
        temperature=0.9,
    )
    
    print(f"Original story: {existing_story}")
    print(f"Continued story: {continued_story}")
    
    # 9. Save configuration
    print("\n9. Saving configuration...")
    config_path = "configs/example_config.yaml"
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/streamlit_app.py")
    print("2. Train a model: python scripts/train.py --config configs/training.yaml")
    print("3. Evaluate models: python scripts/evaluate.py --model-path gpt2")
    print("4. Generate samples: python scripts/sample.py --mode interactive")


if __name__ == "__main__":
    main()
