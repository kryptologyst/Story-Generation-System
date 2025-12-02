"""Unit tests for story generation system."""

import pytest
import torch
from transformers import GPT2Tokenizer

from src.data.dataset import StoryDataset, create_sample_data
from src.evaluation.metrics import StoryEvaluationMetrics
from src.models.story_generator import StoryGenerator, StoryGeneratorConfig
from src.utils.config import StoryGenerationConfig


class TestStoryGenerator:
    """Test cases for StoryGenerator class."""
    
    def test_story_generator_initialization(self):
        """Test StoryGenerator initialization."""
        config = StoryGeneratorConfig(model_name="gpt2")
        generator = StoryGenerator(
            model_name=config.model_name,
            seed=42,
        )
        
        assert generator.model_name == "gpt2"
        assert generator.seed == 42
        assert generator.tokenizer is not None
        assert generator.model is not None
    
    def test_generate_story(self):
        """Test story generation."""
        generator = StoryGenerator(model_name="gpt2", seed=42)
        
        prompt = "Once upon a time"
        story = generator.generate_story(
            prompt,
            max_length=50,
            temperature=1.0,
            do_sample=True,
        )
        
        assert isinstance(story, str)
        assert len(story) > len(prompt)
        assert story.startswith(prompt)
    
    def test_generate_multiple_stories(self):
        """Test multiple story generation."""
        generator = StoryGenerator(model_name="gpt2", seed=42)
        
        prompt = "In a small town"
        stories = generator.generate_multiple_stories(
            prompt,
            num_stories=3,
            max_length=50,
        )
        
        assert isinstance(stories, list)
        assert len(stories) == 3
        
        for story in stories:
            assert isinstance(story, str)
            assert len(story) > len(prompt)
    
    def test_continue_story(self):
        """Test story continuation."""
        generator = StoryGenerator(model_name="gpt2", seed=42)
        
        existing_story = "Once upon a time, there was a brave knight."
        continued_story = generator.continue_story(
            existing_story,
            continuation_length=50,
        )
        
        assert isinstance(continued_story, str)
        assert len(continued_story) > len(existing_story)
        assert continued_story.startswith(existing_story)


class TestStoryDataset:
    """Test cases for StoryDataset class."""
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation."""
        dataset = StoryDataset()
        
        assert dataset.dataset is not None
        assert "train" in dataset.dataset
        assert "validation" in dataset.dataset
        assert "test" in dataset.dataset
        
        # Check that splits sum to approximately 1.0
        total_samples = (
            len(dataset.dataset["train"]) +
            len(dataset.dataset["validation"]) +
            len(dataset.dataset["test"])
        )
        assert total_samples > 0
    
    def test_get_prompts_and_stories(self):
        """Test getting prompts and stories from dataset."""
        dataset = StoryDataset()
        
        prompts = dataset.get_prompts("test")
        stories = dataset.get_stories("test")
        
        assert isinstance(prompts, list)
        assert isinstance(stories, list)
        assert len(prompts) == len(stories)
        assert len(prompts) > 0
    
    def test_preprocess_function(self):
        """Test dataset preprocessing function."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = StoryDataset(tokenizer=tokenizer)
        
        # Test preprocessing
        examples = {
            "story": ["This is a test story.", "Another test story."]
        }
        
        processed = dataset.preprocess_function(examples)
        
        assert "input_ids" in processed
        assert "attention_mask" in processed
        assert "labels" in processed
        
        assert len(processed["input_ids"]) == 2
        assert len(processed["attention_mask"]) == 2
        assert len(processed["labels"]) == 2


class TestStoryEvaluationMetrics:
    """Test cases for StoryEvaluationMetrics class."""
    
    def test_evaluation_metrics_initialization(self):
        """Test evaluation metrics initialization."""
        evaluator = StoryEvaluationMetrics()
        
        assert evaluator.device is not None
        assert evaluator.rouge_scorer is not None
        assert evaluator.bleu_scorer is not None
    
    def test_length_metrics(self):
        """Test length-based metrics computation."""
        evaluator = StoryEvaluationMetrics()
        
        stories = [
            "This is a short story.",
            "This is a much longer story with more words and sentences.",
            "Medium length story here.",
        ]
        
        metrics = evaluator.compute_length_metrics(stories)
        
        assert "avg_word_count" in metrics
        assert "std_word_count" in metrics
        assert "avg_char_count" in metrics
        assert "min_word_count" in metrics
        assert "max_word_count" in metrics
        
        assert metrics["avg_word_count"] > 0
        assert metrics["min_word_count"] <= metrics["max_word_count"]
    
    def test_diversity_metrics(self):
        """Test diversity metrics computation."""
        evaluator = StoryEvaluationMetrics()
        
        stories = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "The bird flew in the sky.",
        ]
        
        metrics = evaluator.compute_diversity_metrics(stories)
        
        assert "distinct_1" in metrics
        assert "distinct_2" in metrics
        assert "distinct_3" in metrics
        assert "distinct_4" in metrics
        
        for metric in metrics.values():
            assert 0 <= metric <= 1
    
    def test_coherence_score(self):
        """Test coherence score computation."""
        evaluator = StoryEvaluationMetrics()
        
        stories = [
            "First sentence. Then second sentence. Finally third sentence.",
            "Single sentence story.",
            "First. However second. Therefore third.",
        ]
        
        coherence_scores = []
        for story in stories:
            score = evaluator.compute_coherence_score([story])
            coherence_scores.append(score)
        
        assert len(coherence_scores) == 3
        assert all(0 <= score <= 1 for score in coherence_scores)
    
    def test_rouge_scores(self):
        """Test ROUGE scores computation."""
        evaluator = StoryEvaluationMetrics()
        
        generated = ["The cat sat on the mat."]
        reference = ["The cat was sitting on the mat."]
        
        scores = evaluator.compute_rouge_scores(generated, reference)
        
        assert "rouge_rouge1" in scores
        assert "rouge_rouge2" in scores
        assert "rouge_rougeL" in scores
        
        for score in scores.values():
            assert 0 <= score <= 1
    
    def test_bleu_score(self):
        """Test BLEU score computation."""
        evaluator = StoryEvaluationMetrics()
        
        generated = ["The cat sat on the mat."]
        reference = ["The cat was sitting on the mat."]
        
        score = evaluator.compute_bleu_score(generated, reference)
        
        assert 0 <= score <= 1
    
    def test_evaluate_stories(self):
        """Test comprehensive story evaluation."""
        evaluator = StoryEvaluationMetrics()
        
        generated = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
        ]
        reference = [
            "The cat was sitting on the mat.",
            "The dog was running in the park.",
        ]
        
        results = evaluator.evaluate_stories(generated, reference)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "avg_word_count", "std_word_count", "avg_char_count",
            "distinct_1", "distinct_2", "distinct_3", "distinct_4",
            "coherence", "rouge_rouge1", "rouge_rouge2", "rouge_rougeL",
            "bleu"
        ]
        
        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))


class TestStoryGenerationConfig:
    """Test cases for configuration classes."""
    
    def test_story_generation_config_initialization(self):
        """Test StoryGenerationConfig initialization."""
        config = StoryGenerationConfig()
        
        assert config.model.name == "gpt2"
        assert config.data.batch_size == 8
        assert config.training.learning_rate == 5e-5
        assert config.generation.temperature == 1.0
        assert config.system.seed == 42
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = StoryGenerationConfig()
        config._validate_config()  # Should not raise
        
        # Test invalid splits
        config.data.train_split = 0.5
        config.data.val_split = 0.3
        config.data.test_split = 0.3  # Total > 1.0
        
        with pytest.raises(ValueError):
            config._validate_config()
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = StoryGenerationConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "data" in config_dict
        assert "training" in config_dict
        assert "evaluation" in config_dict
        assert "generation" in config_dict
        assert "system" in config_dict
    
    def test_config_save_load(self, tmp_path):
        """Test configuration save and load."""
        config = StoryGenerationConfig()
        config_path = tmp_path / "test_config.yaml"
        
        # Save configuration
        config.save(config_path)
        assert config_path.exists()
        
        # Load configuration
        loaded_config = StoryGenerationConfig.load(config_path)
        
        assert loaded_config.model.name == config.model.name
        assert loaded_config.data.batch_size == config.data.batch_size
        assert loaded_config.training.learning_rate == config.training.learning_rate


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_sample_data(self, tmp_path):
        """Test sample data creation."""
        output_path = tmp_path / "sample_data.jsonl"
        
        create_sample_data(str(output_path), num_samples=10)
        
        assert output_path.exists()
        
        # Check file content
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == 10
        
        # Check JSON format
        import json
        for line in lines:
            data = json.loads(line.strip())
            assert "prompt" in data
            assert "story" in data
            assert "genre" in data
            assert "length" in data


if __name__ == "__main__":
    pytest.main([__file__])
