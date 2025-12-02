"""Data loading and preprocessing utilities for story generation."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class StoryDataset:
    """A dataset class for story generation tasks."""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        """Initialize the story dataset.
        
        Args:
            data_path: Path to the dataset file or directory.
            tokenizer: Tokenizer for text preprocessing.
            max_length: Maximum sequence length.
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.dataset: Optional[DatasetDict] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess the dataset."""
        if self.data_path and os.path.exists(self.data_path):
            self.dataset = self._load_from_file()
        else:
            logger.warning("No data path provided or file doesn't exist. Creating sample dataset.")
            self.dataset = self._create_sample_dataset()
    
    def _load_from_file(self) -> DatasetDict:
        """Load dataset from file."""
        data_path = Path(self.data_path)
        
        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif data_path.suffix == ".jsonl":
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
            data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Convert to HuggingFace dataset format
        dataset = Dataset.from_list(data)
        
        # Split the dataset
        train_size = int(len(dataset) * self.train_split)
        val_size = int(len(dataset) * self.val_split)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })
    
    def _create_sample_dataset(self) -> DatasetDict:
        """Create a sample dataset for demonstration."""
        sample_stories = [
            {
                "prompt": "Once upon a time in a faraway kingdom, there was a princess who",
                "story": "Once upon a time in a faraway kingdom, there was a princess who loved to read books. She spent her days in the royal library, discovering magical tales and ancient wisdom. One day, she found a mysterious book that seemed to glow with its own light.",
                "genre": "fantasy",
                "length": "short",
            },
            {
                "prompt": "In a small town by the sea, an old lighthouse keeper",
                "story": "In a small town by the sea, an old lighthouse keeper discovered that the light he tended every night was actually powered by the dreams of children. The brighter their dreams, the stronger the light shone, guiding ships safely to shore.",
                "genre": "magical_realism",
                "length": "short",
            },
            {
                "prompt": "The last robot on Earth",
                "story": "The last robot on Earth wandered through empty cities, its circuits humming with memories of a world that once bustled with life. It had been programmed to preserve humanity's stories, and now it told them to the wind, hoping someone would listen.",
                "genre": "science_fiction",
                "length": "short",
            },
            {
                "prompt": "A detective walked into a coffee shop",
                "story": "A detective walked into a coffee shop and immediately noticed something was wrong. The barista's hands were shaking, the customers were avoiding eye contact, and there was an unusual silence that hung in the air like a thick fog.",
                "genre": "mystery",
                "length": "short",
            },
            {
                "prompt": "The time traveler arrived in the year 1920",
                "story": "The time traveler arrived in the year 1920, but something was different. The people spoke in a language that sounded familiar yet foreign, and the buildings seemed to shimmer with an otherworldly quality that made her question everything she knew about history.",
                "genre": "science_fiction",
                "length": "short",
            },
        ]
        
        # Create multiple variations for a larger dataset
        expanded_stories = []
        for story in sample_stories:
            for i in range(20):  # Create 20 variations of each story
                expanded_stories.append({
                    **story,
                    "id": f"{story['genre']}_{i}",
                })
        
        dataset = Dataset.from_list(expanded_stories)
        
        # Split the dataset
        train_size = int(len(dataset) * self.train_split)
        val_size = int(len(dataset) * self.val_split)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess examples for training.
        
        Args:
            examples: Batch of examples.
            
        Returns:
            Preprocessed examples.
        """
        if self.tokenizer is None:
            return examples
        
        # Tokenize the stories
        stories = examples["story"]
        inputs = self.tokenizer(
            stories,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"],
        }
    
    def get_dataset(self) -> DatasetDict:
        """Get the loaded dataset.
        
        Returns:
            The dataset dictionary with train/val/test splits.
        """
        return self.dataset
    
    def get_prompts(self, split: str = "test") -> List[str]:
        """Get prompts from a specific split.
        
        Args:
            split: Dataset split ("train", "validation", "test").
            
        Returns:
            List of prompts.
        """
        if self.dataset is None:
            return []
        
        return self.dataset[split]["prompt"]
    
    def get_stories(self, split: str = "test") -> List[str]:
        """Get stories from a specific split.
        
        Args:
            split: Dataset split ("train", "validation", "test").
            
        Returns:
            List of stories.
        """
        if self.dataset is None:
            return []
        
        return self.dataset[split]["story"]


def load_story_dataset(
    data_path: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs,
) -> StoryDataset:
    """Load a story dataset.
    
    Args:
        data_path: Path to the dataset file.
        tokenizer: Tokenizer for preprocessing.
        **kwargs: Additional arguments for StoryDataset.
        
    Returns:
        Loaded StoryDataset instance.
    """
    return StoryDataset(data_path=data_path, tokenizer=tokenizer, **kwargs)


def create_sample_data(output_path: str, num_samples: int = 100) -> None:
    """Create sample story data for testing.
    
    Args:
        output_path: Path to save the sample data.
        num_samples: Number of sample stories to create.
    """
    sample_stories = []
    
    prompts = [
        "Once upon a time in a faraway kingdom, there was a princess who",
        "In a small town by the sea, an old lighthouse keeper",
        "The last robot on Earth",
        "A detective walked into a coffee shop",
        "The time traveler arrived in the year 1920",
        "In the depths of space, a lone astronaut",
        "The ancient library held secrets that",
        "A young wizard discovered that magic",
        "The abandoned mansion on the hill",
        "In a world where memories could be traded",
    ]
    
    genres = ["fantasy", "science_fiction", "mystery", "romance", "horror"]
    lengths = ["short", "medium", "long"]
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        genre = genres[i % len(genres)]
        length = lengths[i % len(lengths)]
        
        # Generate a simple story continuation
        story = f"{prompt} embarked on an incredible journey that would change everything. The adventure began with a mysterious discovery that led to unexpected revelations about the nature of reality itself."
        
        sample_stories.append({
            "id": f"story_{i:04d}",
            "prompt": prompt,
            "story": story,
            "genre": genre,
            "length": length,
            "word_count": len(story.split()),
        })
    
    # Save as JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for story in sample_stories:
            f.write(json.dumps(story) + "\n")
    
    logger.info(f"Created sample dataset with {num_samples} stories at {output_path}")
