#!/usr/bin/env python3
"""Setup script for the Story Generation System."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("STORY GENERATION SYSTEM - SETUP")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    print("\nCreating directories...")
    directories = [
        "data",
        "outputs", 
        "assets/generated",
        "logs",
        "configs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create sample data
    print("\nCreating sample data...")
    try:
        from src.data.dataset import create_sample_data
        create_sample_data("data/sample_stories.jsonl", num_samples=100)
        print("✓ Sample data created")
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
    
    # Test imports
    print("\nTesting imports...")
    try:
        from src.models.story_generator import StoryGenerator
        from src.data.dataset import StoryDataset
        from src.evaluation.metrics import StoryEvaluationMetrics
        from src.utils.config import StoryGenerationConfig
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        sys.exit(1)
    
    # Run basic test
    print("\nRunning basic functionality test...")
    try:
        from src.models.story_generator import StoryGenerator
        generator = StoryGenerator(model_name="gpt2", seed=42)
        story = generator.generate_story("Once upon a time", max_length=50)
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the example: python example.py")
    print("2. Launch the demo: streamlit run demo/streamlit_app.py")
    print("3. Run tests: pytest tests/ -v")
    print("4. Train a model: python scripts/train.py --config configs/training.yaml")


if __name__ == "__main__":
    main()
