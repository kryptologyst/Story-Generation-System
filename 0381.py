#!/usr/bin/env python3
"""Legacy file - Story Generation System has been modernized and refactored.

This file is kept for reference. The modernized system is now available in the
src/ directory with proper structure, type hints, and comprehensive features.

To use the modern system:

1. Run the example: python example.py
2. Launch the demo: streamlit run demo/streamlit_app.py
3. Train a model: python scripts/train.py
4. Evaluate models: python scripts/evaluate.py

The original simple implementation is preserved below for reference.
"""

# Original implementation (preserved for reference)
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

def generate_story_legacy(prompt, max_length=200):
    """Legacy story generation function."""
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.eval()
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            temperature=1.0
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    print("This is the legacy implementation.")
    print("Please use the modernized system:")
    print("  python example.py")
    print("  streamlit run demo/streamlit_app.py")
    
    # Example usage of legacy function
    prompt = "Once upon a time in a faraway kingdom, there was a princess who"
    generated_story = generate_story_legacy(prompt)
    
    print("\nLegacy Generated Story:")
    print(generated_story)