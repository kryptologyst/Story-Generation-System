"""Streamlit demo for story generation system."""

import logging
import random
from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.models.story_generator import StoryGenerator, StoryGeneratorConfig
from src.utils.config import StoryGenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Story Generation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .story-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_story_generator(model_name: str = "gpt2") -> StoryGenerator:
    """Load and cache the story generator.
    
    Args:
        model_name: Name of the model to load.
        
    Returns:
        Loaded StoryGenerator instance.
    """
    try:
        config = StoryGeneratorConfig(model_name=model_name)
        generator = StoryGenerator(
            model_name=config.model_name,
            seed=42,
        )
        return generator
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def generate_story_with_progress(
    generator: StoryGenerator,
    prompt: str,
    **kwargs,
) -> str:
    """Generate story with progress bar.
    
    Args:
        generator: Story generator instance.
        prompt: Input prompt.
        **kwargs: Generation parameters.
        
    Returns:
        Generated story.
    """
    with st.spinner("Generating story..."):
        story = generator.generate_story(prompt, **kwargs)
    return story


def display_story_metrics(story: str) -> None:
    """Display metrics for a generated story.
    
    Args:
        story: Generated story text.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", len(story.split()))
    
    with col2:
        st.metric("Character Count", len(story))
    
    with col3:
        sentences = story.count(".") + story.count("!") + story.count("?")
        st.metric("Sentences", sentences)
    
    with col4:
        avg_words_per_sentence = len(story.split()) / max(sentences, 1)
        st.metric("Avg Words/Sentence", f"{avg_words_per_sentence:.1f}")


def main() -> None:
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Story Generation System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "GPT-2 Small": "gpt2",
        "GPT-2 Medium": "gpt2-medium",
        "GPT-2 Large": "gpt2-large",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
    )
    
    model_name = model_options[selected_model]
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.95,
        step=0.05,
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
    
    # Load model
    generator = load_story_generator(model_name)
    
    if generator is None:
        st.error("Failed to load the story generator. Please check your configuration.")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Generate", "Batch Generate", "Continue Story", "About"])
    
    with tab1:
        st.header("Single Story Generation")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your story prompt:",
            value="Once upon a time in a faraway kingdom, there was a princess who",
            height=100,
            help="Enter the beginning of your story. The AI will continue from here.",
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            generate_button = st.button("Generate Story", type="primary")
        
        with col2:
            num_stories = st.selectbox(
                "Number of stories to generate",
                options=[1, 2, 3, 4, 5],
                index=0,
            )
        
        if generate_button and prompt:
            if num_stories == 1:
                story = generate_story_with_progress(
                    generator,
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                st.markdown('<div class="story-box">', unsafe_allow_html=True)
                st.write("**Generated Story:**")
                st.write(story)
                st.markdown('</div>', unsafe_allow_html=True)
                
                display_story_metrics(story)
                
            else:
                stories = generator.generate_multiple_stories(
                    prompt,
                    num_stories=num_stories,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                for i, story in enumerate(stories):
                    st.markdown(f"### Story {i + 1}")
                    st.markdown('<div class="story-box">', unsafe_allow_html=True)
                    st.write(story)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    display_story_metrics(story)
                    st.divider()
    
    with tab2:
        st.header("Batch Story Generation")
        
        # Predefined prompts
        predefined_prompts = [
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
        
        selected_prompts = st.multiselect(
            "Select prompts to generate stories for:",
            options=predefined_prompts,
            default=predefined_prompts[:3],
        )
        
        batch_num_stories = st.slider(
            "Stories per prompt",
            min_value=1,
            max_value=5,
            value=2,
        )
        
        if st.button("Generate Batch Stories", type="primary"):
            if selected_prompts:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = {}
                
                for i, prompt in enumerate(selected_prompts):
                    status_text.text(f"Generating stories for prompt {i + 1}/{len(selected_prompts)}")
                    
                    stories = generator.generate_multiple_stories(
                        prompt,
                        num_stories=batch_num_stories,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    all_results[prompt] = stories
                    progress_bar.progress((i + 1) / len(selected_prompts))
                
                status_text.text("Generation complete!")
                
                # Display results
                for prompt, stories in all_results.items():
                    st.markdown(f"### Prompt: {prompt}")
                    
                    for j, story in enumerate(stories):
                        st.markdown(f"**Story {j + 1}:**")
                        st.markdown('<div class="story-box">', unsafe_allow_html=True)
                        st.write(story)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.divider()
            else:
                st.warning("Please select at least one prompt.")
    
    with tab3:
        st.header("Continue Existing Story")
        
        existing_story = st.text_area(
            "Enter your existing story:",
            value="Once upon a time, there was a brave knight who lived in a castle. He had a magical sword that could cut through anything.",
            height=150,
            help="Enter the story you want to continue. The AI will generate a continuation.",
        )
        
        continuation_length = st.slider(
            "Continuation Length",
            min_value=50,
            max_value=300,
            value=100,
            step=10,
        )
        
        if st.button("Continue Story", type="primary"):
            if existing_story:
                continued_story = generator.continue_story(
                    existing_story,
                    continuation_length=continuation_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                st.markdown("### Original Story:")
                st.markdown('<div class="story-box">', unsafe_allow_html=True)
                st.write(existing_story)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### Continued Story:")
                st.markdown('<div class="story-box">', unsafe_allow_html=True)
                st.write(continued_story)
                st.markdown('</div>', unsafe_allow_html=True)
                
                display_story_metrics(continued_story)
    
    with tab4:
        st.header("About the Story Generation System")
        
        st.markdown("""
        This is a modern story generation system built using transformer-based language models.
        
        ### Features:
        - **Multiple Models**: Support for GPT-2 variants (small, medium, large)
        - **Flexible Generation**: Adjustable parameters for creativity and coherence
        - **Batch Processing**: Generate multiple stories from multiple prompts
        - **Story Continuation**: Continue existing stories
        - **Interactive Interface**: User-friendly Streamlit interface
        
        ### Generation Parameters:
        - **Temperature**: Controls randomness (higher = more creative)
        - **Top-k**: Limits sampling to top-k most likely tokens
        - **Top-p**: Nucleus sampling (cumulative probability threshold)
        - **Repetition Penalty**: Reduces repetitive text
        - **Max Length**: Maximum length of generated text
        
        ### Technical Details:
        - Built with PyTorch and Transformers library
        - Supports CUDA, MPS (Apple Silicon), and CPU
        - Deterministic seeding for reproducibility
        - Modern Python 3.10+ with type hints
        
        ### Usage Tips:
        1. Start with a clear, descriptive prompt
        2. Adjust temperature for creativity vs. coherence trade-off
        3. Use lower repetition penalty for more diverse text
        4. Experiment with different models for different styles
        """)


if __name__ == "__main__":
    main()
