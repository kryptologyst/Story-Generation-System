"""Evaluation metrics and utilities for story generation."""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from torchmetrics import Metric, MetricCollection
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class StoryEvaluationMetrics:
    """Comprehensive evaluation metrics for story generation."""
    
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize evaluation metrics.
        
        Args:
            tokenizer: Tokenizer for text preprocessing.
            device: Device for computations.
        """
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bleu_scorer = BLEU()
    
    def compute_perplexity(
        self,
        model,
        texts: List[str],
        batch_size: int = 8,
    ) -> float:
        """Compute perplexity of generated texts.
        
        Args:
            model: Language model for evaluation.
            texts: List of generated texts.
            batch_size: Batch size for processing.
            
        Returns:
            Average perplexity score.
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer provided, skipping perplexity computation")
            return 0.0
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                
                # Compute loss
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def compute_rouge_scores(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> Dict[str, float]:
        """Compute ROUGE scores.
        
        Args:
            generated_texts: List of generated stories.
            reference_texts: List of reference stories.
            
        Returns:
            Dictionary of ROUGE scores.
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            scores = self.rouge_scorer.score(ref_text, gen_text)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average scores
        avg_scores = {}
        for metric in rouge_scores:
            avg_scores[f"rouge_{metric}"] = np.mean(rouge_scores[metric])
        
        return avg_scores
    
    def compute_bleu_score(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> float:
        """Compute BLEU score.
        
        Args:
            generated_texts: List of generated stories.
            reference_texts: List of reference stories.
            
        Returns:
            BLEU score.
        """
        # Prepare texts for BLEU computation
        references = [[ref.split()] for ref in reference_texts]
        hypotheses = [gen.split() for gen in generated_texts]
        
        bleu_score = self.bleu_scorer.corpus_score(hypotheses, references)
        return bleu_score.score / 100.0  # Normalize to 0-1 range
    
    def compute_bert_score(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Compute BERTScore.
        
        Args:
            generated_texts: List of generated stories.
            reference_texts: List of reference stories.
            batch_size: Batch size for BERTScore computation.
            
        Returns:
            Dictionary of BERTScore metrics.
        """
        try:
            P, R, F1 = bert_score(
                generated_texts,
                reference_texts,
                lang="en",
                batch_size=batch_size,
                device=str(self.device),
            )
            
            return {
                "bert_precision": P.mean().item(),
                "bert_recall": R.mean().item(),
                "bert_f1": F1.mean().item(),
            }
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    
    def compute_diversity_metrics(
        self,
        generated_texts: List[str],
        n_grams: Tuple[int, ...] = (1, 2, 3, 4),
    ) -> Dict[str, float]:
        """Compute diversity metrics (Distinct-n).
        
        Args:
            generated_texts: List of generated stories.
            n_grams: N-gram sizes to compute.
            
        Returns:
            Dictionary of diversity metrics.
        """
        diversity_scores = {}
        
        for n in n_grams:
            all_ngrams = []
            total_tokens = 0
            
            for text in generated_texts:
                tokens = text.split()
                total_tokens += len(tokens)
                
                # Generate n-grams
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i:i + n])
                    all_ngrams.append(ngram)
            
            # Compute distinct-n
            unique_ngrams = len(set(all_ngrams))
            distinct_n = unique_ngrams / total_tokens if total_tokens > 0 else 0.0
            
            diversity_scores[f"distinct_{n}"] = distinct_n
        
        return diversity_scores
    
    def compute_coherence_score(
        self,
        generated_texts: List[str],
    ) -> float:
        """Compute a simple coherence score based on sentence transitions.
        
        Args:
            generated_texts: List of generated stories.
            
        Returns:
            Average coherence score.
        """
        coherence_scores = []
        
        for text in generated_texts:
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                coherence_scores.append(0.0)
                continue
            
            # Simple coherence: check for common transition words
            transition_words = [
                "then", "next", "after", "before", "while", "during",
                "meanwhile", "however", "therefore", "because", "although",
                "furthermore", "moreover", "additionally", "consequently",
            ]
            
            transitions = 0
            for sentence in sentences[1:]:  # Skip first sentence
                sentence_lower = sentence.lower()
                for transition in transition_words:
                    if transition in sentence_lower:
                        transitions += 1
                        break
            
            coherence_score = transitions / (len(sentences) - 1)
            coherence_scores.append(coherence_score)
        
        return np.mean(coherence_scores)
    
    def compute_length_metrics(
        self,
        generated_texts: List[str],
    ) -> Dict[str, float]:
        """Compute length-based metrics.
        
        Args:
            generated_texts: List of generated stories.
            
        Returns:
            Dictionary of length metrics.
        """
        word_counts = [len(text.split()) for text in generated_texts]
        char_counts = [len(text) for text in generated_texts]
        
        return {
            "avg_word_count": np.mean(word_counts),
            "std_word_count": np.std(word_counts),
            "avg_char_count": np.mean(char_counts),
            "std_char_count": np.std(char_counts),
            "min_word_count": np.min(word_counts),
            "max_word_count": np.max(word_counts),
        }
    
    def evaluate_stories(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        model=None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of generated stories.
        
        Args:
            generated_texts: List of generated stories.
            reference_texts: List of reference stories (optional).
            model: Language model for perplexity computation (optional).
            
        Returns:
            Dictionary of all evaluation metrics.
        """
        results = {}
        
        # Length metrics
        results.update(self.compute_length_metrics(generated_texts))
        
        # Diversity metrics
        results.update(self.compute_diversity_metrics(generated_texts))
        
        # Coherence score
        results["coherence"] = self.compute_coherence_score(generated_texts)
        
        # Reference-based metrics (if references provided)
        if reference_texts is not None:
            results.update(self.compute_rouge_scores(generated_texts, reference_texts))
            results["bleu"] = self.compute_bleu_score(generated_texts, reference_texts)
            results.update(self.compute_bert_score(generated_texts, reference_texts))
        
        # Perplexity (if model provided)
        if model is not None:
            results["perplexity"] = self.compute_perplexity(model, generated_texts)
        
        return results


class StoryQualityMetrics(Metric):
    """Custom PyTorch Lightning metric for story quality evaluation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, generated_texts: List[str], reference_texts: List[str]) -> None:
        """Update metric with new batch of texts."""
        evaluator = StoryEvaluationMetrics()
        scores = evaluator.evaluate_stories(generated_texts, reference_texts)
        
        # Use BLEU score as primary quality metric
        quality_score = scores.get("bleu", 0.0)
        
        self.total_score += quality_score * len(generated_texts)
        self.total_samples += len(generated_texts)
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return self.total_score / self.total_samples


def create_evaluation_metrics() -> MetricCollection:
    """Create a collection of evaluation metrics.
    
    Returns:
        Collection of PyTorch Lightning metrics.
    """
    return MetricCollection({
        "story_quality": StoryQualityMetrics(),
    })


def evaluate_model_performance(
    model,
    tokenizer: PreTrainedTokenizer,
    test_dataset,
    num_samples: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance on test dataset.
    
    Args:
        model: Trained language model.
        tokenizer: Tokenizer for the model.
        test_dataset: Test dataset.
        num_samples: Number of samples to evaluate.
        device: Device for computation.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    evaluator = StoryEvaluationMetrics(tokenizer=tokenizer, device=device)
    
    # Generate stories
    generated_texts = []
    reference_texts = []
    
    prompts = test_dataset["prompt"][:num_samples]
    references = test_dataset["story"][:num_samples]
    
    with torch.no_grad():
        for prompt in prompts:
            # Generate story
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs,
                max_length=200,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_story)
    
    reference_texts = references[:num_samples]
    
    # Evaluate
    results = evaluator.evaluate_stories(
        generated_texts,
        reference_texts,
        model=model,
    )
    
    return results
