import torch
import numpy as np
import hashlib
import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.stats import binom
import json

@dataclass
class WatermarkConfig:
    """Configuration for SynthID-Text watermarking"""
    num_layers: int = 30  # m in the paper
    context_window: int = 4  # H in the paper
    g_value_dist: str = "bernoulli"  # "bernoulli" or "uniform"
    n_sec: int = 64  # security parameter for random seeds
    repeated_context_masking: bool = True
    k_sequences: int = 1  # K for repeated context masking

class SynthIDText:
    """
    Implementation of SynthID-Text watermarking as described in the Nature paper.
    
    This implements the Tournament sampling algorithm for watermarking LLM outputs
    and provides detection capabilities.
    """
    
    def __init__(self, watermark_key: str, config: WatermarkConfig = None):
        self.watermark_key = watermark_key.encode('utf-8')
        self.config = config or WatermarkConfig()
        self.context_history = []  # For repeated context masking
        
    def _hash_function(self, tokens: List[int], layer: int = 0) -> int:
        """
        Pseudorandom hash function for generating random seeds and g-values.
        
        Args:
            tokens: List of token ids
            layer: Layer number for tournament (0 for seed generation)
            
        Returns:
            Hash value as integer
        """
        # Create hash input from tokens, layer, and watermark key
        hash_input = json.dumps(tokens).encode('utf-8') + str(layer).encode('utf-8') + self.watermark_key
        hash_obj = hashlib.sha256(hash_input)
        # Convert to integer and mask to n_sec bits
        return int(hash_obj.hexdigest(), 16) % (2 ** self.config.n_sec)
    
    def _get_random_seed(self, context_tokens: List[int]) -> int:
        """
        Generate random seed using sliding window approach.
        
        Args:
            context_tokens: Recent context tokens (length H)
            
        Returns:
            Random seed as integer
        """
        # Use last H tokens as context window
        window = context_tokens[-self.config.context_window:] if len(context_tokens) >= self.config.context_window else context_tokens
        return self._hash_function(window, layer=0)
    
    def _get_g_value(self, token_id: int, random_seed: int, layer: int) -> float:
        """
        Compute g-value for a token at a specific tournament layer.
        
        Args:
            token_id: Token ID
            random_seed: Random seed for this timestep
            layer: Tournament layer number (1-indexed)
            
        Returns:
            G-value as float
        """
        # Create hash for this token, layer, and seed
        # Use a more robust hash combination
        hash_input = f"{token_id}_{layer}_{random_seed}".encode('utf-8')
        hash_obj = hashlib.sha256(hash_input)
        hash_bytes = hash_obj.digest()
        
        # Use first 8 bytes for better randomness
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        
        # Convert to [0, 1] uniform value with better precision
        uniform_val = hash_int / (2 ** 64)
        
        if self.config.g_value_dist == "bernoulli":
            # Bernoulli(0.5) distribution
            return 1.0 if uniform_val >= 0.5 else 0.0
        elif self.config.g_value_dist == "uniform":
            # Uniform[0, 1] distribution
            return uniform_val
        else:
            raise ValueError(f"Unknown g-value distribution: {self.config.g_value_dist}")
    
    def _single_layer_tournament(self, candidates: List[int], random_seed: int, layer: int) -> int:
        """
        Run single layer of tournament sampling.
        
        Args:
            candidates: List of candidate token IDs
            random_seed: Random seed for this timestep
            layer: Tournament layer number (1-indexed)
            
        Returns:
            Winning token ID
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute g-values for all candidates
        g_values = [self._get_g_value(token_id, random_seed, layer) for token_id in candidates]
        max_g_value = max(g_values)
        
        # Find all tokens with maximum g-value
        winners = [candidates[i] for i, g_val in enumerate(g_values) if g_val == max_g_value]
        
        # Random tie-breaking
        return random.choice(winners)
    
    def tournament_sampling(self, logits: torch.Tensor, context_tokens: List[int]) -> int:
        """
        Apply Tournament sampling to select next token.
        
        Args:
            logits: Model logits for next token prediction
            context_tokens: Previous tokens in the sequence
            
        Returns:
            Selected token ID
        """
        # Check for repeated context masking
        if self.config.repeated_context_masking:
            recent_context = tuple(context_tokens[-self.config.context_window:])
            if recent_context in self.context_history[-self.config.k_sequences:]:
                # Don't apply watermark, sample normally
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()
            else:
                # Add to context history
                if len(self.context_history) >= self.config.k_sequences:
                    self.context_history.pop(0)
                self.context_history.append(recent_context)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        vocab_size = probs.size(-1)
        
        # Generate random seed
        random_seed = self._get_random_seed(context_tokens)
        
        # Sample 2^m candidates from the distribution
        num_candidates = 2 ** self.config.num_layers
        candidates = torch.multinomial(probs, num_candidates, replacement=True).tolist()
        
        # Run tournament with m layers
        current_candidates = candidates
        for layer in range(1, self.config.num_layers + 1):
            if len(current_candidates) == 1:
                break
                
            # Pair up candidates and run tournament matches
            next_candidates = []
            for i in range(0, len(current_candidates), 2):
                if i + 1 < len(current_candidates):
                    # Tournament match between two candidates
                    match_candidates = [current_candidates[i], current_candidates[i + 1]]
                else:
                    # Odd number of candidates, last one advances automatically
                    match_candidates = [current_candidates[i]]
                
                winner = self._single_layer_tournament(match_candidates, random_seed, layer)
                next_candidates.append(winner)
            
            current_candidates = next_candidates
        
        return current_candidates[0]
    
    def compute_mean_score(self, tokens: List[int], context_prefix: List[int] = None) -> float:
        """
        Compute mean g-value score for watermark detection.
        
        Args:
            tokens: List of token IDs to score
            context_prefix: Initial context tokens (if any)
            
        Returns:
            Mean score across all tokens and layers
        """
        if len(tokens) == 0:
            return 0.0
            
        total_score = 0.0
        total_count = 0
        
        # Build full context including prefix
        full_context = (context_prefix or []) + tokens
        
        for t, token_id in enumerate(tokens):
            # Get context up to this position
            context_tokens = full_context[:len(context_prefix or []) + t]
            
            # Check if watermark was applied at this position
            if self.config.repeated_context_masking:
                recent_context = tuple(context_tokens[-self.config.context_window:])
                # For detection, we need to simulate the same history tracking
                # This is a simplified version - in practice you'd need the exact history
            
            # Generate random seed for this position
            random_seed = self._get_random_seed(context_tokens)
            
            # Compute g-values for all layers
            for layer in range(1, self.config.num_layers + 1):
                g_value = self._get_g_value(token_id, random_seed, layer)
                total_score += g_value
                total_count += 1
        
        return total_score / total_count if total_count > 0 else 0.0
    
    def detect_watermark(self, tokens: List[int], threshold: float = None, 
                        context_prefix: List[int] = None, significance_level: float = 0.01) -> Tuple[bool, float, float]:
        """
        Detect if text is watermarked using statistical testing.
        
        Args:
            tokens: List of token IDs to analyze
            threshold: Detection threshold (if None, use p-value based detection)
            context_prefix: Initial context tokens (if any)
            significance_level: Significance level for p-value based detection
            
        Returns:
            Tuple of (is_watermarked, score, p_value)
        """
        if len(tokens) == 0:
            return False, 0.0, 1.0
            
        score = self.compute_mean_score(tokens, context_prefix)
        
        # Compute p-value assuming null hypothesis (no watermark)
        if self.config.g_value_dist == "bernoulli":
            # Expected score is 0.5 under null hypothesis
            expected_score = 0.5
            n_total = len(tokens) * self.config.num_layers
            
            # Use binomial test - more accurate calculation
            if n_total > 0:
                observed_successes = int(score * n_total)
                # Two-tailed test: we want to detect if score is significantly > 0.5
                p_value = 1 - binom.cdf(observed_successes - 1, n_total, expected_score)
            else:
                p_value = 1.0
        else:
            # For uniform distribution, use normal approximation
            expected_score = 0.5
            variance = 1.0 / 12  # variance of uniform[0,1]
            n_total = len(tokens) * self.config.num_layers
            
            if n_total > 0:
                std_error = np.sqrt(variance / n_total)
                z_score = (score - expected_score) / std_error
                # One-tailed test for detecting watermark
                from scipy.stats import norm
                p_value = 1 - norm.cdf(z_score)
            else:
                p_value = 1.0
        
        # Determine if watermarked
        if threshold is not None:
            # Use threshold-based detection
            is_watermarked = score > threshold
        else:
            # Use p-value based detection (more statistically sound)
            is_watermarked = p_value < significance_level and score > 0.5
        
        return is_watermarked, score, p_value
    
    def reset_context_history(self):
        """Reset the context history for repeated context masking."""
        self.context_history = []

class WatermarkDetector:
    """
    Standalone detector for SynthID-Text watermarks.
    """
    
    def __init__(self, watermark_key: str, config: WatermarkConfig = None):
        self.synthid = SynthIDText(watermark_key, config)
    
    def detect(self, tokens: List[int], threshold: float = None, significance_level: float = 0.01) -> Dict:
        """
        Detect watermark in a sequence of tokens.
        
        Args:
            tokens: Token IDs to analyze
            threshold: Detection threshold (if None, use p-value based detection)
            significance_level: Significance level for statistical test
            
        Returns:
            Dictionary with detection results
        """
        is_watermarked, score, p_value = self.synthid.detect_watermark(
            tokens, threshold, significance_level=significance_level
        )
        
        return {
            'is_watermarked': is_watermarked,
            'score': score,
            'p_value': p_value,
            'threshold': threshold,
            'significance_level': significance_level,
            'num_tokens': len(tokens),
            'num_layers': self.synthid.config.num_layers,
            'detection_method': 'p_value' if threshold is None else 'threshold'
        }

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    watermark_key = "my_secret_key_123"
    config = WatermarkConfig(num_layers=10)  # Smaller for testing
    
    # Create watermarker
    watermarker = SynthIDText(watermark_key, config)
    
    # Simulate some token logits (normally from LLM)
    vocab_size = 1000
    sequence_length = 100  # Increased for better statistics
    
    print("Testing SynthID-Text Watermarking (Improved)")
    print("-" * 50)
    
    # Generate watermarked sequence
    watermarked_tokens = []
    context = [1, 2, 3]  # Initial context
    
    # Set seed for reproducible results
    random.seed(42)
    torch.manual_seed(42)
    
    for i in range(sequence_length):
        # Simulate logits (normally from your LLM)
        logits = torch.randn(vocab_size) * 2  # Random logits for demo
        
        # Apply watermarking
        selected_token = watermarker.tournament_sampling(logits, context + watermarked_tokens)
        watermarked_tokens.append(selected_token)
    
    print(f"Generated {len(watermarked_tokens)} watermarked tokens")
    
    # Reset for fresh detection
    watermarker.reset_context_history()
    
    # Test detection with p-value based method (more reliable)
    detector = WatermarkDetector(watermark_key, config)
    
    # Detect watermarked text using p-value method
    watermark_result = detector.detect(watermarked_tokens)
    print(f"Watermarked text detection (p-value): {watermark_result}")
    
    # Test on unwatermarked text (random tokens)
    random.seed(123)  # Different seed for unwatermarked
    unwatermarked_tokens = [random.randint(0, vocab_size-1) for _ in range(sequence_length)]
    unwatermarked_result = detector.detect(unwatermarked_tokens)
    print(f"Unwatermarked text detection (p-value): {unwatermarked_result}")
    
    # Test with threshold-based detection
    print("\nThreshold-based detection:")
    for threshold in [0.51, 0.52, 0.53, 0.55]:
        wm_result = detector.detect(watermarked_tokens, threshold=threshold)
        unwm_result = detector.detect(unwatermarked_tokens, threshold=threshold)
        print(f"Threshold {threshold}: WM={wm_result['is_watermarked']} (score={wm_result['score']:.3f}), "
              f"UnWM={unwm_result['is_watermarked']} (score={unwm_result['score']:.3f})")
    
    # Test with different sequence lengths
    print("\nDetection performance vs sequence length:")
    for length in [20, 50, 100, 200]:
        if length <= len(watermarked_tokens):
            wm_subset = watermarked_tokens[:length]
            unwm_subset = unwatermarked_tokens[:length]
            
            wm_result = detector.detect(wm_subset)
            unwm_result = detector.detect(unwm_subset)
            
            print(f"Length {length:3d}: WM_score={wm_result['score']:.3f} (p={wm_result['p_value']:.4f}), "
                  f"UnWM_score={unwm_result['score']:.3f} (p={unwm_result['p_value']:.4f})")
    
    print(f"\nExpected score for unwatermarked text: ~0.500")
    print(f"Expected p-value for unwatermarked text: >0.01 (not significant)")
    print(f"Expected score for watermarked text: >0.500")
    print(f"Expected p-value for watermarked text: <0.01 (significant)")