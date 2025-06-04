"""
Utility functions for MTAGEC project.
"""
import os
import yaml
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from transformers import PreTrainedTokenizer
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config YAML file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def compute_metrics(
    predictions: List[str],
    references: List[str],
    error_type_preds: Optional[List[str]] = None,
    error_type_refs: Optional[List[str]] = None,
    evidence_preds: Optional[List[List[str]]] = None,
    evidence_refs: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for MTAGEC evaluation.
    
    Args:
        predictions: List of predicted corrected sentences
        references: List of reference corrected sentences
        error_type_preds: List of predicted error types
        error_type_refs: List of reference error types
        evidence_preds: List of predicted evidence tokens
        evidence_refs: List of reference evidence tokens
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Correction metrics (M2 scorer would be used in evaluate.py)
    # Here we compute simple exact match for quick evaluation
    exact_matches = sum(p == r for p, r in zip(predictions, references))
    metrics['exact_match'] = exact_matches / len(predictions) if predictions else 0
    
    # Error type classification metrics
    if error_type_preds and error_type_refs:
        precision, recall, f1, _ = precision_recall_fscore_support(
            error_type_refs, error_type_preds, average='weighted'
        )
        metrics['error_type_precision'] = precision
        metrics['error_type_recall'] = recall
        metrics['error_type_f1'] = f1
        
        # Accuracy
        accuracy = sum(p == r for p, r in zip(error_type_preds, error_type_refs)) / len(error_type_preds)
        metrics['error_type_accuracy'] = accuracy
    
    # Evidence extraction metrics
    if evidence_preds and evidence_refs:
        # Compute token-level precision, recall, F1
        total_precision, total_recall, total_f1 = 0, 0, 0
        
        for pred_tokens, ref_tokens in zip(evidence_preds, evidence_refs):
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            
            if not pred_set and not ref_set:
                # Both empty, perfect match
                precision, recall, f1 = 1.0, 1.0, 1.0
            elif not pred_set:
                # No predictions but should have some
                precision, recall, f1 = 0.0, 0.0, 0.0
            elif not ref_set:
                # Predictions but should be none
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                # Normal case
                true_positives = len(pred_set.intersection(ref_set))
                precision = true_positives / len(pred_set) if pred_set else 0
                recall = true_positives / len(ref_set) if ref_set else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        n_samples = len(evidence_preds)
        metrics['evidence_precision'] = total_precision / n_samples
        metrics['evidence_recall'] = total_recall / n_samples
        metrics['evidence_f1'] = total_f1 / n_samples
        
        # F0.5 score (precision-weighted)
        beta = 0.5
        metrics['evidence_f0.5'] = ((1 + beta**2) * metrics['evidence_precision'] * metrics['evidence_recall']) / \
                                  (beta**2 * metrics['evidence_precision'] + metrics['evidence_recall']) \
                                  if (metrics['evidence_precision'] + metrics['evidence_recall']) > 0 else 0
    
    return metrics


def prepare_mtagec_batch(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    error_types: List[str],
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Prepare batch for MTAGEC training/inference.
    
    Args:
        batch: Dictionary containing input_ids, attention_mask, etc.
        tokenizer: Tokenizer for encoding/decoding
        device: Device to place tensors on
        error_types: List of error type labels
        max_length: Maximum sequence length
        
    Returns:
        Processed batch with unified labels for multi-task learning
    """
    # Move tensors to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # For training
    if "labels" in batch:
        labels = batch["labels"].to(device)
    else:
        labels = None
    
    # For evidence extraction (pointer network)
    if "evidence_positions" in batch:
        evidence_positions = batch["evidence_positions"].to(device)
        
        # Convert evidence positions to pointer indices
        # Add vocab_size offset to distinguish from vocabulary tokens
        vocab_size = tokenizer.vocab_size
        pointer_indices = evidence_positions + vocab_size
    else:
        pointer_indices = None
    
    # For error type classification
    if "error_types" in batch:
        error_type_indices = batch["error_types"].to(device)
        
        # Convert error type indices to unified label space
        # Add vocab_size + max_length offset to distinguish from vocab and pointers
        vocab_size = tokenizer.vocab_size
        error_type_indices = error_type_indices + vocab_size + max_length
    else:
        error_type_indices = None
    
    # Combine labels for unified multi-task learning
    if labels is not None and (pointer_indices is not None or error_type_indices is not None):
        # Create unified labels
        unified_labels = labels.clone()
        
        # Replace special positions with pointer indices
        if pointer_indices is not None:
            evidence_mask = batch.get("evidence_mask", torch.zeros_like(labels)).to(device)
            unified_labels[evidence_mask == 1] = pointer_indices[evidence_mask == 1]
        
        # Replace special positions with error type indices
        if error_type_indices is not None:
            error_mask = batch.get("error_mask", torch.zeros_like(labels)).to(device)
            unified_labels[error_mask == 1] = error_type_indices[error_mask == 1]
        
        processed_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": unified_labels,
        }
    else:
        processed_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            processed_batch["labels"] = labels
    
    return processed_batch


def postprocess_mtagec_output(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    error_types: List[str],
    max_length: int = 512,
) -> Tuple[str, List[str], List[str]]:
    """
    Postprocess MTAGEC model output to extract correction, evidence, and error types.
    
    Args:
        output_ids: Output token IDs from the model
        tokenizer: Tokenizer for decoding
        input_ids: Input token IDs
        error_types: List of error type labels
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (corrected_text, evidence_tokens, error_types)
    """
    vocab_size = tokenizer.vocab_size
    
    # Separate tokens by type
    correction_mask = output_ids < vocab_size
    pointer_mask = (output_ids >= vocab_size) & (output_ids < vocab_size + max_length)
    error_type_mask = output_ids >= vocab_size + max_length
    
    # Extract correction tokens
    correction_ids = output_ids.clone()
    correction_ids[~correction_mask] = tokenizer.pad_token_id
    corrected_text = tokenizer.decode(correction_ids, skip_special_tokens=True)
    
    # Extract evidence tokens (pointers to input)
    evidence_indices = output_ids[pointer_mask] - vocab_size
    evidence_tokens = []
    
    for idx in evidence_indices:
        if 0 <= idx < input_ids.size(1):
            token_id = input_ids[0, idx].item()  # Assuming batch size 1
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            evidence_tokens.append(token)
    
    # Extract error types
    error_type_indices = output_ids[error_type_mask] - (vocab_size + max_length)
    detected_error_types = []
    
    for idx in error_type_indices:
        if 0 <= idx < len(error_types):
            detected_error_types.append(error_types[idx])
    
    return corrected_text, evidence_tokens, detected_error_types


def find_separator_token(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> int:
    """
    Find the position of the separator token in the output sequence.
    
    Args:
        output_ids: Output token IDs from the model
        tokenizer: Tokenizer for encoding/decoding
        
    Returns:
        Position of separator token, or -1 if not found
    """
    sep_token_id = tokenizer.convert_tokens_to_ids(["<sep>"])[0]
    sep_positions = (output_ids == sep_token_id).nonzero(as_tuple=True)[0]
    
    if sep_positions.size(0) > 0:
        return sep_positions[0].item()
    else:
        return -1


def split_correction_explanation(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    post_explaining: bool = True,
) -> Tuple[str, str]:
    """
    Split the output into correction and explanation parts.
    
    Args:
        output_ids: Output token IDs from the model
        tokenizer: Tokenizer for decoding
        post_explaining: If True, correction comes before explanation
        
    Returns:
        Tuple of (correction, explanation)
    """
    sep_pos = find_separator_token(output_ids, tokenizer)
    
    if sep_pos == -1:
        # No separator found, treat everything as correction
        correction = tokenizer.decode(output_ids, skip_special_tokens=True)
        explanation = ""
    else:
        if post_explaining:
            # Correction before explanation
            correction = tokenizer.decode(output_ids[:sep_pos], skip_special_tokens=True)
            explanation = tokenizer.decode(output_ids[sep_pos+1:], skip_special_tokens=True)
        else:
            # Explanation before correction
            explanation = tokenizer.decode(output_ids[:sep_pos], skip_special_tokens=True)
            correction = tokenizer.decode(output_ids[sep_pos+1:], skip_special_tokens=True)
    
    return correction, explanation