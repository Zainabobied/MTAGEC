"""
Evaluation script for MTAGEC.

This script evaluates the MTAGEC model on test datasets.
It supports evaluation on both ExplAGEC and QALB benchmarks.
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Any
from transformers import AutoTokenizer
from models.mtagec_transformer import MTAGECModel
from scripts.utils import load_config, compute_metrics, split_correction_explanation


def evaluate_model(
    model,
    tokenizer,
    test_data,
    config,
    output_file=None,
    dataset_name="test",
):
    """
    Evaluate model on test data.
    
    Args:
        model: MTAGEC model
        tokenizer: Tokenizer
        test_data: Test data
        config: Configuration
        output_file: Path to save predictions (optional)
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary of metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Prepare lists for metrics
    predictions = []
    references = []
    error_type_preds = []
    error_type_refs = []
    evidence_preds = []
    evidence_refs = []
    
    # Generate predictions
    for item in tqdm(test_data, desc=f"Evaluating on {dataset_name}"):
        # Get input text
        input_text = item["modified"] if "modified" in item else item["original"]
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            max_length=config["model"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate output
        with torch.no_grad():
            output_ids = model.generate_with_explanation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=config["model"]["decoding"]["max_length"],
                num_beams=config["model"]["decoding"]["beam_size"],
                top_p=config["model"]["decoding"]["top_p"],
                top_k=config["model"]["decoding"]["top_k"],
                temperature=config["model"]["decoding"]["temperature"],
            )
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Split correction and explanation
        correction, explanation = split_correction_explanation(
            output_ids[0],
            tokenizer,
            post_explaining=config["model"]["decoding"]["post_explaining"],
        )
        
        # Get reference text
        reference = item["original"] if "original" in item else item["corrected"]
        
        # Extract error types and evidence from explanation
        pred_error_types = []
        pred_evidence = []
        
        if explanation:
            # Parse explanation
            for exp_part in explanation.split(";"):
                exp_part = exp_part.strip()
                if not exp_part:
                    continue
                
                # Extract error type and evidence
                if ":" in exp_part:
                    error_type, exp_text = exp_part.split(":", 1)
                    pred_error_types.append(error_type.strip())
                    
                    # Extract evidence words from explanation
                    evidence_words = []
                    for word in exp_text.split():
                        if word.startswith("'") and word.endswith("'"):
                            evidence_words.append(word.strip("'"))
                    
                    pred_evidence.append(evidence_words)
        
        # Get reference error types and evidence
        ref_error_types = []
        ref_evidence = []
        
        if "errors" in item:
            for error in item["errors"]:
                ref_error_types.append(error["type"])
                
                # Extract evidence indices
                evidence_indices = error["evidence"]
                evidence_words = []
                
                # Convert indices to words
                input_words = input_text.split()
                for idx in evidence_indices:
                    if 0 <= idx < len(input_words):
                        evidence_words.append(input_words[idx])
                
                ref_evidence.append(evidence_words)
        
        # Add to lists for metrics
        predictions.append(correction)
        references.append(reference)
        
        if pred_error_types:
            error_type_preds.append(pred_error_types)
        if ref_error_types:
            error_type_refs.append(ref_error_types)
        
        if pred_evidence:
            evidence_preds.append(pred_evidence)
        if ref_evidence:
            evidence_refs.append(ref_evidence)
        
        # Save prediction
        if output_file:
            item["prediction"] = correction
            item["explanation"] = explanation
            item["pred_error_types"] = pred_error_types
            item["pred_evidence"] = pred_evidence
    
    # Compute metrics
    metrics = compute_metrics(
        predictions=predictions,
        references=references,
        error_type_preds=error_type_preds if error_type_preds and error_type_refs else None,
        error_type_refs=error_type_refs if error_type_preds and error_type_refs else None,
        evidence_preds=evidence_preds if evidence_preds and evidence_refs else None,
        evidence_refs=evidence_refs if evidence_preds and evidence_refs else None,
    )
    
    # Save predictions
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return metrics


def main():
    """
    Main function to evaluate model.
    """
    parser = argparse.ArgumentParser(description="Evaluate MTAGEC model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--explagec", action="store_true", help="Evaluate on ExplAGEC test set")
    parser.add_argument("--qalb2014", action="store_true", help="Evaluate on QALB-2014 test set")
    parser.add_argument("--qalb2015", action="store_true", help="Evaluate on QALB-2015 test set")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MTAGECModel.from_pretrained(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on ExplAGEC test set
    if args.explagec:
        test_file = os.path.join(config["data"]["explagec_dir"], "test.json")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            output_file = os.path.join(args.output_dir, "explagec_predictions.json")
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                config=config,
                output_file=output_file,
                dataset_name="ExplAGEC",
            )
            
            print("\nExplAGEC Test Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save metrics
            metrics_file = os.path.join(args.output_dir, "explagec_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            print(f"Warning: ExplAGEC test file not found at {test_file}")
    
    # Evaluate on QALB-2014 test set
    if args.qalb2014:
        test_file = os.path.join(config["data"]["processed_dir"], "qalb2014", "test.json")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            output_file = os.path.join(args.output_dir, "qalb2014_predictions.json")
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                config=config,
                output_file=output_file,
                dataset_name="QALB-2014",
            )
            
            print("\nQALB-2014 Test Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save metrics
            metrics_file = os.path.join(args.output_dir, "qalb2014_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            print(f"Warning: QALB-2014 test file not found at {test_file}")
    
    # Evaluate on QALB-2015 test set
    if args.qalb2015:
        test_file = os.path.join(config["data"]["processed_dir"], "qalb2015", "test.json")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            output_file = os.path.join(args.output_dir, "qalb2015_predictions.json")
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                config=config,
                output_file=output_file,
                dataset_name="QALB-2015",
            )
            
            print("\nQALB-2015 Test Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save metrics
            metrics_file = os.path.join(args.output_dir, "qalb2015_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            print(f"Warning: QALB-2015 test file not found at {test_file}")
    
    # If no specific dataset is selected, evaluate on all
    if not (args.explagec or args.qalb2014 or args.qalb2015):
        print("No dataset specified. Please use --explagec, --qalb2014, or --qalb2015.")


if __name__ == "__main__":
    main()