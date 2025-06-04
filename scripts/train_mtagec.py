"""
MTAGEC Model Training Script.
Trains a Transformer (e.g., AraT5, AraBART) on multi-task GEC + explanation.
"""
import os
import json
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    set_seed,
)
from models.mtagec_transformer import MTAGECModel
from scripts.utils import load_config, compute_metrics


class MTAGECDataset(Dataset):
    """
    Dataset for MTAGEC training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        training_mode: str = "self_rationalization",
        post_explaining: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            training_mode: Training mode (baseline, infusion, explanation, self_rationalization)
            post_explaining: Whether to generate correction before explanation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training_mode = training_mode
        self.post_explaining = post_explaining
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Add special tokens for explanation
        special_tokens = ["<sep>"]
        self.tokenizer.add_tokens(special_tokens)
        
        # Get separator token ID
        self.sep_token = "<sep>"
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get input and output based on training mode
        if self.training_mode == "baseline":
            # Standard GEC: input = erroneous, output = corrected
            input_text = item["modified"]
            output_text = item["original"]
            
        elif self.training_mode == "infusion":
            # Infusion: input = erroneous + explanation, output = corrected
            input_text = item["modified"]
            
            # Prepare explanation text
            explanations = []
            for error in item["errors"]:
                error_type = error["type"]
                explanation = error["explanation"]
                explanations.append(f"{error_type}: {explanation}")
            
            explanation_text = "; ".join(explanations)
            output_text = item["original"] + f" {self.sep_token} " + explanation_text
            
        elif self.training_mode == "explanation":
            # Explanation only: input = erroneous, output = explanation
            input_text = item["modified"]
            
            # Prepare explanation text
            explanations = []
            for error in item["errors"]:
                error_type = error["type"]
                explanation = error["explanation"]
                explanations.append(f"{error_type}: {explanation}")
            
            output_text = "; ".join(explanations)
            
        elif self.training_mode == "self_rationalization":
            # Self-rationalization: input = erroneous, output = corrected + explanation
            input_text = item["modified"]
            
            # Prepare explanation text
            explanations = []
            for error in item["errors"]:
                error_type = error["type"]
                explanation = error["explanation"]
                explanations.append(f"{error_type}: {explanation}")
            
            explanation_text = "; ".join(explanations)
            
            # Order based on post_explaining flag
            if self.post_explaining:
                # Correction before explanation
                output_text = item["original"] + f" {self.sep_token} " + explanation_text
            else:
                # Explanation before correction
                output_text = explanation_text + f" {self.sep_token} " + item["original"]
        
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        
        # Tokenize input and output
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        outputs = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Prepare final inputs
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = outputs["input_ids"].squeeze()
        
        # Replace padding token ID with -100 in labels for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train(config_path: str):
    """
    Train MTAGEC model.
    
    Args:
        config_path: Path to config file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Create output directory
    output_dir = os.path.join("models", "checkpoints", config["model"]["name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained_model"],
        use_fast=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model_config = AutoConfig.from_pretrained(config["model"]["pretrained_model"])
    model = MTAGECModel.from_pretrained(
        config["model"]["pretrained_model"],
        config=model_config,
    )
    
    # Resize token embeddings to account for added special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare datasets
    train_dataset = MTAGECDataset(
        data_path=os.path.join(config["data"]["explagec_dir"], "train.json"),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        training_mode=config.get("training_mode", "self_rationalization"),
        post_explaining=config["model"]["decoding"]["post_explaining"],
    )
    
    val_dataset = MTAGECDataset(
        data_path=os.path.join(config["data"]["explagec_dir"], "val.json"),
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        training_mode=config.get("training_mode", "self_rationalization"),
        post_explaining=config["model"]["decoding"]["post_explaining"],
    )
    
    # Prepare dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
    )
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["training"]["learning_rate"],
        weight_decay=config["model"]["training"]["weight_decay"],
    )
    
    total_steps = len(train_dataloader) * config["model"]["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["model"]["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config["model"]["training"]["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['model']['training']['epochs']}")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                lambda_weight=config["model"]["training"]["lambda_weight"],
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config["model"]["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    lambda_weight=config["model"]["training"]["lambda_weight"],
                )
                
                loss = outputs.loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['model']['training']['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Save configuration
            with open(os.path.join(model_path, "training_config.yaml"), "w") as f:
                yaml.dump(config, f)
            
            print(f"Saved checkpoint to {model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{config['model']['training']['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config["model"]["training"]["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training completed!")
    return os.path.join(output_dir, f"checkpoint-epoch-{epoch+1-patience_counter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    best_model_path = train(args.config)
    print(f"Best model saved at: {best_model_path}")
