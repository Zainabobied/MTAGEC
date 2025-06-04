"""
Preprocessing script for MTAGEC.

This script preprocesses the raw data and prepares it for training.
It handles both synthetic ExplAGEC data and QALB benchmark datasets.
"""
import os
import json
import argparse
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
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


def preprocess_explagec(
    input_file: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> None:
    """
    Preprocess ExplAGEC dataset and split into train/val/test.
    
    Args:
        input_file: Path to ExplAGEC JSON file
        output_dir: Directory to save processed files
        split_ratio: Ratio for train/val/test split
    """
    print(f"Preprocessing ExplAGEC data from {input_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate split sizes
    total_size = len(data)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save splits
    for split_name, split_data in [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]:
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(split_data)} examples to {output_file}")
    
    # Create a smaller sample for quick testing
    sample_size = min(1000, len(train_data))
    sample_data = train_data[:sample_size]
    sample_file = os.path.join(output_dir, "sample.json")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample_data)} examples to {sample_file} for quick testing")


def preprocess_qalb(
    qalb_dir: str,
    output_dir: str,
    dataset: str = "qalb2014"
) -> None:
    """
    Preprocess QALB dataset.
    
    Args:
        qalb_dir: Directory containing QALB data
        output_dir: Directory to save processed files
        dataset: Which QALB dataset to process ("qalb2014" or "qalb2015")
    """
    print(f"Preprocessing {dataset} data from {qalb_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(qalb_dir, f"{split}.txt")
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping")
            continue
        
        # Read and process data
        processed_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # QALB format has pairs of lines: original and corrected
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines):
                    break
                    
                original = lines[i].strip()
                corrected = lines[i + 1].strip()
                
                # Skip empty lines
                if not original or not corrected:
                    continue
                
                entry = {
                    "original": original,
                    "corrected": corrected
                }
                processed_data.append(entry)
        
        # Save processed data
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_data)} examples to {output_file}")


def main():
    """
    Main function to preprocess data.
    """
    parser = argparse.ArgumentParser(description='Preprocess data for MTAGEC')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--explagec', action='store_true', help='Preprocess ExplAGEC data')
    parser.add_argument('--qalb', action='store_true', help='Preprocess QALB data')
    parser.add_argument('--qalb-dataset', choices=['qalb2014', 'qalb2015'], default='qalb2014',
                        help='Which QALB dataset to process')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process ExplAGEC data
    if args.explagec:
        input_file = config['data']['synthetic']['output_file']
        output_dir = config['data']['explagec_dir']
        preprocess_explagec(input_file, output_dir)
    
    # Process QALB data
    if args.qalb:
        qalb_dir = os.path.join(config['data']['qalb_dir'], args.qalb_dataset)
        output_dir = os.path.join(config['data']['processed_dir'], args.qalb_dataset)
        preprocess_qalb(qalb_dir, output_dir, args.qalb_dataset)
    
    # If no specific task is selected, process all
    if not (args.explagec or args.qalb):
        # Process ExplAGEC
        input_file = config['data']['synthetic']['output_file']
        output_dir = config['data']['explagec_dir']
        preprocess_explagec(input_file, output_dir)
        
        # Process QALB 2014
        qalb_dir = os.path.join(config['data']['qalb_dir'], 'qalb2014')
        output_dir = os.path.join(config['data']['processed_dir'], 'qalb2014')
        preprocess_qalb(qalb_dir, output_dir, 'qalb2014')
        
        # Process QALB 2015
        qalb_dir = os.path.join(config['data']['qalb_dir'], 'qalb2015')
        output_dir = os.path.join(config['data']['processed_dir'], 'qalb2015')
        preprocess_qalb(qalb_dir, output_dir, 'qalb2015')


if __name__ == "__main__":
    main()