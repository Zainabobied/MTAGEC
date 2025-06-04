#!/usr/bin/env python
"""
Setup script to download and install the required models for MTAGEC.
This script downloads AraBERT and AraT5v2 models from Hugging Face.
"""
import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

def download_models(output_dir="models/pretrained"):
    """
    Download the required models for MTAGEC.
    
    Args:
        output_dir: Directory to save the models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Models to download
    models = {
        "arabert": "aubmindlab/arabert-base-v2",
        "arat5v2": "UBC-NLP/AraT5v2-base-1024"
    }
    
    # Download models
    for model_name, model_path in models.items():
        print(f"Downloading {model_name} from {model_path}...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer_dir = os.path.join(output_dir, model_name + "-tokenizer")
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Tokenizer saved to {tokenizer_dir}")
        
        # Download model
        if "t5" in model_path.lower():
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        
        model_dir = os.path.join(output_dir, model_name + "-model")
        model.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")
    
    print("\nAll models downloaded successfully!")
    print("\nTo use these models, update your config.yaml file with the local paths:")
    print('  pretrained_model: "models/pretrained/arat5v2-model"')
    print('  or')
    print('  pretrained_model: "models/pretrained/arabert-model"')

def main():
    parser = argparse.ArgumentParser(description="Download models for MTAGEC")
    parser.add_argument("--output-dir", default="models/pretrained", 
                        help="Directory to save the models")
    args = parser.parse_args()
    
    download_models(args.output_dir)

if __name__ == "__main__":
    main()