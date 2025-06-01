#!/usr/bin/env python3
"""
Basic Empathy Model Trainer

A basic version of the local empathy trainer that works with the latest
versions of PyTorch and Transformers, with simplified parameters.
"""

import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from datasets import Dataset

def load_jsonl_data(file_path):
    """Load data from a JSONL file"""
    data = []
    problematic_lines = []
    
    # Handle encoding errors with a more robust approach
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Try to parse JSON
                entry = json.loads(line)
                
                # Handle different data formats
                if 'messages' in entry and isinstance(entry['messages'], list):
                    # OpenAI format with messages array
                    system_msg = next((msg['content'] for msg in entry['messages'] if msg.get('role') == 'system'), "")
                    user_msg = next((msg['content'] for msg in entry['messages'] if msg.get('role') == 'user'), "")
                    assistant_msg = next((msg['content'] for msg in entry['messages'] if msg.get('role') == 'assistant'), "")
                elif 'prompt' in entry and 'completion' in entry:
                    # Simple prompt-completion format
                    system_msg = ""
                    user_msg = entry['prompt']
                    assistant_msg = entry['completion']
                else:
                    # Unknown format
                    problematic_lines.append((line_num, "Unknown data format", line[:50]))
                    continue
                
                # Validate that we have both user and assistant messages
                if not user_msg or not assistant_msg:
                    problematic_lines.append((line_num, "Missing user or assistant message", line[:50]))
                    continue
                    
                # Clean and sanitize the text to prevent encoding issues
                try:
                    system_msg = system_msg.encode('ascii', 'ignore').decode('ascii')
                    user_msg = user_msg.encode('ascii', 'ignore').decode('ascii')
                    assistant_msg = assistant_msg.encode('ascii', 'ignore').decode('ascii')
                except Exception as e:
                    problematic_lines.append((line_num, f"Encoding error: {e}", line[:50]))
                    continue
                
                # Format for training
                formatted_text = f"System: {system_msg}\nUser: {user_msg}\nAssistant: {assistant_msg}"
                data.append({"text": formatted_text})
                
            except json.JSONDecodeError as e:
                problematic_lines.append((line_num, f"JSON error: {e}", line[:50]))
            except Exception as e:
                problematic_lines.append((line_num, f"Unexpected error: {e}", line[:50]))
    
    # Report problematic lines
    if problematic_lines:
        print(f"\nFound {len(problematic_lines)} problematic lines in the dataset:")
        for line_num, error, preview in problematic_lines[:5]:  # Show only first 5 issues
            print(f"Line {line_num}: {error} - '{preview}...'")
        if len(problematic_lines) > 5:
            print(f"...and {len(problematic_lines) - 5} more issues.")
    
    print(f"\nSuccessfully loaded {len(data)} valid examples for training.")
    return data
    
    return data

def train_empathy_model(data_file, output_dir, base_model="distilgpt2", epochs=3, batch_size=2, 
                     cpu_optimize=False, force_gpu=False, use_mixed_precision=True):
    """Train a local empathy model with optional mixed precision"""
    # Initialize device and mixed precision settings
    mixed_precision_dtype = "no"
    
    # Check for GPU availability
    if torch.cuda.is_available() and not cpu_optimize:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Check if GPU supports mixed precision
        if use_mixed_precision:
            if torch.cuda.is_bf16_supported():
                print("BFloat16 precision supported and enabled")
                mixed_precision_dtype = "bf16"
            else:
                print("FP16 precision supported and enabled")
                mixed_precision_dtype = "fp16"
        else:
            print("Mixed precision disabled by user")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
        
        if force_gpu:
            print("\nWARNING: GPU was requested but not available.")
            # Check if NVIDIA GPU exists
            import subprocess
            try:
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], 
                                      capture_output=True, text=True, check=True)
                if "NVIDIA" in result.stdout:
                    print("\nNVIDIA GPU detected in system:")
                    print(result.stdout)
                    print("\nSuggested fixes:")
                    print("1. Install the latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
                    print("2. Reinstall PyTorch with CUDA support using:")
                    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print("\nNo NVIDIA GPU detected in system.")
            except Exception as e:
                print(f"\nCould not check GPU information: {e}")
    
    # Print detailed device information
    if device.type == "cuda":
        print(f"\n[SUCCESS] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\nUsing CPU for training (CUDA not available)")
    
    # Load tokenizer and model
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))
    
    # Load and prepare data
    print(f"Loading data from {data_file}")
    data = load_jsonl_data(data_file)
    df = pd.DataFrame(data)
    
    # Handle small datasets appropriately
    if len(df) < 5:
        # For very small datasets, use all data for both training and evaluation
        train_df = df.copy()
        eval_df = df.copy()
        print(f"Dataset too small for splitting. Using all {len(df)} examples for both training and evaluation.")
    elif len(df) < 10:
        # For small datasets, use 80% for training, 20% for eval, but ensure at least 1 example for eval
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"Small dataset: Training on {len(train_df)} examples, evaluating on {len(eval_df)} examples")
    else:
        # Normal split for larger datasets
        train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
        print(f"Training on {len(train_df)} examples, evaluating on {len(eval_df)} examples")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Tokenize function with error handling
    def tokenize_function(examples):
        try:
            # Ensure text is properly encoded to prevent issues
            cleaned_texts = []
            for text in examples["text"]:
                # Replace any problematic characters
                cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
                cleaned_texts.append(cleaned_text)
                
            return tokenizer(cleaned_texts, padding="max_length", truncation=True, max_length=512)
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Return empty encodings as fallback
            return {"input_ids": [[0] * 512] * len(examples["text"]), 
                    "attention_mask": [[0] * 512] * len(examples["text"])}
    
    # Tokenize datasets
    print("Tokenizing datasets")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments with optimizations
    print("Setting up training with optimizations")
    
    # Apply CPU optimizations if requested
    if cpu_optimize and device.type == "cpu":
        print("Applying CPU optimizations")
        gradient_accumulation_steps = 4  # Reduce memory usage
        fp16 = False  # Not useful on CPU
        optim = "adamw_torch"  # More CPU-friendly optimizer
        logging_steps = 10  # More frequent logging for feedback
    else:
        gradient_accumulation_steps = 1
        fp16 = torch.cuda.is_available()  # Use fp16 if GPU available
        optim = "adamw_hf"
        logging_steps = 50
    
    # Calculate warmup steps (10% of training steps)
    total_steps = (len(train_dataset) // batch_size) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Create training arguments based on the available parameters
    # (compatible with older versions of Transformers)
    try:
        # Try with the full set of optimized parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir=f"{output_dir}/logs",
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=(mixed_precision_dtype == "fp16"),
            bf16=(mixed_precision_dtype == "bf16"),
            optim=optim,
            logging_steps=logging_steps,
            save_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=max(1, total_steps // 5),
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            learning_rate=5e-5,
            report_to="none"
        )
    except TypeError as e:
        print(f"Warning: Using simplified training arguments due to compatibility issue: {e}")
        # Ultra-simplified parameters for very old Transformers versions
        try:
            # First try the most basic parameters
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                logging_dir=f"{output_dir}/logs",
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=fp16,
                logging_steps=logging_steps,
                save_steps=len(train_dataset) // batch_size,  # Save once per epoch
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                learning_rate=5e-5
            )
        except TypeError as e2:
            print(f"Falling back to minimal training arguments: {e2}")
            # Last resort: absolute minimal parameters
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                logging_dir=f"{output_dir}/logs"
            )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")
    return model, tokenizer

def generate_response(prompt, model_path, max_length=150, temperature=0.7):
    """Generate a response using the trained model"""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Format prompt
    formatted_prompt = f"User: {prompt}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + max_length,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    response_parts = response.split("Assistant:")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Basic Local Empathy Model Trainer")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to the JSONL data file")
    train_parser.add_argument("--output", default="./local_empathy_model", help="Output directory for the model")
    train_parser.add_argument("--base-model", default="distilgpt2", help="Base model to fine-tune")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    train_parser.add_argument("--cpu-optimize", action="store_true", help="Apply CPU-specific optimizations for better performance")
    train_parser.add_argument("--force-gpu", action="store_true", help="Force GPU usage and provide diagnostics if not available")
    train_parser.add_argument("--use-mixed-precision", action="store_true", help="Enable mixed precision training (FP16/BF16) if supported")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a response")
    generate_parser.add_argument("--model", required=True, help="Path to the trained model")
    generate_parser.add_argument("--prompt", required=True, help="Prompt to generate a response for")
    generate_parser.add_argument("--max-length", type=int, default=150, help="Maximum length of the generated response")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_empathy_model(
            args.data,
            args.output,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cpu_optimize=args.cpu_optimize,
            force_gpu=args.force_gpu,
            use_mixed_precision=args.use_mixed_precision
        )
    
    elif args.command == "generate":
        response = generate_response(
            args.prompt,
            args.model,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"\nGenerated response:\n{response}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
