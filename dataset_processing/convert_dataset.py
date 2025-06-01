#!/usr/bin/env python3
"""
Dataset Converter for OpenAI Fine-Tuning
Converts the empathy_responses_clean.jsonl dataset to the format required by OpenAI's fine-tuning API.
"""

import json
import jsonlines
import os
from pathlib import Path

def convert_dataset(input_file, output_file, system_message="You are an empathetic assistant that provides supportive and compassionate responses."):
    """
    Convert a dataset from {prompt, response} format to OpenAI's fine-tuning format with messages.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        system_message (str): System message to include in each example
    """
    print(f"Converting dataset from {input_file} to {output_file}...")
    
    # Load the original dataset
    original_examples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        original_examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line[:50]}... - {e}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    print(f"Loaded {len(original_examples)} examples from the original dataset")
    
    # Convert to the OpenAI format
    new_examples = []
    for i, example in enumerate(original_examples):
        try:
            if "prompt" in example and "response" in example:
                new_example = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["response"]}
                    ]
                }
                new_examples.append(new_example)
            else:
                print(f"Example {i} is missing 'prompt' or 'response' fields: {example}")
        except Exception as e:
            print(f"Error converting example {i}: {e}")
    
    print(f"Converted {len(new_examples)} examples to OpenAI format")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the new dataset
    try:
        with jsonlines.open(output_file, 'w') as writer:
            for example in new_examples:
                writer.write(example)
        print(f"Successfully wrote {len(new_examples)} examples to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

def validate_dataset(file_path):
    """
    Validate that a dataset is in the correct format for OpenAI fine-tuning.
    
    Args:
        file_path (str): Path to the JSONL file to validate
    """
    print(f"Validating dataset: {file_path}")
    valid_count = 0
    invalid_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        example = json.loads(line)
                        if "messages" in example and isinstance(example["messages"], list):
                            # Check if messages have the required fields
                            valid_msg = True
                            for msg in example["messages"]:
                                if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
                                    valid_msg = False
                                    break
                            
                            if valid_msg:
                                valid_count += 1
                            else:
                                invalid_count += 1
                                print(f"Example {i+1} has invalid message format")
                        else:
                            invalid_count += 1
                            print(f"Example {i+1} is missing 'messages' field or it's not a list")
                    except json.JSONDecodeError as e:
                        invalid_count += 1
                        print(f"Error parsing line {i+1}: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    print(f"Validation results:")
    print(f"  Valid examples: {valid_count}")
    print(f"  Invalid examples: {invalid_count}")
    print(f"  Total examples: {valid_count + invalid_count}")
    
    return valid_count > 0 and invalid_count == 0

if __name__ == "__main__":
    # Define file paths
    input_file = "empathy_responses_clean.jsonl"
    output_file = "empathy_responses_openai_format.jsonl"
    
    # Get the current directory
    current_dir = Path(__file__).parent
    input_path = current_dir / input_file
    output_path = current_dir / output_file
    
    # Convert the dataset
    if convert_dataset(input_path, output_path):
        print("Conversion completed successfully!")
        
        # Validate the converted dataset
        if validate_dataset(output_path):
            print("Dataset validation passed! The dataset is ready for fine-tuning.")
        else:
            print("Dataset validation failed. Please check the errors above.")
    else:
        print("Conversion failed. Please check the errors above.")
