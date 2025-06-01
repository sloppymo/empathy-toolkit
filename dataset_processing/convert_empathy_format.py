#!/usr/bin/env python3
"""
Empathy Dataset Format Converter

Converts between different empathy dataset formats to ensure compatibility
with the training script.
"""

import os
import json
import argparse
from typing import List, Dict, Any

def convert_tagged_to_openai_format(input_file: str, output_file: str) -> int:
    """
    Convert a tagged empathy dataset (prompt, response, tags) to OpenAI format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        
    Returns:
        Number of examples converted
    """
    converted_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Read input data
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                
                # Check if entry has the expected format
                if "prompt" in entry and "response" in entry:
                    # Convert to OpenAI format
                    openai_entry = {
                        "messages": [
                            {"role": "system", "content": "You are an empathetic assistant that provides supportive responses to people in emotional distress."},
                            {"role": "user", "content": entry["prompt"]},
                            {"role": "assistant", "content": entry["response"]}
                        ]
                    }
                    
                    converted_data.append(openai_entry)
            except json.JSONDecodeError:
                continue
    
    # Write converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')
    
    return len(converted_data)

def convert_to_simple_format(input_file: str, output_file: str) -> int:
    """
    Convert a dataset to simple prompt-completion format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        
    Returns:
        Number of examples converted
    """
    converted_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Read input data
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                
                # Handle different formats
                if "messages" in entry and isinstance(entry["messages"], list):
                    # OpenAI format
                    user_messages = [msg for msg in entry["messages"] if msg.get("role") == "user"]
                    assistant_messages = [msg for msg in entry["messages"] if msg.get("role") == "assistant"]
                    
                    if user_messages and assistant_messages:
                        simple_entry = {
                            "prompt": user_messages[0]["content"],
                            "completion": assistant_messages[0]["content"]
                        }
                        converted_data.append(simple_entry)
                
                elif "prompt" in entry and "response" in entry:
                    # Tagged format
                    simple_entry = {
                        "prompt": entry["prompt"],
                        "completion": entry["response"]
                    }
                    converted_data.append(simple_entry)
                
                elif "prompt" in entry and "completion" in entry:
                    # Already in simple format
                    converted_data.append(entry)
            
            except json.JSONDecodeError:
                continue
    
    # Write converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in converted_data:
            f.write(json.dumps(entry) + '\n')
    
    return len(converted_data)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Convert empathy dataset formats")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--format", choices=["openai", "simple"], default="openai",
                       help="Output format (openai or simple)")
    
    args = parser.parse_args()
    
    if args.format == "openai":
        count = convert_tagged_to_openai_format(args.input, args.output)
        print(f"Converted {count} examples to OpenAI format")
    else:
        count = convert_to_simple_format(args.input, args.output)
        print(f"Converted {count} examples to simple format")

if __name__ == "__main__":
    main()
