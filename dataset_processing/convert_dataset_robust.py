#!/usr/bin/env python3
"""
Robust Dataset Converter for OpenAI Fine-Tuning
Converts the empathy_responses_clean.jsonl dataset to the format required by OpenAI's fine-tuning API.
"""

import json
import os

def fix_and_convert_dataset(input_file, output_file, system_message="You are an empathetic assistant that provides supportive and compassionate responses."):
    """
    Fix a multi-line JSON file and convert it to OpenAI's fine-tuning format.
    
    Args:
        input_file (str): Path to the input file with multi-line JSON objects
        output_file (str): Path to the output file for OpenAI format
        system_message (str): System message to include in each example
    """
    print(f"Processing file: {input_file}")
    
    # Read the entire file content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract JSON objects manually
    objects = []
    current_obj = ""
    in_object = False
    brace_count = 0
    
    for char in content:
        if char == '{':
            if not in_object:
                in_object = True
            brace_count += 1
            current_obj += char
        elif char == '}':
            brace_count -= 1
            current_obj += char
            if brace_count == 0 and in_object:
                # We've completed an object
                objects.append(current_obj)
                current_obj = ""
                in_object = False
        elif in_object:
            current_obj += char
    
    print(f"Extracted {len(objects)} JSON objects")
    
    # Parse each object and convert to OpenAI format
    openai_objects = []
    for i, obj_str in enumerate(objects):
        try:
            obj = json.loads(obj_str)
            if "prompt" in obj and "response" in obj:
                openai_obj = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": obj["prompt"]},
                        {"role": "assistant", "content": obj["response"]}
                    ]
                }
                openai_objects.append(openai_obj)
            else:
                print(f"Object {i+1} is missing 'prompt' or 'response' fields")
        except json.JSONDecodeError as e:
            print(f"Error parsing object {i+1}: {e}")
            # Try to fix common JSON issues
            try:
                fixed_str = obj_str.replace("'", '"')
                obj = json.loads(fixed_str)
                if "prompt" in obj and "response" in obj:
                    openai_obj = {
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": obj["prompt"]},
                            {"role": "assistant", "content": obj["response"]}
                        ]
                    }
                    openai_objects.append(openai_obj)
                    print(f"Fixed and parsed object {i+1}")
                else:
                    print(f"Fixed object {i+1} is missing 'prompt' or 'response' fields")
            except:
                print(f"Could not fix object {i+1}")
    
    print(f"Successfully converted {len(openai_objects)} objects to OpenAI format")
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in openai_objects:
            f.write(json.dumps(obj) + '\n')
    
    print(f"Successfully wrote {len(openai_objects)} examples to {output_file}")
    return True

if __name__ == "__main__":
    # File paths
    input_file = "C:/Users/comrade_morgy/Desktop/CONCLAVE/Data/empathy_responses_clean.jsonl"
    output_file = "C:/Users/comrade_morgy/Desktop/CONCLAVE/Data/empathy_responses_openai.jsonl"
    
    # Fix and convert the dataset
    if fix_and_convert_dataset(input_file, output_file):
        print(f"Conversion completed successfully!")
        print(f"The converted dataset is ready for fine-tuning with SylvaFine.")
        print(f"Path to the converted file: {output_file}")
    else:
        print("Conversion failed. Please check the errors above.")
