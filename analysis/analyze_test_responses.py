#!/usr/bin/env python3
"""
Test Response Analyzer
This script analyzes the test responses to evaluate the quality of the fine-tuned model
"""

import json
from collections import Counter
import re
import os

def load_responses(filename):
    """Load responses from JSON file"""
    responses = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                responses = data
            else:
                print("Error: File does not contain a list of responses")
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
    
    print(f"Loaded {len(responses)} valid responses")
    return responses

def analyze_responses(responses):
    """Analyze the responses"""
    # Initialize counters
    word_counts = Counter()
    empathy_indicators = Counter()
    response_lengths = []
    
    # Define empathy indicators to look for
    empathy_words = [
        'sorry', 'understand', 'feel', 'listen', 'support', 'care', 'here', 'okay',
        'normal', 'valid', 'challenging', 'difficult', 'tough', 'hard'
    ]
    
    # Analyze each response
    for response in responses:
        text = response.get('response', '').lower()
        prompt = response.get('prompt', '').lower()
        
        # Count words
        words = re.findall(r'\w+', text)
        word_counts.update(words)
        
        # Track response length
        response_lengths.append(len(words))
        
        # Count empathy indicators
        for word in empathy_words:
            if word in text:
                empathy_indicators[word] += 1
        
        # Check for validation phrases
        if "it's okay" in text or "it is okay" in text:
            empathy_indicators['validation_phrase'] += 1
        
        # Check for offering support
        if "i'm here" in text or "i am here" in text:
            empathy_indicators['offering_support'] += 1
        
        # Check for normalization
        if "normal" in text or "common" in text:
            empathy_indicators['normalization'] += 1
    
    # Calculate statistics
    avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    # Get most common words
    common_words = word_counts.most_common(20)
    
    return {
        'average_length': avg_length,
        'response_count': len(responses),
        'word_counts': word_counts,
        'empathy_indicators': empathy_indicators,
        'common_words': common_words
    }

def print_analysis(results):
    """Print analysis results"""
    print("\n=== Response Analysis ===")
    print(f"\nAnalyzed {results['response_count']} responses")
    print(f"Average response length (words): {results['average_length']:.2f}")
    
    print("\nMost common words:")
    for word, count in results['common_words'][:10]:
        print(f"{word}: {count}")
    
    print("\nEmpathy indicators:")
    for indicator, count in sorted(results['empathy_indicators'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['response_count']) * 100
        print(f"{indicator}: {count} ({percentage:.1f}%)")
    
    # Calculate overall empathy score
    empathy_count = sum(results['empathy_indicators'].values())
    empathy_score = empathy_count / (len(results['empathy_indicators']) * results['response_count']) * 10
    print(f"\nOverall empathy score (0-10): {empathy_score:.1f}")
    
    # Provide qualitative assessment
    if empathy_score >= 8:
        print("Assessment: Excellent empathetic responses")
    elif empathy_score >= 6:
        print("Assessment: Good empathetic responses")
    elif empathy_score >= 4:
        print("Assessment: Moderate empathetic responses")
    else:
        print("Assessment: Limited empathetic responses")

if __name__ == "__main__":
    # Load responses
    responses = load_responses('test_responses.json')
    
    # Analyze responses
    results = analyze_responses(responses)
    
    # Print analysis
    print_analysis(results)
    
    # Save analysis to file
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "empathy_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
