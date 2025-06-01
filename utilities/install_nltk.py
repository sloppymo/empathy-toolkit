#!/usr/bin/env python3
"""
NLTK Resource Downloader
This script downloads the required NLTK resources for SylvaFine
"""

import nltk

print("Downloading NLTK resources...")

# Download the punkt tokenizer
nltk.download('punkt')

# Download other potentially useful NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

print("NLTK resources downloaded successfully!")
print("You can now restart SylvaFine and try testing the model again.")
