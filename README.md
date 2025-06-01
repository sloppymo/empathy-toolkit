# Empathy Toolkit

A comprehensive collection of tools for empathy modeling, dataset processing, and language model fine-tuning.

## Overview

This toolkit contains utilities and scripts for working with empathy-related datasets, training machine learning models for empathy detection and generation, and analyzing empathetic responses. Built with Python 3.13 compatibility.

## Directory Structure

- `/dataset_processing` - Tools for converting, cleaning, and augmenting datasets
- `/model_training` - Scripts for fine-tuning and training empathy models
- `/analysis` - Tools for analyzing and visualizing model performance
- `/utilities` - Helper utilities for system configuration
- `/applications` - End-user applications and dashboards
- `/api` - API services for deploying empathy models

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up NLTK: `python utilities/install_nltk.py`
4. Explore the toolkit!

## Key Features

- Dataset conversion and formatting for various LLM fine-tuning formats
- Empathy metrics and analysis tools
- Interactive dashboards for visualizing empathy model performance
- Fine-tuning utilities for LLMs
- Data augmentation techniques specific to empathy datasets
- Direct OpenAI API integration with httpx (avoiding SDK compatibility issues)

## Python 3.13 Compatibility

This toolkit is designed to work with Python 3.13, taking into account recent changes that affect dependencies like Rust-based packages. We use alternatives where needed to ensure compatibility.

## License

MIT License
