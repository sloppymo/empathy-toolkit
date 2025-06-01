# Empathy Toolkit

<div align="center">

![Empathy Toolkit](https://img.shields.io/badge/Empathy-Toolkit-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13_Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

A comprehensive collection of tools for developing, training, and deploying empathy-focused AI systems. This toolkit addresses the challenges of empathetic response generation and analysis with Python 3.13 compatibility.

## 🧠 Overview

The Empathy Toolkit provides a unified framework for working with empathy-related data, models, and applications. It supports the entire empathy AI development lifecycle, from dataset processing and model training to deployment and analysis.

### Why Empathy Modeling Matters

Empathetic AI systems that can understand and respond appropriately to human emotions are critical for:
- 🤝 Building trust in human-AI interactions
- 🎯 Creating more effective conversational agents
- 🔍 Understanding and analyzing emotional content in text
- 📈 Improving user satisfaction and engagement

## 📁 Directory Structure

```
empathy-toolkit/
├── dataset_processing/    # Tools for converting, cleaning, and augmenting datasets
├── model_training/        # Scripts for fine-tuning and training empathy models
├── analysis/              # Tools for analyzing and visualizing performance
├── utilities/             # Helper utilities for system configuration
├── applications/          # End-user applications and dashboards
└── api/                   # API services for deploying empathy models
```

## 🚀 Key Components

### Dataset Processing Tools
- **Dataset Conversion**: Transform data between various formats (JSON, JSONL, CSV)
- **Robust Processing**: Handle edge cases, malformed data, and ensure consistency
- **Format Standardization**: Normalize empathy annotations and response structures

### Model Training
- **Basic Empathy Trainer**: Entry-level training pipeline for empathy classification and generation
- **Advanced Model Training**: Fine-tuning capabilities for LLMs with empathy-specific parameters
- **Python 3.13 Compatible**: Avoids Rust-dependent packages that cause issues with newer Python

### Analysis Tools
- **Response Analysis**: Evaluate and score AI responses based on empathy metrics
- **Benchmarking**: Compare different models and approaches using standardized metrics
- **Visualization**: Generate insights through data visualizations and metrics dashboards

### Applications
- **Interactive Dashboards**: Monitor and control empathy model performance
- **Chatbot UIs**: Test empathy models in real-time conversation settings
- **Streamlit Integration**: Easy deployment of web interfaces for non-technical users

### API Services
- **Flask Backend**: Lightweight API services for model deployment
- **Direct OpenAI Integration**: Uses httpx for API calls instead of SDK (avoiding Python 3.13 compatibility issues)
- **Streaming Support**: Real-time response generation with streaming capabilities

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/empathy-toolkit.git
cd empathy-toolkit

# Install dependencies
pip install -r requirements.txt

# Optional: Install as a package
pip install -e .

# Set up NLTK resources
python utilities/install_nltk.py
```

## 🔧 Usage Examples

### Dataset Processing
```python
# Convert dataset to empathy format
from empathy_toolkit.dataset_processing import convert_empathy_format

convert_empathy_format.convert_file(
    input_file="raw_data.json", 
    output_file="empathy_dataset.jsonl"
)
```

### Model Training
```python
# Train a basic empathy model
from empathy_toolkit.model_training import basic_empathy_trainer

trainer = basic_empathy_trainer.EmpathyModelTrainer(
    dataset_path="empathy_dataset.jsonl",
    model_output_dir="my_empathy_model"
)
trainer.train(epochs=3)
```

### Analysis
```python
# Analyze test responses
from empathy_toolkit.analysis import analyze_test_responses

analyzer = analyze_test_responses.EmpathyAnalyzer(
    response_file="model_responses.jsonl",
    ground_truth="ground_truth.jsonl"
)
results = analyzer.evaluate()
analyzer.generate_report("analysis_report.html")
```

## 🧪 Python 3.13 Compatibility

This toolkit is designed to work with Python 3.13, addressing compatibility issues with Rust-dependent packages. We use:

- **httpx** instead of the OpenAI SDK for API calls
- Carefully selected dependencies that don't require Rust compilation
- Alternative implementations for common NLP tasks

## 📊 Example Projects

- **Empathy Chatbot**: Deploy a Flask-based chatbot with empathetic responses
- **Empathy Dashboard**: Streamlit application for monitoring model performance
- **Dataset Converter**: Bulk convert datasets to formats suitable for empathy modeling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Morgan Roberts MSW LPHA - morgan@forestwithintherapy.com

Project Link: [https://github.com/sloppymo/empathy-toolkit](https://github.com/sloppymo/empathy-toolkit)

