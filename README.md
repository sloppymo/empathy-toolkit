# Empathy Toolkit

<div align="center">

![Empathy Toolkit](https://img.shields.io/badge/Empathy-Toolkit-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13_Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

A comprehensive toolkit for multi-dimensional empathy scoring, analysis, and visualization with full Python 3.13 compatibility. Designed for clinical and mental health applications, researchers, and AI developers focused on creating more empathetic systems.

## 🧠 Overview

The Empathy Toolkit provides a unified API for scoring empathy across five dimensions (cognitive, emotional, behavioral, contextual, and cultural) with detailed feedback and improvement suggestions. Built with Python 3.13 compatibility in mind, it avoids Rust-dependent packages like the OpenAI SDK in favor of direct API calls using `httpx`.

### Why Multi-dimensional Empathy Analysis Matters

Empathy is not a single trait but a complex construct with multiple dimensions:
- 🧩 **Cognitive Empathy**: Understanding another's perspective
- 💫 **Emotional Empathy**: Feeling what others feel
- 🤲 **Behavioral Empathy**: Responding supportively to others
- 🌐 **Contextual Empathy**: Adapting to specific situations
- 🌍 **Cultural Empathy**: Respecting cultural differences

## 📁 Project Structure

```
empathy-toolkit/
├── analysis/               # Core empathy scoring framework
│   └── empathy_scorer.py   # Multi-dimensional scoring system
├── applications/           # Example applications using the toolkit
├── cli.py                  # Command-line interface
├── visualize.py            # Visualization utilities
├── cache_manager.py        # Cache management tools
├── empathy_toolkit_api.py  # Unified API for all functionality
├── integration_example.py  # Example integration
└── requirements.txt        # Python 3.13 compatible dependencies
```

## 🚀 Key Features

### Unified API
- **Multi-dimensional Analysis**: Score empathy across five dimensions
- **Detailed Feedback**: Get strengths, improvement areas, and suggestions
- **Batch Processing**: Score multiple responses efficiently with controlled concurrency
- **Python 3.13 Compatibility**: Uses `httpx` instead of OpenAI SDK (avoids Rust dependencies)
- **Efficient Caching**: Reduce API costs with intelligent caching

### Command Line Interface
- **Direct Scoring**: Score individual responses from command line
- **Dataset Enhancement**: Add empathy scores to entire datasets
- **Dimension Descriptions**: View detailed descriptions of empathy dimensions
- **Rich Console Output**: Beautifully formatted results with the `rich` library
- **JSON Export**: Format results as structured JSON for further processing

### Visualization Tools
- **Radar Charts**: Compare empathy dimensions visually
- **Bar Charts**: Side-by-side comparison of scores
- **Heatmaps**: Analyze patterns in large datasets
- **Custom Styling**: Configurable visualization options
- **Export Capabilities**: Save charts as PNG, SVG, or PDF

### Cache Management
- **Detailed Statistics**: View cache usage and savings
- **Clean Old Entries**: Remove outdated cache entries
- **Export Cache Data**: Export cache contents for analysis
- **Rich Terminal UI**: Interactive terminal-based management

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/empathy-toolkit.git
cd empathy-toolkit

# Install dependencies (Python 3.13 compatible)
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 🔧 Usage Examples

### Command Line Interface

```bash
# Score an individual response
python cli.py score --response "I understand how difficult that must be for you. Would you like to talk about it more?"

# Score with context
python cli.py score --response "I'm here for you" --context "Patient: I just received a difficult diagnosis."

# Export as JSON
python cli.py score --response "Tell me more about how you're feeling" --format json

# Enhance a dataset with empathy scores
python cli.py enhance --input dataset.csv --output enhanced_dataset.csv --response-column "response" --context-column "context"

# View empathy dimension descriptions
python cli.py dimensions
```

### Visualization

```bash
# Create a radar chart from a dataset
python visualize.py --input enhanced_dataset.csv --type radar --output radar_chart.png

# Create a bar chart
python visualize.py --input enhanced_dataset.csv --type bar --output bar_chart.png

# Create a heatmap
python visualize.py --input enhanced_dataset.csv --type heatmap --output heatmap.png
```

### Cache Management

```bash
# View cache statistics
python cache_manager.py view

# View detailed cache information
python cache_manager.py view --details

# Clear cache entries older than 7 days
python cache_manager.py clear --older-than 7d

# Export cache data for analysis
python cache_manager.py export --output cache_export.json
```

### Programmatic Usage

```python
import asyncio
from empathy_toolkit_api import EmpathyToolkit

async def analyze_empathy():
    # Initialize the toolkit (will use OPENAI_API_KEY env variable)
    toolkit = EmpathyToolkit(use_cached_results=True)
    
    # Score a single response
    response = "I understand this is difficult for you. Let's talk about how you're feeling."
    context = "Patient: I've been struggling with anxiety lately."
    
    score, feedback = await toolkit.score_empathy(
        response=response,
        context=context,
        include_feedback=True
    )
    
    print(f"Total Empathy Score: {score.total}")
    print(f"Cognitive: {score.cognitive}")
    print(f"Emotional: {score.emotional}")
    print(f"Behavioral: {score.behavioral}")
    print(f"Contextual: {score.contextual}")
    print(f"Cultural: {score.cultural}")
    
    # Process multiple responses in batch (with controlled concurrency)
    items = [
        {"id": 1, "response": "I understand how you feel.", "context": "User: I'm feeling sad."},
        {"id": 2, "response": "That must be challenging.", "context": "User: My job is stressful."},
        {"id": 3, "response": "Have you tried just being happy?", "context": "User: I'm depressed."},
    ]
    
    results = await toolkit.batch_score_empathy(
        items=items,
        include_feedback=True,
        max_concurrency=3
    )
    
    for result in results:
        print(f"Response {result['id']} - Total Score: {result['score']['total']}")

# Run the async function
asyncio.run(analyze_empathy())
```

## 📊 Multi-dimensional Empathy Framework

The toolkit evaluates empathy across five key dimensions:

1. **Cognitive Empathy** (Understanding)
   - Comprehending others' thoughts and feelings
   - Taking others' perspectives
   - Recognizing thought patterns

2. **Emotional Empathy** (Feeling)
   - Sharing or mirroring others' emotions
   - Showing emotional resonance
   - Expressing genuine emotional concern

3. **Behavioral Empathy** (Acting)
   - Demonstrating supportive actions
   - Active listening behaviors
   - Providing appropriate assistance

4. **Contextual Empathy** (Adapting)
   - Considering situational factors
   - Adapting to specific circumstances
   - Recognizing unique contextual needs

5. **Cultural Empathy** (Respecting)
   - Awareness of cultural differences
   - Sensitivity to cultural values
   - Cross-cultural understanding

## 🔄 Batch Processing

The toolkit provides efficient batch processing for analyzing multiple responses:

- **Controlled Concurrency**: Limit simultaneous API calls to prevent rate limiting
- **Intelligent Error Handling**: Resilient to individual failures in batch processing
- **Progress Tracking**: See real-time progress during batch operations
- **CSV/JSONL Support**: Work with standard data formats

## 🧪 Python 3.13 Compatibility

This toolkit is designed to work with Python 3.13, addressing compatibility issues with Rust-dependent packages. We use:

- **httpx** instead of the OpenAI SDK for API calls (avoiding Rust compilation issues)
- Direct API interaction with structured JSON payloads
- Carefully selected dependencies that don't require Rust compilation
- Flask for web interfaces rather than FastAPI (which has Python 3.13 compatibility issues)

## 📝 Advanced Configuration

```python
# Create a toolkit instance with custom settings
toolkit = EmpathyToolkit(
    use_gpu=False,                 # No GPU acceleration needed
    models={                       # Custom model configuration
        "primary": "gpt-4o",       # For scoring
        "feedback": "gpt-3.5-turbo" # For detailed feedback
    },
    api_key="your_api_key",        # Can also use OPENAI_API_KEY env var
    use_cached_results=True,       # Enable caching to reduce API costs
    cache_dir=".custom_cache"      # Custom cache location
)
```

## 🔑 Environment Variables

The toolkit uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required unless passed directly)
- `EMPATHY_CACHE_DIR`: Custom cache directory (optional)
- `EMPATHY_MAX_CONCURRENCY`: Default max concurrent API calls (optional)

## 💡 Performance Tips

1. **Enable Caching**: Reduce API costs and speed up repeated analyses
2. **Tune Concurrency**: Balance between speed and API rate limits
3. **Batch Process**: More efficient than individual calls for large datasets
4. **Regular Cache Cleanup**: Prevent excessive disk usage

## 🐛 Troubleshooting

### Common Issues

- **API Key Errors**: Ensure OPENAI_API_KEY is set in your environment or .env file
- **Rate Limiting**: Decrease max_concurrency to avoid hitting API limits
- **Asyncio Errors**: Avoid nested event loops in async code
- **Cache Corruption**: Use cache_manager.py clear if you encounter issues

## 📖 Reference

### Score Object Structure

```json
{
  "cognitive": 7.0,   // Understanding perspective
  "emotional": 6.0,  // Emotional connection
  "behavioral": 8.0, // Supportive actions
  "contextual": 5.0, // Context awareness
  "cultural": 4.0,   // Cultural sensitivity
  "total": 6.4       // Weighted average
}
```

### Feedback Object Structure

```json
{
  "strengths": ["Shows understanding", "Offers support"],
  "areas_for_improvement": ["Could acknowledge feelings more explicitly"],
  "suggestions": ["Try reflecting emotions: 'That sounds really frustrating'"],
  "dimension_feedback": {
    "cognitive": "Good perspective-taking but could dig deeper",
    "emotional": "Limited emotional mirroring present",
    "behavioral": "Strong supportive language and questions",
    "contextual": "Moderate awareness of situational factors",
    "cultural": "Minimal cultural considerations evident"
  }
}
```

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

