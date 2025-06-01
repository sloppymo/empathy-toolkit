#!/usr/bin/env python3
"""
Enhanced Empathy Dashboard

An improved version of the empathy dashboard with support for tone conditioning,
mixed precision training, and other advanced features.
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import threading
import torch
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Enhanced Empathy Model Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if we can import training modules
try:
    from basic_empathy_trainer import train_empathy_model, load_jsonl_data, generate_response
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    st.warning("Training module not available. Some features will be disabled.")

# Check if we can import enhanced generator
try:
    from enhanced_empathy_generator import generate_hybrid_response
    ENHANCED_GENERATOR_AVAILABLE = True
except ImportError:
    ENHANCED_GENERATOR_AVAILABLE = False
    st.warning("Enhanced generator not available. Using basic generator instead.")

# Check if we can import data augmentation
try:
    from empathy_data_augmentation import EmpathyDataAugmenter
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    st.warning("Data augmentation module not available. Augmentation features will be disabled.")

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Sidebar
st.sidebar.title("Empathy Model Dashboard")

# GPU information
if torch.cuda.is_available():
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
    gpu_memory = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    st.sidebar.success(f"✅ {gpu_info}")
    st.sidebar.info(f"📊 {gpu_memory}")
else:
    st.sidebar.warning("⚠️ No GPU detected - using CPU")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Local Training", "Response Generation", "Data Augmentation", "Metrics", "Settings"]
)

# Settings in session state
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'use_mixed_precision': True,
        'default_tone': 'supportive',
        'model_weight': 0.7,
        'temperature': 0.7,
        'cpu_optimize': False,
        'batch_size': 2 if torch.cuda.is_available() else 1,
        'epochs': 3,
        'augmentations_per_example': 2
    }

# Training process tracking
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
    
if 'training_log' not in st.session_state:
    st.session_state.training_log = []

# Home page
if page == "Home":
    st.title("Enhanced Empathy Model System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Enhanced Empathy Model Dashboard
        
        This dashboard allows you to:
        
        - **Train** empathy models locally with mixed precision and optimizations
        - **Generate** empathetic responses with tone conditioning
        - **Augment** your training data to improve model quality
        - **Visualize** training metrics and model performance
        
        ### New Features
        
        - **Tone Conditioning**: Generate responses with specific tones (supportive, reflective, gentle, direct, encouraging)
        - **Mixed Precision Training**: Faster training with FP16/BF16 support
        - **Data Augmentation**: Expand your training dataset with intelligent variations
        - **Enhanced Error Handling**: Robust handling of problematic data entries
        
        ### Getting Started
        
        1. Go to the **Local Training** tab to train a model
        2. Use the **Response Generation** tab to test your model
        3. Explore the **Data Augmentation** tab to expand your dataset
        4. Check the **Metrics** tab to visualize performance
        """)
    
    with col2:
        st.markdown("### System Status")
        
        # Check for trained models
        models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        
        if models:
            st.success(f"✅ {len(models)} trained models available")
            for model in models:
                st.info(f"📁 {model}")
        else:
            st.warning("⚠️ No trained models found")
        
        # Check for datasets
        datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
        if datasets:
            st.success(f"✅ {len(datasets)} datasets available")
        else:
            st.error("❌ No datasets found")
        
        # Feature availability
        st.markdown("### Features")
        st.success("✅ Mixed Precision Training") if torch.cuda.is_available() else st.warning("⚠️ Mixed Precision (CPU only)")
        st.success("✅ Tone Conditioning") if ENHANCED_GENERATOR_AVAILABLE else st.warning("⚠️ Tone Conditioning (unavailable)")
        st.success("✅ Data Augmentation") if AUGMENTATION_AVAILABLE else st.warning("⚠️ Data Augmentation (unavailable)")

# Local Training page
elif page == "Local Training":
    st.title("Local Empathy Model Training")
    
    if not TRAINER_AVAILABLE:
        st.error("Training module not available. Please check your installation.")
    else:
        # Training form
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Dataset selection
                datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
                augmented_datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.jsonl')]
                all_datasets = datasets + [os.path.join("datasets", d) for d in augmented_datasets]
                
                if not all_datasets:
                    st.error("No datasets found. Please add a JSONL file to the directory.")
                    data_file = ""
                else:
                    data_file = st.selectbox("Select dataset", all_datasets)
                
                # Model name
                model_name = st.text_input("Model name", value=f"empathy_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Base model selection
                base_model = st.selectbox(
                    "Base model",
                    ["distilgpt2", "gpt2", "gpt2-medium", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
                )
            
            with col2:
                # Training parameters
                epochs = st.slider("Epochs", min_value=1, max_value=10, value=st.session_state.settings['epochs'])
                batch_size = st.slider("Batch size", min_value=1, max_value=8, value=st.session_state.settings['batch_size'])
                
                # Advanced options
                cpu_optimize = st.checkbox("Optimize for CPU", value=st.session_state.settings['cpu_optimize'])
                use_mixed_precision = st.checkbox("Use mixed precision (if available)", value=st.session_state.settings['use_mixed_precision'])
                
                # Update settings
                st.session_state.settings['epochs'] = epochs
                st.session_state.settings['batch_size'] = batch_size
                st.session_state.settings['cpu_optimize'] = cpu_optimize
                st.session_state.settings['use_mixed_precision'] = use_mixed_precision
            
            # Submit button
            submit_button = st.form_submit_button("Start Training")
        
        # Display training status
        if st.session_state.training_process is not None and st.session_state.training_process.is_alive():
            st.info("Training in progress...")
            
            # Add a stop button
            if st.button("Stop Training"):
                st.session_state.training_process.terminate()
                st.session_state.training_process = None
                st.success("Training stopped.")
        
        # Display training log
        if st.session_state.training_log:
            with st.expander("Training Log", expanded=True):
                for log in st.session_state.training_log:
                    st.text(log)
        
        # Handle form submission
        if submit_button and data_file:
            # Validate data file path
            if data_file.startswith("datasets/"):
                data_file_path = os.path.join(DATA_DIR, data_file)
            else:
                data_file_path = os.path.join(DATA_DIR, data_file)
            
            if not os.path.exists(data_file_path):
                st.error(f"Dataset file not found: {data_file_path}")
            else:
                # Clear previous log
                st.session_state.training_log = []
                
                # Set up output directory
                output_dir = os.path.join(MODELS_DIR, model_name)
                
                # Define a function to capture output
                def run_training():
                    try:
                        # Redirect stdout to capture output
                        class StdoutCapture:
                            def __init__(self):
                                self.data = []
                            
                            def write(self, text):
                                self.data.append(text)
                                st.session_state.training_log.append(text)
                                sys.__stdout__.write(text)
                            
                            def flush(self):
                                pass
                        
                        stdout_capture = StdoutCapture()
                        sys.stdout = stdout_capture
                        
                        # Run training
                        train_empathy_model(
                            data_file_path,
                            output_dir,
                            base_model=base_model,
                            epochs=epochs,
                            batch_size=batch_size,
                            cpu_optimize=cpu_optimize,
                            force_gpu=False,
                            use_mixed_precision=use_mixed_precision
                        )
                        
                        # Restore stdout
                        sys.stdout = sys.__stdout__
                        
                        # Add completion message
                        st.session_state.training_log.append("Training completed successfully!")
                    except Exception as e:
                        # Restore stdout
                        sys.stdout = sys.__stdout__
                        
                        # Add error message
                        error_msg = f"Error during training: {str(e)}"
                        st.session_state.training_log.append(error_msg)
                
                # Start training in a separate thread
                st.session_state.training_process = threading.Thread(target=run_training)
                st.session_state.training_process.daemon = True
                st.session_state.training_process.start()
                
                st.success(f"Training started! Model will be saved to: {output_dir}")
                st.info("You can navigate to other tabs while training continues.")

# Response Generation page
elif page == "Response Generation":
    st.title("Empathetic Response Generation")
    
    # Find available models
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        prompt = st.text_area("Enter a message that needs an empathetic response:", height=100)
        
        # Model selection
        if not models:
            st.warning("No trained models found. Please train a model first.")
            model_path = None
        else:
            model_name = st.selectbox("Select model", ["None"] + models)
            model_path = os.path.join(MODELS_DIR, model_name) if model_name != "None" else None
        
        # Generation options
        if ENHANCED_GENERATOR_AVAILABLE:
            tone = st.selectbox(
                "Response tone",
                ["supportive", "reflective", "gentle", "direct", "encouraging"],
                index=["supportive", "reflective", "gentle", "direct", "encouraging"].index(st.session_state.settings['default_tone'])
            )
            st.session_state.settings['default_tone'] = tone
            
            col_a, col_b = st.columns(2)
            with col_a:
                model_weight = st.slider(
                    "Model weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.settings['model_weight'],
                    step=0.1,
                    help="Weight given to model-generated responses vs. templates"
                )
                st.session_state.settings['model_weight'] = model_weight
            
            with col_b:
                temperature = st.slider(
                    "Temperature", 
                    min_value=0.1, 
                    max_value=1.5, 
                    value=st.session_state.settings['temperature'],
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic"
                )
                st.session_state.settings['temperature'] = temperature
        
        # Generate button
        if st.button("Generate Response"):
            if not prompt:
                st.error("Please enter a message.")
            else:
                with st.spinner("Generating response..."):
                    try:
                        if ENHANCED_GENERATOR_AVAILABLE and model_path:
                            # Use enhanced generator with tone conditioning
                            result = generate_hybrid_response(
                                prompt,
                                model_path=model_path,
                                tone=tone,
                                use_model_weight=model_weight,
                                temperature=temperature
                            )
                            response = result["response"]
                            metadata = {
                                "Emotion Detected": result["emotion_detected"],
                                "Tone Used": result["tone_used"],
                                "Source": result["source"]
                            }
                        elif model_path:
                            # Use basic generator
                            response = generate_response(prompt, model_path)
                            metadata = {"Source": "basic model"}
                        else:
                            # Use template only
                            if ENHANCED_GENERATOR_AVAILABLE:
                                result = generate_hybrid_response(prompt, tone=tone)
                                response = result["response"]
                                metadata = {
                                    "Emotion Detected": result["emotion_detected"],
                                    "Tone Used": result["tone_used"],
                                    "Source": "template"
                                }
                            else:
                                response = "No model or enhanced generator available."
                                metadata = {"Source": "none"}
                        
                        st.success("Response generated!")
                        st.markdown(f"### Response\n\n{response}")
                        
                        # Display metadata
                        st.json(metadata)
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    with col2:
        st.markdown("### Response Options")
        
        st.markdown("""
        **Tone Options:**
        
        - **Supportive**: Validating and comforting
        - **Reflective**: Mirroring and exploring feelings
        - **Gentle**: Soft and nurturing approach
        - **Direct**: Clear and straightforward
        - **Encouraging**: Motivating and strength-focused
        
        **Model Weight:**
        
        Controls the balance between model-generated responses and template-based responses. Higher values favor the model.
        
        **Temperature:**
        
        Controls randomness in generation. Higher values produce more varied responses, lower values are more predictable.
        """)

# Data Augmentation page
elif page == "Data Augmentation":
    st.title("Data Augmentation")
    
    if not AUGMENTATION_AVAILABLE:
        st.error("Data augmentation module not available. Please check your installation.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dataset selection
            datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
            
            if not datasets:
                st.error("No datasets found. Please add a JSONL file to the directory.")
                input_file = ""
            else:
                input_file = st.selectbox("Select input dataset", datasets)
            
            # Output file name
            output_file = st.text_input(
                "Output file name", 
                value=f"augmented_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
            )
            
            # Augmentation parameters
            augmentations = st.slider(
                "Augmentations per example", 
                min_value=1, 
                max_value=5, 
                value=st.session_state.settings['augmentations_per_example']
            )
            st.session_state.settings['augmentations_per_example'] = augmentations
            
            # Augmentation button
            if st.button("Augment Data"):
                if not input_file:
                    st.error("Please select an input dataset.")
                else:
                    with st.spinner("Augmenting data..."):
                        try:
                            # Create augmenter
                            augmenter = EmpathyDataAugmenter()
                            
                            # Set up paths
                            input_path = os.path.join(DATA_DIR, input_file)
                            output_path = os.path.join(DATASETS_DIR, output_file)
                            
                            # Run augmentation
                            original_count, augmented_count = augmenter.augment_jsonl_data(
                                input_path, 
                                output_path, 
                                augmentations_per_example=augmentations
                            )
                            
                            # Display results
                            st.success(f"Data augmentation complete!")
                            st.markdown(f"""
                            ### Results
                            
                            - Original examples: {original_count}
                            - Augmented examples: {augmented_count}
                            - Expansion factor: {augmented_count / original_count:.2f}x
                            
                            Augmented data saved to: `{output_path}`
                            """)
                            
                            # Show sample
                            with open(output_path, 'r', encoding='utf-8') as f:
                                sample_lines = [next(f) for _ in range(min(5, augmented_count))]
                            
                            with st.expander("Sample of augmented data"):
                                for i, line in enumerate(sample_lines):
                                    st.code(line, language="json")
                            
                        except Exception as e:
                            st.error(f"Error during data augmentation: {str(e)}")
        
        with col2:
            st.markdown("### Data Augmentation")
            
            st.markdown("""
            Data augmentation creates variations of your training examples to:
            
            - **Expand** small datasets
            - **Improve** model generalization
            - **Reduce** overfitting
            - **Enhance** robustness to different phrasings
            
            **Techniques Used:**
            
            - Synonym replacement
            - Random word swapping
            - Random word deletion
            
            **Emotion Preservation:**
            
            The augmentation preserves emotional words to maintain the empathetic context of each example.
            
            **Recommended Usage:**
            
            For small datasets (<100 examples), use 3-5 augmentations per example.
            For larger datasets, 1-2 augmentations is usually sufficient.
            """)

# Metrics page
elif page == "Metrics":
    st.title("Training Metrics and Analytics")
    
    # Find available models
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    
    if not models:
        st.warning("No trained models found. Please train a model first.")
    else:
        # Model selection
        model_name = st.selectbox("Select model", models)
        model_path = os.path.join(MODELS_DIR, model_name)
        
        # Check for training logs
        log_path = os.path.join(model_path, "logs")
        if not os.path.exists(log_path):
            st.warning(f"No training logs found for model: {model_name}")
        else:
            try:
                # Find log files
                log_files = [f for f in os.listdir(log_path) if f.endswith('.csv') or f.endswith('.txt')]
                
                if not log_files:
                    st.warning("No log files found.")
                else:
                    # Parse logs
                    metrics = {}
                    for log_file in log_files:
                        file_path = os.path.join(log_path, log_file)
                        
                        if log_file.endswith('.csv'):
                            # Parse CSV logs
                            try:
                                df = pd.read_csv(file_path)
                                metrics[log_file] = df
                            except Exception as e:
                                st.error(f"Error parsing {log_file}: {str(e)}")
                        
                        elif log_file.endswith('.txt'):
                            # Parse text logs
                            try:
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                
                                # Extract loss values
                                import re
                                losses = re.findall(r'loss\s*=\s*([0-9.]+)', content)
                                
                                if losses:
                                    metrics[log_file] = {
                                        'step': list(range(len(losses))),
                                        'loss': [float(loss) for loss in losses]
                                    }
                            except Exception as e:
                                st.error(f"Error parsing {log_file}: {str(e)}")
                    
                    # Display metrics
                    if metrics:
                        st.success(f"Found metrics for model: {model_name}")
                        
                        # Plot training loss
                        for name, data in metrics.items():
                            if isinstance(data, pd.DataFrame) and 'loss' in data.columns:
                                st.subheader(f"Training Loss ({name})")
                                fig = px.line(data, x='step' if 'step' in data.columns else data.index, y='loss', title="Training Loss")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif isinstance(data, dict) and 'loss' in data:
                                st.subheader(f"Training Loss ({name})")
                                fig = px.line(data, x='step', y='loss', title="Training Loss")
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No metrics data found in log files.")
            
            except Exception as e:
                st.error(f"Error processing metrics: {str(e)}")

# Settings page
elif page == "Settings":
    st.title("Settings")
    
    with st.form("settings_form"):
        st.subheader("Training Settings")
        
        cpu_optimize = st.checkbox("Optimize for CPU", value=st.session_state.settings['cpu_optimize'])
        use_mixed_precision = st.checkbox("Use mixed precision (if available)", value=st.session_state.settings['use_mixed_precision'])
        batch_size = st.slider("Default batch size", min_value=1, max_value=8, value=st.session_state.settings['batch_size'])
        epochs = st.slider("Default epochs", min_value=1, max_value=10, value=st.session_state.settings['epochs'])
        
        st.subheader("Generation Settings")
        
        default_tone = st.selectbox(
            "Default response tone",
            ["supportive", "reflective", "gentle", "direct", "encouraging"],
            index=["supportive", "reflective", "gentle", "direct", "encouraging"].index(st.session_state.settings['default_tone'])
        )
        
        model_weight = st.slider(
            "Default model weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.settings['model_weight'],
            step=0.1
        )
        
        temperature = st.slider(
            "Default temperature", 
            min_value=0.1, 
            max_value=1.5, 
            value=st.session_state.settings['temperature'],
            step=0.1
        )
        
        st.subheader("Data Augmentation Settings")
        
        augmentations_per_example = st.slider(
            "Default augmentations per example", 
            min_value=1, 
            max_value=5, 
            value=st.session_state.settings['augmentations_per_example']
        )
        
        # Save button
        save_button = st.form_submit_button("Save Settings")
    
    if save_button:
        # Update settings
        st.session_state.settings.update({
            'cpu_optimize': cpu_optimize,
            'use_mixed_precision': use_mixed_precision,
            'batch_size': batch_size,
            'epochs': epochs,
            'default_tone': default_tone,
            'model_weight': model_weight,
            'temperature': temperature,
            'augmentations_per_example': augmentations_per_example
        })
        
        st.success("Settings saved successfully!")
        
        # Display current settings
        st.json(st.session_state.settings)

# Footer
st.markdown("---")
st.markdown("Enhanced Empathy Model System | Developed with ❤️ and Streamlit")
