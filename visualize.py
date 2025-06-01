#!/usr/bin/env python
"""
Empathy Score Visualization Tool

This module provides visualization capabilities for empathy scores using Matplotlib.
It can create radar charts, bar charts, and heatmaps for comparing scores.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib

# Use Agg backend for environments without a display
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

def create_radar_chart(scores, labels=None, title="Empathy Score Comparison", output_file=None):
    """Create a radar chart (spider plot) comparing multiple empathy scores
    
    Args:
        scores (list): List of dictionaries with empathy dimension scores
        labels (list): List of labels for each score set
        title (str): Chart title
        output_file (str): Output file path (if None, display instead)
    """
    # Get all dimension names from the first score set
    if not scores:
        print("No scores provided")
        return
    
    # Extract dimensions (excluding total)
    dimensions = [key for key in scores[0].keys() if key != 'total']
    dim_count = len(dimensions)
    
    # Default labels if not provided
    if not labels:
        labels = [f"Response {i+1}" for i in range(len(scores))]
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each dimension
    angles = np.linspace(0, 2*np.pi, dim_count, endpoint=False).tolist()
    # Close the loop
    angles += angles[:1]
    
    # Set up the grid
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Set labels for each dimension
    plt.xticks(angles[:-1], dimensions)
    
    # Set y-axis limits
    ax.set_ylim(0, 10)
    plt.yticks(range(0, 11, 2), [str(i) for i in range(0, 11, 2)])
    
    # Plot each score set
    for i, score in enumerate(scores):
        values = [score[dim] for dim in dimensions]
        # Close the loop
        values += values[:1]
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=labels[i])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    # Save or display
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to {output_file}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def create_bar_chart(scores, labels=None, title="Empathy Score Comparison", output_file=None):
    """Create a bar chart comparing empathy scores
    
    Args:
        scores (list): List of dictionaries with empathy dimension scores
        labels (list): List of labels for each score set
        title (str): Chart title
        output_file (str): Output file path (if None, display instead)
    """
    if not scores:
        print("No scores provided")
        return
    
    # Extract dimensions (excluding total)
    dimensions = [key for key in scores[0].keys() if key != 'total']
    
    # Default labels if not provided
    if not labels:
        labels = [f"Response {i+1}" for i in range(len(scores))]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set bar width and positions
    bar_width = 0.15
    x = np.arange(len(dimensions))
    
    # Plot bars for each score set
    for i, score in enumerate(scores):
        values = [score[dim] for dim in dimensions]
        position = x + i * bar_width
        ax.bar(position, values, bar_width, label=labels[i])
    
    # Add labels and title
    ax.set_xlabel('Empathy Dimensions')
    ax.set_ylabel('Score (0-10)')
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(scores) - 1) / 2)
    ax.set_xticklabels(dimensions)
    ax.legend()
    
    # Set y-axis limits
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(MaxNLocator(11))
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save or display
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {output_file}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def create_heatmap(df, title="Empathy Scores Heatmap", output_file=None):
    """Create a heatmap from a dataframe of empathy scores
    
    Args:
        df (DataFrame): DataFrame with empathy scores
        title (str): Chart title
        output_file (str): Output file path (if None, display instead)
    """
    # Check if dataframe contains empathy score columns
    empathy_cols = [col for col in df.columns if col.startswith('empathy_')]
    if not empathy_cols:
        print("No empathy score columns found in dataframe")
        return
    
    # Select only empathy columns for the heatmap
    score_df = df[empathy_cols].copy()
    
    # Create a more readable column mapping for display
    col_map = {
        'empathy_total': 'Total',
        'empathy_cognitive': 'Cognitive',
        'empathy_emotional': 'Emotional',
        'empathy_behavioral': 'Behavioral',
        'empathy_contextual': 'Contextual',
        'empathy_cultural': 'Cultural'
    }
    
    # Rename columns for display
    score_df = score_df.rename(columns=col_map)
    
    # Set up the figure
    plt.figure(figsize=(12, max(8, len(score_df) * 0.4)))
    
    # Create a custom colormap: low (red) to high (green)
    cmap = LinearSegmentedColormap.from_list('empathy_cmap', ['#ff6666', '#ffcc66', '#66cc66'])
    
    # Create the heatmap
    ax = plt.imshow(score_df.values, cmap=cmap, aspect='auto', vmin=0, vmax=10)
    
    # Add a colorbar
    cbar = plt.colorbar(ax)
    cbar.set_label('Empathy Score (0-10)')
    
    # Add axis labels and title
    plt.yticks(range(len(score_df)), range(1, len(score_df) + 1))
    plt.ylabel('Response Index')
    plt.xticks(range(len(score_df.columns)), score_df.columns, rotation=45, ha='right')
    plt.title(title)
    
    # Add the score values to the cells
    for i in range(len(score_df)):
        for j in range(len(score_df.columns)):
            value = score_df.iloc[i, j]
            text_color = 'white' if value < 5 else 'black'
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
    
    # Save or display
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_file}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def visualize_from_csv(input_file, chart_type="radar", output_file=None):
    """Create visualization from a CSV file with empathy scores
    
    Args:
        input_file (str): Path to the CSV file
        chart_type (str): Type of chart to create (radar, bar, heatmap)
        output_file (str): Path to save the output chart
    """
    try:
        df = pd.read_csv(input_file)
        
        # Check for empathy score columns
        empathy_cols = [col for col in df.columns if col.startswith('empathy_')]
        if not empathy_cols:
            print(f"No empathy score columns found in {input_file}")
            return
        
        # Extract scores from the dataframe
        scores = []
        labels = []
        
        for i, row in df.iterrows():
            # Create a score dictionary
            score = {}
            for col in empathy_cols:
                # Strip 'empathy_' prefix for visualization
                dim = col.replace('empathy_', '')
                score[dim] = row[col]
            
            scores.append(score)
            
            # Use a descriptive column if available, otherwise use index
            if 'id' in df.columns:
                label = f"Response {row['id']}"
            elif 'response' in df.columns:
                text = row['response']
                label = text[:20] + "..." if len(text) > 20 else text
            else:
                label = f"Response {i+1}"
                
            labels.append(label)
        
        # Create the visualization
        if chart_type == "radar":
            create_radar_chart(scores, labels, f"Empathy Scores from {os.path.basename(input_file)}", output_file)
        elif chart_type == "bar":
            create_bar_chart(scores, labels, f"Empathy Scores from {os.path.basename(input_file)}", output_file)
        elif chart_type == "heatmap":
            create_heatmap(df, f"Empathy Scores from {os.path.basename(input_file)}", output_file)
        else:
            print(f"Unknown chart type: {chart_type}")
    
    except Exception as e:
        print(f"Error visualizing from {input_file}: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Empathy Score Visualization Tool")
    
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with empathy scores")
    parser.add_argument("--type", "-t", choices=["radar", "bar", "heatmap"], default="radar", 
                        help="Type of visualization to create")
    parser.add_argument("--output", "-o", help="Output file path (PNG, JPG, PDF, SVG)")
    
    args = parser.parse_args()
    
    visualize_from_csv(args.input, args.type, args.output)

if __name__ == "__main__":
    main()
