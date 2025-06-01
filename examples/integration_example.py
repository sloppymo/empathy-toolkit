#!/usr/bin/env python3
"""
Empathy Toolkit Integration Example

This script demonstrates how to integrate the empathy scoring framework
with other components of the empathy toolkit.
"""

import os
import sys
import asyncio
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import rich.box

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import the toolkit API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified API
from empathy_toolkit_api import EmpathyToolkit

# Initialize Rich console for output
console = Console()

async def demo_response_evaluation():
    """Demonstrate integrating empathy scoring with response evaluation"""
    console.print("[bold green]Demonstrating Response Evaluation Integration[/bold green]")
    
    # Create a sample conversation
    scenario = {
        "patient_context": "I just received a diagnosis of Type 2 diabetes and I'm feeling overwhelmed.",
        "provider_responses": [
            "You'll need to monitor your blood sugar levels and change your diet. Here's a pamphlet explaining everything.",
            "I understand this diagnosis can feel overwhelming. Many patients feel the same way initially. Let's talk about what this means for you and how we can manage it together. What questions do you have right now?"
        ]
    }
    
    # Initialize the toolkit
    toolkit = EmpathyToolkit()
    
    # Create a table for comparison
    table = Table(title="Empathy Score Comparison")
    table.add_column("Response", style="cyan")
    table.add_column("Cognitive", justify="center")
    table.add_column("Emotional", justify="center")
    table.add_column("Behavioral", justify="center")
    table.add_column("Contextual", justify="center")
    table.add_column("Cultural", justify="center")
    table.add_column("Total", justify="center", style="bold")
    
    # Analyze each response
    for i, response in enumerate(scenario["provider_responses"]):
        console.print(f"[bold]Analyzing response {i+1}...[/bold]")
        
        # Score empathy
        score, feedback = await toolkit.score_empathy(
            response=response,
            context=scenario["patient_context"]
        )
        
        # Add to table
        table.add_row(
            response[:50] + "..." if len(response) > 50 else response,
            f"{score.cognitive:.1f}",
            f"{score.emotional:.1f}",
            f"{score.behavioral:.1f}",
            f"{score.contextual:.1f}",
            f"{score.cultural:.1f}",
            f"{score.total:.1f}"
        )
        
        # Print feedback for the response
        console.print(f"\n[bold]Response {i+1} Feedback:[/bold]")
        console.print(f"[italic]{response}[/italic]\n")
        
        if feedback:
            if feedback.strengths:
                console.print("[bold green]Strengths:[/bold green]")
                for strength in feedback.strengths:
                    console.print(f"+ {strength}")
            
            if feedback.areas_for_improvement:
                console.print("\n[bold yellow]Areas for Improvement:[/bold yellow]")
                for area in feedback.areas_for_improvement:
                    console.print(f"- {area}")
        
        console.print("")
    
    # Display the comparison table
    console.print(table)

async def demo_dataset_enhancement():
    """Demonstrate batch scoring of multiple responses"""
    console.print("\n[bold green]Demonstrating Batch Empathy Scoring[/bold green]")
    
    # Create sample data for batch scoring
    console.print("Creating sample data...")
    sample_items = [
        {
            "id": "1",
            "context": "Patient: I'm feeling anxious about this procedure.",
            "response": "This is a routine procedure with minimal risk."
        },
        {
            "id": "2",
            "context": "Client: I'm not sure I can afford this treatment.",
            "response": "I understand your concern about the cost. Let's discuss some options that might make this more affordable for you."
        },
        {
            "id": "3",
            "context": "Friend: I just got some bad news about my health.",
            "response": "That sounds tough. Let me know if you want to talk about it."
        }
    ]
    
    # Initialize the toolkit
    toolkit = EmpathyToolkit()
    
    # Score responses in batch
    console.print("Scoring responses in batch...")
    results = await toolkit.batch_score_empathy(sample_items)
    
    # Display the results
    console.print("\n[bold]Batch Scoring Results:[/bold]")
    
    # Create a simple, narrow table that will fit in the terminal
    table = Table(title="Empathy Scores by Response", box=rich.box.SIMPLE)
    
    # Use only essential columns
    table.add_column("ID", style="dim", width=3, justify="center")
    table.add_column("Context", style="cyan", width=20)
    table.add_column("Response", style="green", width=20)
    table.add_column("Total", style="bold white", width=5, justify="center")
    
    # Add rows
    for result in results:
        # Truncate text to fit column width
        context = result["context"][:17] + "..." if len(result["context"]) > 17 else result["context"]
        response = result["response"][:17] + "..." if len(result["response"]) > 17 else result["response"]
        
        table.add_row(
            result["id"],
            context,
            response,
            f"{result['score'].total:.1f}"
        )
    
    console.print(table)

async def main():
    """Main function to run all demos"""
    console.print("[bold]===== Empathy Toolkit Integration Examples =====[/bold]\n")
    
    # Demo 1: Response Evaluation
    await demo_response_evaluation()
    
    # Demo 2: Dataset Enhancement
    await demo_dataset_enhancement()
    
    console.print("\n[bold]===== Integration Examples Complete =====[/bold]")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
