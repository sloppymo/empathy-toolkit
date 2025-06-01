#!/usr/bin/env python3
"""
Empathy Scorer Demo Application

This script provides a command-line interface for demonstrating the 
multi-dimensional empathy scoring framework.
"""

import os
import sys
import asyncio
import argparse
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.markdown import Markdown

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.empathy_scorer import (
    EmpathyScoringSystem, 
    EmpathyScore, 
    EmpathyFeedback, 
    BatchEmpathyAnalyzer
)

console = Console()

# Example scenarios to analyze
EXAMPLE_SCENARIOS = [
    {
        "name": "Medical Diagnosis",
        "context": "Patient: I've been feeling overwhelmed with my new diagnosis. I don't know how I'll manage everything.",
        "responses": {
            "high_empathy": "I understand this is a difficult time for you. A new diagnosis can feel overwhelming, and it's completely normal to worry about how you'll cope. Many patients feel similarly when facing new health challenges. What specific aspects are you most concerned about? We can break this down into manageable steps and explore resources that might help you.",
            "medium_empathy": "That's tough to hear. Many people struggle with new diagnoses. We have some resources that might help you manage things better. What questions do you have?",
            "low_empathy": "The treatment plan is straightforward. Just follow the instructions and you'll be fine. Most patients handle this without any problems."
        }
    },
    {
        "name": "Job Loss",
        "context": "Client: I just lost my job after 12 years. I'm scared about how I'll support my family and what this means for my career.",
        "responses": {
            "high_empathy": "Losing a job after 12 years is incredibly destabilizing, especially with a family counting on you. It's natural to feel both the practical worry about finances and the deeper questions about your professional identity. This kind of unexpected change can shake your sense of security. Would it help to talk through both your immediate financial concerns and your thoughts about your career path moving forward?",
            "medium_empathy": "Losing a job is difficult. You should update your resume and start applying right away. The job market is pretty good right now, so you'll probably find something soon.",
            "low_empathy": "At least unemployment benefits exist. You should focus on the positive - now you can find a better job that pays more!"
        }
    },
    {
        "name": "Grief",
        "context": "Friend: My mother passed away last week. I thought I was doing okay, but today I just can't seem to function. I feel like I'm letting everyone down.",
        "responses": {
            "high_empathy": "I'm so sorry about your mother. Grief comes in waves, and it makes complete sense that you'd have days where functioning feels impossible. You're not letting anyone down - this is part of the natural process of grieving someone you love deeply. Is there anything I can help take off your plate today? And please know that whatever you're feeling right now is valid.",
            "medium_empathy": "Sorry for your loss. Grief is hard. You should try to take it one day at a time. Maybe try some self-care activities.",
            "low_empathy": "You need to stay strong for your family. Everyone loses their parents eventually. Try to keep busy so you don't think about it too much."
        }
    }
]


async def compare_responses(
    scenario: Dict,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    feedback_model: str = "gpt-3.5-turbo"
) -> None:
    """
    Compare different responses to the same scenario
    
    Args:
        scenario: Dictionary containing context and different response options
        api_key: OpenAI API key (optional)
        model: Model to use for scoring
        feedback_model: Model to use for feedback
    """
    with console.status(f"[bold green]Analyzing responses for '{scenario['name']}' scenario..."):
        # Initialize scoring system
        scorer = EmpathyScoringSystem(
            primary_model=model,
            feedback_model=feedback_model,
            api_key=api_key
        )
        
        # Collect results
        results = []
        for level, response in scenario["responses"].items():
            score, feedback = await scorer.score_response(response, scenario["context"])
            results.append((level, response, score, feedback))
    
    # Display context
    console.print(Panel(
        f"[bold]Context:[/bold]\n{scenario['context']}", 
        title=f"Scenario: {scenario['name']}", 
        expand=False
    ))
    
    # Display scores in a table
    table = Table(title="Empathy Score Comparison")
    table.add_column("Response Type", style="cyan")
    table.add_column("Cognitive", justify="center")
    table.add_column("Emotional", justify="center")
    table.add_column("Behavioral", justify="center")
    table.add_column("Contextual", justify="center")
    table.add_column("Cultural", justify="center")
    table.add_column("Total", justify="center", style="bold")
    
    # Color mapping for scores
    def score_color(score):
        if score >= 8:
            return "[green]"
        elif score >= 5:
            return "[yellow]"
        else:
            return "[red]"
    
    # Add rows to table
    for level, _, score, _ in results:
        table.add_row(
            level.replace("_", " ").title(),
            f"{score_color(score.cognitive)}{score.cognitive:.1f}[/]",
            f"{score_color(score.emotional)}{score.emotional:.1f}[/]",
            f"{score_color(score.behavioral)}{score.behavioral:.1f}[/]",
            f"{score_color(score.contextual)}{score.contextual:.1f}[/]",
            f"{score_color(score.cultural)}{score.cultural:.1f}[/]",
            f"{score_color(score.total)}{score.total:.1f}[/]"
        )
    
    console.print(table)
    
    # Display detailed analysis of each response
    for level, response, score, feedback in results:
        console.print(f"\n[bold]{level.replace('_', ' ').title()} Empathy Response:[/bold]")
        console.print(f"[italic]{response}[/italic]")
        
        if feedback:
            # Display strengths
            if feedback.strengths:
                console.print("\n[bold green]Strengths:[/bold green]")
                for strength in feedback.strengths:
                    console.print(f"+ {strength}")
            
            # Display areas for improvement
            if feedback.areas_for_improvement:
                console.print("\n[bold yellow]Areas for Improvement:[/bold yellow]")
                for area in feedback.areas_for_improvement:
                    console.print(f"- {area}")
            
            # Display suggestions
            if feedback.suggestions:
                console.print("\n[bold blue]Suggestions:[/bold blue]")
                for suggestion in feedback.suggestions:
                    console.print(f"> {suggestion}")
        
        console.print("\n" + "-" * 80)


async def analyze_custom_response(
    context: str, 
    response: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    feedback_model: str = "gpt-3.5-turbo"
) -> None:
    """
    Analyze a custom response
    
    Args:
        context: Context of the conversation
        response: Response to analyze
        api_key: OpenAI API key (optional)
        model: Model to use for scoring
        feedback_model: Model to use for feedback
    """
    with console.status("[bold green]Analyzing your response..."):
        # Initialize scoring system
        scorer = EmpathyScoringSystem(
            primary_model=model,
            feedback_model=feedback_model,
            api_key=api_key
        )
        
        # Score the response
        score, feedback = await scorer.score_response(response, context)
    
    # Display context
    console.print(Panel(
        f"[bold]Context:[/bold]\n{context}", 
        title="Your Scenario", 
        expand=False
    ))
    
    # Display response
    console.print(Panel(
        f"{response}", 
        title="Your Response", 
        expand=False
    ))
    
    # Display scores
    table = Table(title="Empathy Score")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score (0-10)", justify="center")
    table.add_column("Interpretation", style="green")
    
    # Define interpretation bands
    def get_interpretation(score):
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Moderate"
        else:
            return "Needs improvement"
    
    # Add rows to table
    dimensions = [
        ("Cognitive", score.cognitive, "Understanding others' perspectives"),
        ("Emotional", score.emotional, "Resonating with feelings"),
        ("Behavioral", score.behavioral, "Expressing support through actions/words"),
        ("Contextual", score.contextual, "Recognizing situational factors"),
        ("Cultural", score.cultural, "Cultural sensitivity"),
        ("Total", score.total, "Overall empathy score")
    ]
    
    for dim, score_val, desc in dimensions:
        interp = get_interpretation(score_val)
        table.add_row(
            f"{dim} Empathy",
            f"{score_val:.1f}",
            f"{interp} - {desc}"
        )
    
    console.print(table)
    
    # Display feedback
    if feedback:
        # Strengths
        if feedback.strengths:
            console.print(Panel(
                "\n".join([f"✓ {strength}" for strength in feedback.strengths]),
                title="[bold green]Strengths[/bold green]",
                expand=False
            ))
        
        # Areas for improvement
        if feedback.areas_for_improvement:
            console.print(Panel(
                "\n".join([f"! {area}" for area in feedback.areas_for_improvement]),
                title="[bold yellow]Areas for Improvement[/bold yellow]",
                expand=False
            ))
        
        # Suggestions
        if feedback.suggestions:
            console.print(Panel(
                "\n".join([f"→ {suggestion}" for suggestion in feedback.suggestions]),
                title="[bold blue]Suggestions[/bold blue]",
                expand=False
            ))
        
        # Dimension feedback
        for dim, fb in feedback.dimension_feedback.items():
            console.print(f"[bold]{dim.capitalize()} Dimension:[/bold] {fb}")


async def analyze_dataset(
    file_path: str,
    response_col: str,
    context_col: Optional[str] = None,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> None:
    """
    Analyze empathy in a dataset
    
    Args:
        file_path: Path to CSV or JSONL file
        response_col: Column name containing responses to evaluate
        context_col: Column name with context (optional)
        output_path: Path to save results (optional)
        api_key: OpenAI API key (optional)
        model: Model to use for scoring
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            return
        
        # Initialize scoring system
        scorer = EmpathyScoringSystem(
            primary_model=model,
            api_key=api_key
        )
        
        # Initialize analyzer
        analyzer = BatchEmpathyAnalyzer(scorer)
        
        with console.status(f"[bold green]Analyzing dataset: {file_path}..."):
            # Analyze dataset
            results = await analyzer.analyze_dataset(
                file_path, 
                response_col, 
                context_col, 
                output_path
            )
            
            # Generate insights
            insights = analyzer.generate_insights(results)
        
        # Display results summary
        console.print(Panel(
            f"Analyzed [bold]{len(results)}[/bold] responses",
            title="Dataset Analysis Complete",
            expand=False
        ))
        
        # Display insights
        table = Table(title="Empathy Dimension Statistics")
        table.add_column("Dimension", style="cyan")
        table.add_column("Mean", justify="center")
        table.add_column("Median", justify="center")
        
        for dim in ['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural', 'total']:
            table.add_row(
                dim.capitalize(),
                f"{insights['mean_scores'][dim]:.2f}",
                f"{insights['median_scores'][dim]:.2f}"
            )
        
        console.print(table)
        
        # Display high-level insights
        console.print(Panel(
            f"[bold]Strongest dimension:[/bold] {insights['strongest_dimension'].capitalize()}\n"
            f"[bold]Weakest dimension:[/bold] {insights['weakest_dimension'].capitalize()}\n\n"
            f"[bold]Empathy level distribution:[/bold]\n"
            f"- High empathy responses: {insights['high_empathy_count']}\n"
            f"- Medium empathy responses: {insights['medium_empathy_count']}\n"
            f"- Low empathy responses: {insights['low_empathy_count']}\n\n"
            f"[bold]Balance score:[/bold] {insights['balance_score']:.2f} "
            f"(lower means more balanced across dimensions)",
            title="Key Insights",
            expand=False
        ))
        
        if output_path:
            console.print(f"[bold green]Results saved to:[/bold green] {output_path}")
    
    except Exception as e:
        console.print(f"[bold red]Error during analysis:[/bold red] {str(e)}")


def display_examples() -> None:
    """Display the available example scenarios"""
    console.print("[bold]Available Example Scenarios:[/bold]")
    for i, scenario in enumerate(EXAMPLE_SCENARIOS, 1):
        console.print(f"{i}. {scenario['name']}")
        console.print(f"   Context: {scenario['context'][:70]}...")


async def main():
    parser = argparse.ArgumentParser(description="Empathy Scoring Framework Demo")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compare example responses
    compare_parser = subparsers.add_parser("compare", help="Compare example responses")
    compare_parser.add_argument(
        "--scenario", type=int, choices=range(1, len(EXAMPLE_SCENARIOS)+1),
        help="Scenario number to analyze"
    )
    compare_parser.add_argument(
        "--all", action="store_true", help="Analyze all scenarios"
    )
    compare_parser.add_argument(
        "--api-key", type=str, help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    compare_parser.add_argument(
        "--model", type=str, default="gpt-4o", 
        help="Model to use for scoring (default: gpt-4o)"
    )
    compare_parser.add_argument(
        "--feedback-model", type=str, default="gpt-3.5-turbo",
        help="Model to use for feedback (default: gpt-3.5-turbo)"
    )
    
    # Analyze custom response
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a custom response")
    analyze_parser.add_argument("--context", type=str, required=True, help="Context of the conversation")
    analyze_parser.add_argument("--response", type=str, required=True, help="Response to analyze")
    analyze_parser.add_argument(
        "--api-key", type=str, help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    analyze_parser.add_argument(
        "--model", type=str, default="gpt-4o", 
        help="Model to use for scoring (default: gpt-4o)"
    )
    analyze_parser.add_argument(
        "--feedback-model", type=str, default="gpt-3.5-turbo",
        help="Model to use for feedback (default: gpt-3.5-turbo)"
    )
    
    # Analyze dataset
    dataset_parser = subparsers.add_parser("dataset", help="Analyze empathy in a dataset")
    dataset_parser.add_argument("--file", type=str, required=True, help="Path to CSV or JSONL file")
    dataset_parser.add_argument("--response-col", type=str, required=True, help="Column with responses")
    dataset_parser.add_argument("--context-col", type=str, help="Column with context (optional)")
    dataset_parser.add_argument("--output", type=str, help="Path to save results (optional)")
    dataset_parser.add_argument(
        "--api-key", type=str, help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    dataset_parser.add_argument(
        "--model", type=str, default="gpt-4o", 
        help="Model to use for scoring (default: gpt-4o)"
    )
    
    # List examples
    subparsers.add_parser("examples", help="List available example scenarios")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        console.print("[bold yellow]Warning:[/bold yellow] No API key provided. Set OPENAI_API_KEY environment variable or use --api-key.")
    
    if args.command == "compare":
        if args.all:
            for scenario in EXAMPLE_SCENARIOS:
                await compare_responses(
                    scenario, 
                    api_key=api_key, 
                    model=args.model,
                    feedback_model=args.feedback_model
                )
                console.print("\n")
        elif args.scenario:
            scenario = EXAMPLE_SCENARIOS[args.scenario - 1]
            await compare_responses(
                scenario, 
                api_key=api_key, 
                model=args.model,
                feedback_model=args.feedback_model
            )
        else:
            display_examples()
            console.print("\n[bold yellow]Please specify a scenario with --scenario or use --all to analyze all scenarios[/bold yellow]")
    
    elif args.command == "analyze":
        await analyze_custom_response(
            args.context, 
            args.response, 
            api_key=api_key,
            model=args.model,
            feedback_model=args.feedback_model
        )
    
    elif args.command == "dataset":
        await analyze_dataset(
            args.file, 
            args.response_col, 
            args.context_col, 
            args.output, 
            api_key=api_key,
            model=args.model
        )
    
    elif args.command == "examples":
        display_examples()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("[bold red]Operation cancelled by user[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
