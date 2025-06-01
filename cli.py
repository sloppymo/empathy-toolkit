#!/usr/bin/env python
"""
Empathy Toolkit CLI - Command line interface for the empathy scoring toolkit
"""
import os
import sys
import asyncio
import argparse
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
import rich.box
from dotenv import load_dotenv

from empathy_toolkit_api import EmpathyToolkit

# Load environment variables from .env file
load_dotenv()

console = Console()

async def score_response(args):
    """Score a single response"""
    toolkit = EmpathyToolkit()
    
    # If input is provided via stdin, read it
    if not sys.stdin.isatty():
        console.print("[dim]Reading from stdin...[/dim]")
        args.response = sys.stdin.read().strip()
    
    # Validate input
    if not args.response:
        console.print("[bold red]Error:[/bold red] No response text provided. Use --response or pipe text.", style="red")
        return
    
    # Score the response
    score, feedback = await toolkit.score_empathy(
        response=args.response,
        context=args.context,
        include_feedback=not args.no_feedback
    )
    
    # Display results based on format
    if args.format == "json":
        result = {
            "score": score.to_dict(),
            "feedback": feedback.to_dict() if feedback else None
        }
        print(json.dumps(result, indent=2))
    else:
        # Display score in a table
        table = Table(title="Empathy Score", box=rich.box.ROUNDED)
        
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="center", style="green")
        
        table.add_row("Cognitive", f"{score.cognitive:.1f}")
        table.add_row("Emotional", f"{score.emotional:.1f}")
        table.add_row("Behavioral", f"{score.behavioral:.1f}")
        table.add_row("Contextual", f"{score.contextual:.1f}")
        table.add_row("Cultural", f"{score.cultural:.1f}")
        table.add_row("Total", f"{score.total:.1f}", style="bold")
        
        console.print(table)
        
        # Display feedback if available
        if feedback:
            console.print("\n[bold]Feedback:[/bold]")
            console.print(f"[cyan]{args.response}[/cyan]\n")
            
            if feedback.strengths:
                console.print("[bold green]Strengths:[/bold green]")
                for strength in feedback.strengths:
                    console.print(f"+ {strength}")
                console.print("")
            
            if feedback.areas_for_improvement:
                console.print("[bold yellow]Areas for Improvement:[/bold yellow]")
                for area in feedback.areas_for_improvement:
                    console.print(f"- {area}")
                console.print("")
            
            if feedback.suggestions:
                console.print("[bold blue]Suggestions:[/bold blue]")
                for suggestion in feedback.suggestions:
                    console.print(f"> {suggestion}")

async def enhance_dataset(args):
    """Enhance a dataset with empathy scores"""
    toolkit = EmpathyToolkit()
    
    # Validate input
    if not args.input:
        console.print("[bold red]Error:[/bold red] No input file provided.", style="red")
        return
    
    if not os.path.exists(args.input):
        console.print(f"[bold red]Error:[/bold red] Input file '{args.input}' not found.", style="red")
        return
    
    if not args.output:
        # Use the input filename with _enhanced suffix
        base_name, ext = os.path.splitext(args.input)
        args.output = f"{base_name}_enhanced{ext}"
    
    console.print(f"[bold]Enhancing dataset:[/bold] {args.input}")
    console.print(f"[bold]Output file:[/bold] {args.output}")
    console.print(f"[bold]Response column:[/bold] {args.response_column}")
    if args.context_column:
        console.print(f"[bold]Context column:[/bold] {args.context_column}")
    
    # Process the dataset
    try:
        success = await toolkit.enhance_dataset_with_scores(
            dataset_path=args.input,
            output_path=args.output,
            response_column=args.response_column,
            context_column=args.context_column
        )
        
        if success:
            console.print(f"[bold green]Dataset successfully enhanced and saved to:[/bold green] {args.output}")
            
            # Show a preview of the results
            if args.input.endswith('.csv'):
                df = pd.read_csv(args.output)
            elif args.input.endswith('.jsonl'):
                import jsonlines
                with jsonlines.open(args.output) as reader:
                    df = pd.DataFrame([item for item in reader])
            
            console.print("\n[bold]Preview of enhanced dataset:[/bold]")
            preview = df.head(5)
            
            # Show preview in a table
            table = Table(box=rich.box.SIMPLE)
            
            # Add columns
            for col in preview.columns:
                if col.startswith('empathy_'):
                    table.add_column(col, justify="center", style="green", width=8)
                else:
                    table.add_column(col, width=15)
            
            # Add rows
            for _, row in preview.iterrows():
                table.add_row(*[str(row[col])[:12] + "..." if isinstance(row[col], str) and len(str(row[col])) > 12 else str(row[col]) for col in preview.columns])
            
            console.print(table)
        else:
            console.print("[bold red]Failed to enhance dataset.[/bold red]", style="red")
    except Exception as e:
        console.print(f"[bold red]Error enhancing dataset:[/bold red] {str(e)}", style="red")

def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(description="Empathy Toolkit Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score a response for empathy")
    score_parser.add_argument("--response", "-r", help="The response text to score")
    score_parser.add_argument("--context", "-c", default="", help="Optional context for the response")
    score_parser.add_argument("--no-feedback", action="store_true", help="Skip detailed feedback")
    score_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    # Enhance dataset command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance a dataset with empathy scores")
    enhance_parser.add_argument("--input", "-i", required=True, help="Input dataset file (CSV or JSONL)")
    enhance_parser.add_argument("--output", "-o", help="Output dataset file (defaults to input_enhanced.ext)")
    enhance_parser.add_argument("--response-column", "-r", required=True, help="Column containing responses to score")
    enhance_parser.add_argument("--context-column", "-c", help="Optional column containing context for responses")
    
    # Dimensions command to show dimension descriptions
    dimensions_parser = subparsers.add_parser("dimensions", help="Show empathy dimension descriptions")
    
    args = parser.parse_args()
    
    if args.command == "score":
        asyncio.run(score_response(args))
    elif args.command == "enhance":
        asyncio.run(enhance_dataset(args))
    elif args.command == "dimensions":
        toolkit = EmpathyToolkit()
        console.print("[bold]Empathy Dimension Descriptions:[/bold]")
        for dimension, description in toolkit.get_dimension_descriptions().items():
            console.print(f"[bold cyan]{dimension}:[/bold cyan] {description}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
