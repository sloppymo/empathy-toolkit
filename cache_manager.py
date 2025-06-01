#!/usr/bin/env python
"""
Cache Manager for Empathy Toolkit

This utility helps manage the cache used by the empathy toolkit to store API responses.
It provides functions to clear, view, and manage the cache to optimize API usage costs.
"""

import os
import json
import shutil
import argparse
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
import rich.box

console = Console()

def get_cache_size(cache_dir):
    """Get the size of the cache directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def clear_cache(cache_dir, age_days=None):
    """Clear the cache or remove entries older than specified days"""
    if not os.path.exists(cache_dir):
        console.print(f"[yellow]Cache directory {cache_dir} does not exist.[/yellow]")
        return
    
    if age_days is not None:
        # Only remove files older than age_days
        cutoff_time = datetime.now() - timedelta(days=age_days)
        count = 0
        total_size = 0
        
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff_time:
                    file_size = os.path.getsize(filepath)
                    total_size += file_size
                    os.remove(filepath)
                    count += 1
        
        console.print(f"[green]Removed {count} cache files older than {age_days} days ({format_size(total_size)}).[/green]")
    else:
        # Clear the entire cache
        cache_size = get_cache_size(cache_dir)
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        console.print(f"[green]Cleared cache directory ({format_size(cache_size)}).[/green]")

def view_cache(cache_dir, details=False):
    """View cache statistics and optionally details of cached items"""
    if not os.path.exists(cache_dir):
        console.print(f"[yellow]Cache directory {cache_dir} does not exist.[/yellow]")
        return
    
    # Get basic stats
    file_count = 0
    total_size = 0
    oldest_time = datetime.now()
    newest_time = datetime.fromtimestamp(0)
    cache_files = []
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            # Track stats
            file_count += 1
            total_size += file_size
            oldest_time = min(oldest_time, file_time)
            newest_time = max(newest_time, file_time)
            
            # Store file details if needed
            if details:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract a snippet of content to help identify the cache entry
                        snippet = ""
                        if 'args' in data and isinstance(data['args'], list) and len(data['args']) > 0:
                            arg = data['args'][0]
                            if isinstance(arg, str):
                                snippet = arg[:50] + "..." if len(arg) > 50 else arg
                        
                        cache_files.append({
                            'filename': filename,
                            'size': file_size,
                            'time': file_time,
                            'snippet': snippet
                        })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Skip files that aren't valid JSON
                    cache_files.append({
                        'filename': filename,
                        'size': file_size,
                        'time': file_time,
                        'snippet': "[Invalid JSON]"
                    })
    
    # Display basic stats
    console.print(f"[bold]Cache Statistics:[/bold]")
    console.print(f"Cache directory: {os.path.abspath(cache_dir)}")
    console.print(f"Total files: {file_count}")
    console.print(f"Total size: {format_size(total_size)}")
    if file_count > 0:
        console.print(f"Oldest entry: {oldest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Newest entry: {newest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Age range: {(datetime.now() - oldest_time).days} days")
    
    # Display detailed table if requested
    if details and cache_files:
        console.print("\n[bold]Cache Contents:[/bold]")
        
        # Sort by most recent first
        cache_files.sort(key=lambda x: x['time'], reverse=True)
        
        table = Table(box=rich.box.SIMPLE)
        table.add_column("Date", style="cyan")
        table.add_column("Size", style="green", justify="right")
        table.add_column("Content Preview", style="yellow")
        
        # Show most recent 20 entries
        for entry in cache_files[:20]:
            table.add_row(
                entry['time'].strftime('%Y-%m-%d %H:%M:%S'),
                format_size(entry['size']),
                entry['snippet']
            )
        
        console.print(table)
        
        if len(cache_files) > 20:
            console.print(f"\n[dim]Showing 20 of {len(cache_files)} entries. Use --export to see all.[/dim]")

def export_cache_list(cache_dir, output_file):
    """Export a list of all cache entries to a file"""
    if not os.path.exists(cache_dir):
        console.print(f"[yellow]Cache directory {cache_dir} does not exist.[/yellow]")
        return
    
    cache_files = []
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract a snippet of content
                    snippet = ""
                    if 'args' in data and isinstance(data['args'], list) and len(data['args']) > 0:
                        arg = data['args'][0]
                        if isinstance(arg, str):
                            snippet = arg[:100] + "..." if len(arg) > 100 else arg
                    
                    cache_files.append({
                        'filename': filename,
                        'size': file_size,
                        'time': file_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'content': snippet
                    })
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Skip files that aren't valid JSON
                cache_files.append({
                    'filename': filename,
                    'size': file_size,
                    'time': file_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'content': "[Invalid JSON]"
                })
    
    # Sort by most recent first
    cache_files.sort(key=lambda x: x['time'], reverse=True)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cache_files, f, indent=2)
    
    console.print(f"[green]Exported {len(cache_files)} cache entries to {output_file}[/green]")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Empathy Toolkit Cache Manager")
    parser.add_argument("--cache-dir", default=".empathy_cache", help="Cache directory location")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View cache statistics")
    view_parser.add_argument("--details", action="store_true", help="Show detailed information about cache entries")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the cache")
    clear_parser.add_argument("--older-than", type=int, help="Clear only entries older than specified days")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export cache list to a file")
    export_parser.add_argument("--output", required=True, help="Output file for the cache list")
    
    args = parser.parse_args()
    
    if args.command == "view":
        view_cache(args.cache_dir, args.details)
    elif args.command == "clear":
        clear_cache(args.cache_dir, args.older_than)
    elif args.command == "export":
        export_cache_list(args.cache_dir, args.output)
    else:
        # Default to view if no command specified
        view_cache(args.cache_dir, False)
        parser.print_help()

if __name__ == "__main__":
    main()
