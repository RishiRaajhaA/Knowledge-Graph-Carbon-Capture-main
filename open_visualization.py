#!/usr/bin/env python3
"""
Open the knowledge graph visualization in the default browser.
"""

import webbrowser
import os
from pathlib import Path

def open_visualization():
    """Open the knowledge graph visualization in the default web browser."""
    
    # Get the absolute path to the HTML file
    current_dir = Path(__file__).parent
    html_file = current_dir / "output" / "knowledge_graph_visualization.html"
    
    if html_file.exists():
        # Convert to file URL for the browser
        file_url = html_file.as_uri()
        
        print(f"Opening knowledge graph visualization...")
        print(f"File: {html_file}")
        print(f"URL: {file_url}")
        
        # Open in default browser
        webbrowser.open(file_url)
        print("Visualization opened in your default browser!")
        
    else:
        print(f"Error: Visualization file not found at {html_file}")
        print("Please run create_kg_visualization.py first.")

if __name__ == "__main__":
    open_visualization()