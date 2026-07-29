#!/usr/bin/env python3
"""
Create HTML visualization for knowledge graph data using the template.
"""

import json
import os

def load_knowledge_graph(filepath):
    """Load the knowledge graph JSON data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_template(filepath):
    """Load the HTML template."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_visualization(kg_data, template, output_path):
    """Create the HTML visualization by embedding data into template."""
    
    # Convert the knowledge graph data to the format expected by the template
    # The template expects RAW_GRAPH with nodes and edges
    raw_graph = {
        "nodes": [],
        "edges": []
    }
    
    # Process nodes
    for node in kg_data.get("nodes", []):
        # Extract relevant information for visualization
        node_data = {
            "text": node.get("name", node.get("id", "Unknown")),
            "type": node.get("type", "Unknown"),
            "id": node.get("id", ""),
            "provenance": {
                "chunk_id": node.get("chunk_id", ""),
                "section": node.get("metadata", {}).get("section", ""),
                "source_file": node.get("metadata", {}).get("paper_id", "unknown"),
                "entity_mention": node.get("metadata", {}).get("entity_mention", "")
            }
        }
        raw_graph["nodes"].append(node_data)
    
    # Process edges
    for edge in kg_data.get("edges", []):
        edge_data = {
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "relation": edge.get("relation", "related_to")
        }
        raw_graph["edges"].append(edge_data)
    
    # Convert to JSON string for embedding
    graph_json = json.dumps(raw_graph, separators=(',', ':'))
    
    # Replace the RAW_GRAPH placeholder in the template
    # Find the line with RAW_GRAPH definition and replace it
    lines = template.split('\n')
    for i, line in enumerate(lines):
        if 'const RAW_GRAPH = ' in line:
            # Replace the entire line with our data
            lines[i] = f'        const RAW_GRAPH = {graph_json};'
            break
    
    # Update the total count in the stats section
    total_nodes = len(raw_graph["nodes"])
    for i, line in enumerate(lines):
        if '<span class="stat-number">1675</span>' in line:
            lines[i] = line.replace('1675', str(total_nodes))
            break
    
    # Join the lines back together
    output_html = '\n'.join(lines)
    
    # Write the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_html)
    
    print(f"HTML visualization created: {output_path}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {len(raw_graph['edges'])}")

def main():
    # Define file paths
    kg_path = "output/knowledge_graph.json"
    template_path = "output/kg_template.html"
    output_path = "output/knowledge_graph_visualization.html"
    
    # Check if files exist
    if not os.path.exists(kg_path):
        print(f"Error: Knowledge graph file not found: {kg_path}")
        return
    
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return
    
    # Load data and template
    print("Loading knowledge graph data...")
    kg_data = load_knowledge_graph(kg_path)
    
    print("Loading HTML template...")
    template = load_template(template_path)
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(kg_data, template, output_path)
    
    print(f"\nVisualization complete! Open {output_path} in your browser to view the interactive knowledge graph.")

if __name__ == "__main__":
    main()