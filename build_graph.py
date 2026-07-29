"""
Build Knowledge Graph from chunked texts using Gemini API.
Processes all chunks (2000 char each) to maximize entity extraction.
Output: output/knowledge_graph.json
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

ENTITY_EXTRACTION_PROMPT = """
Extract key entities and their relationships from this carbon capture research text.

Entity Types (use EXACTLY these labels):
1. Material Composite - Materials, adsorbents, substances
2. Property-Value Pairs - Measurements with numeric values and units (ALWAYS include the material name in context)
3. Process - Methods, techniques, procedures, applications
4. Conditions - Temperature, pressure, pH, concentration

Rules:
- Extract 8-12 entities per chunk (prioritize properties with values)
- IMPORTANT: For each property, mention which material it belongs to
- Extract ALL numeric measurements (CO2 capacity, surface area, pore volume, selectivity, etc.)
- Include the specific material variation (e.g., "GO", "rGO", "3D graphene" - keep distinctions)
- Create direct relationships between materials and their properties
- Link materials to the processes that create them
- Connect properties to the conditions they were measured at

Text:
{text}

Return valid JSON with MORE relationships (aim for 2-3 relations per entity):
{{
  "entities": [
    {{"type": "Material Composite", "name": "3D graphene", "value": null, "chunk": "3D graphene was synthesized..."}},
    {{"type": "Property-Value Pairs", "name": "CO2 capacity", "value": "8.9 mmol/g", "chunk": "3D graphene showed CO2 capacity of 8.9 mmol/g..."}},
    {{"type": "Process", "name": "thermal reduction", "value": null, "chunk": "thermal reduction method..."}},
    {{"type": "Conditions", "name": "temperature", "value": "298K", "chunk": "measured at 298K..."}}
  ],
  "relations": [
    {{"source": "3D graphene", "target": "CO2 capacity", "relation": "has_property"}},
    {{"source": "3D graphene", "target": "thermal reduction", "relation": "synthesized_by"}},
    {{"source": "CO2 capacity", "target": "temperature", "relation": "measured_at"}},
    {{"source": "CO2 capacity", "target": "3D graphene", "relation": "property_of"}}
  ]
}}
"""


def extract_entities_with_gemini(text, paper_id, chunk_ref):
    """Extract entities and relations from text using Gemini."""
    try:
        # Clean null bytes and problematic characters from text
        text = text.replace('\u0000', '').replace('\x00', '')
        
        # Additional aggressive cleaning
        text = text.replace('selfassembly', 'self assembly')
        text = text.replace('desorbing', 'desorption')
        text = text.replace('adsorption -desorption', 'adsorption desorption')
        text = text.replace('bestperforming', 'best performing')
        
        # Use gemini-2.0-flash-001 (better safety handling)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Text is already chunked to 2000 chars, use it directly
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        # Safety settings - MUST use enum format (dict, not list!)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,  # Increased to allow more entities
            ),
            safety_settings=safety_settings
        )
        
        # Check if response was blocked
        if not response.candidates or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            print(f"  ⚠ Response blocked (finish_reason: {finish_reason}) for {paper_id[:40]} ({chunk_ref})")
            return {"entities": [], "relations": []}
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text)
        
        # Add metadata to entities
        for entity in data.get("entities", []):
            entity["source"] = f"{paper_id}_{chunk_ref}"
            entity["metadata"] = {"paper_id": paper_id, "chunk": chunk_ref}
        
        return data
    
    except Exception as e:
        print(f"Error extracting from {paper_id[:40]} ({chunk_ref}): {e}")
        return {"entities": [], "relations": []}


def build_knowledge_graph(chunked_texts):
    """Build knowledge graph from chunked texts."""
    all_nodes = []
    all_edges = []
    node_id_counter = 1
    entity_to_id = {}  # Map entity names to node IDs
    
    print("\nExtracting entities with Gemini API...")
    
    # Calculate total chunks
    total_chunks = sum(data['total_chunks'] for data in chunked_texts.values())
    
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for paper_id, paper_data in chunked_texts.items():
            for chunk_info in paper_data['chunks']:
                section = chunk_info['section']
                chunk_id = chunk_info['chunk_id']
                text = chunk_info['text']
                
                # Extract entities from this chunk
                result = extract_entities_with_gemini(
                    text, 
                    paper_id, 
                    f"{section}_chunk{chunk_id}"
                )
                
                for entity in result.get("entities", []):
                    entity_key = f"{entity['name']}_{entity.get('value', '')}"
                    
                    if entity_key not in entity_to_id:
                        # Store chunk_id reference instead of full text for efficiency
                        chunk_ref = f"{paper_id}#{section}#{chunk_id}"
                        
                        node = {
                            "id": f"node_{node_id_counter}",
                            "type": entity["type"],
                            "name": entity["name"],
                            "chunk_id": chunk_ref,  # Reference to chunk in extracted_texts_chunked.json
                            "source": entity.get("source", ""),
                            "metadata": {
                                "paper_id": paper_id,
                                "section": section,
                                "chunk_index": chunk_id,
                                "entity_mention": entity.get("chunk", "")  # Keep original mention for reference
                            }
                        }
                        if "value" in entity:
                            node["value"] = entity["value"]
                        
                        all_nodes.append(node)
                        entity_to_id[entity_key] = node["id"]
                        node_id_counter += 1
                
                # Add relations
                for relation in result.get("relations", []):
                    source_key = f"{relation['source']}_"
                    target_key = f"{relation['target']}_"
                    
                    # Find matching entities
                    source_id = None
                    target_id = None
                    
                    for key, nid in entity_to_id.items():
                        if key.startswith(source_key):
                            source_id = nid
                        if key.startswith(target_key):
                            target_id = nid
                    
                    if source_id and target_id:
                        edge = {
                            "source": source_id,
                            "target": target_id,
                            "relation": relation["relation"]
                        }
                        all_edges.append(edge)
                
                pbar.update(1)
                time.sleep(0.5)  # Rate limiting
    
    return {"nodes": all_nodes, "edges": all_edges}


def main():
    """Main function to build knowledge graph."""
    # Load chunked texts
    input_path = Path("output/extracted_texts_chunked.json")
    if not input_path.exists():
        print("❌ Error: output/extracted_texts_chunked.json not found. Run extract_text.py first.")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunked_texts = json.load(f)
    
    total_chunks = sum(data['total_chunks'] for data in chunked_texts.values())
    print(f"Loaded {len(chunked_texts)} papers with {total_chunks} total chunks")
    
    # Build knowledge graph
    kg = build_knowledge_graph(chunked_texts)
    
    # Save results
    output_path = Path("output/knowledge_graph.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Knowledge graph built successfully")
    print(f"✓ Saved to {output_path}")
    print(f"  - Total nodes: {len(kg['nodes'])}")
    print(f"  - Total edges: {len(kg['edges'])}")
    
    # Print node distribution
    node_types = {}
    for node in kg['nodes']:
        node_types[node['type']] = node_types.get(node['type'], 0) + 1
    
    print(f"\nNode distribution:")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {node_type}: {count}")


if __name__ == "__main__":
    main()
