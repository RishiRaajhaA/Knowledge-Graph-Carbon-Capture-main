"""
RAG Chatbot UI - Web-based interface for Carbon Capture Knowledge Graph
Based on the RAG framework from evaluate.py
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Initialize models and data
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
CHUNK_DATA = None
KNOWLEDGE_GRAPH = None

def load_data():
    """Load knowledge graph and chunk data."""
    global CHUNK_DATA, KNOWLEDGE_GRAPH
    
    # Load knowledge graph
    kg_path = Path("output/knowledge_graph.json")
    if kg_path.exists():
        with open(kg_path, 'r', encoding='utf-8') as f:
            KNOWLEDGE_GRAPH = json.load(f)
        print(f"✅ Loaded knowledge graph: {len(KNOWLEDGE_GRAPH.get('nodes', []))} nodes, {len(KNOWLEDGE_GRAPH.get('edges', []))} edges")
    else:
        print("❌ Knowledge graph not found!")
        KNOWLEDGE_GRAPH = {'nodes': [], 'edges': []}
    
    # Load chunk data
    chunk_path = Path("output/extracted_texts_chunked.json")
    if chunk_path.exists():
        with open(chunk_path, 'r', encoding='utf-8') as f:
            CHUNK_DATA = json.load(f)
        print(f"✅ Loaded chunk data: {len(CHUNK_DATA)} papers")
    else:
        print("❌ Chunk data not found!")
        CHUNK_DATA = {}

def get_chunk_by_id(chunk_id):
    """Fetch chunk text by chunk_id (format: paper_id#section#chunk_index)."""
    try:
        parts = chunk_id.split('#')
        if len(parts) != 3:
            return ""
        
        paper_id, section, chunk_index = parts
        chunk_index = int(chunk_index)
        
        if paper_id in CHUNK_DATA and 'chunks' in CHUNK_DATA[paper_id]:
            chunks = CHUNK_DATA[paper_id]['chunks']
            for chunk in chunks:
                if chunk.get('section') == section and chunk.get('chunk_id') == chunk_index:
                    return chunk.get('text', '')
        
        return ""
    except Exception as e:
        print(f"Error fetching chunk {chunk_id}: {e}")
        return ""

def get_connected_nodes(node_id, kg, max_hops=2):
    """Get nodes connected to the given node via graph edges."""
    connected = set()
    current_level = {node_id}
    visited = {node_id}
    
    for hop in range(max_hops):
        next_level = set()
        
        for edge in kg.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            
            if source in current_level and target not in visited:
                next_level.add(target)
                connected.add(target)
                visited.add(target)
            elif target in current_level and source not in visited:
                next_level.add(source)
                connected.add(source)
                visited.add(source)
        
        if not next_level:
            break
        
        current_level = next_level
    
    return connected

def retrieve_context_for_question(question, top_k=5, use_graph_expansion=True, max_hops=1):
    """Enhanced retrieval: Entity matching + semantic search + graph expansion."""
    if not KNOWLEDGE_GRAPH or not question.strip():
        return "", []
    
    question_lower = question.lower()
    question_embedding = semantic_model.encode([question])
    
    # Step 1: Find nodes whose entity names appear in the question
    node_id_to_node = {node['id']: node for node in KNOWLEDGE_GRAPH['nodes']}
    candidate_nodes = []
    
    for node_id, node in node_id_to_node.items():
        entity_name = node.get('name', '').lower()
        
        if len(entity_name) < 3:
            continue
        
        if entity_name in question_lower:
            candidate_nodes.append(node)
    
    # If no direct matches, use keyword overlap
    if len(candidate_nodes) < top_k:
        name_candidates = []
        for node in KNOWLEDGE_GRAPH['nodes']:
            entity_name = node.get('name', '')
            if not entity_name or len(entity_name) < 3:
                continue
            
            name_words = set(entity_name.lower().split())
            question_words = set(question_lower.split())
            overlap = len(name_words & question_words)
            
            if overlap > 0:
                name_candidates.append((node, overlap))
        
        name_candidates.sort(key=lambda x: x[1], reverse=True)
        for node, _ in name_candidates[:10]:
            if node not in candidate_nodes:
                candidate_nodes.append(node)
    
    # Step 2: Graph expansion
    if use_graph_expansion and candidate_nodes:
        expanded_nodes = []
        seed_node_ids = {node['id'] for node in candidate_nodes}
        
        for seed_node in candidate_nodes:
            connected_node_ids = get_connected_nodes(seed_node['id'], KNOWLEDGE_GRAPH, max_hops=max_hops)
            
            for conn_id in connected_node_ids:
                if conn_id not in seed_node_ids and conn_id in node_id_to_node:
                    expanded_nodes.append(node_id_to_node[conn_id])
        
        candidate_nodes.extend(expanded_nodes[:20])
    
    # Step 3: Score and rank chunks
    candidates = []
    seen_chunks = set()
    
    for node in candidate_nodes:
        chunk_id = node.get('chunk_id', '')
        if not chunk_id:
            continue
        
        chunk = get_chunk_by_id(chunk_id)
        if not chunk or len(chunk) < 50:
            continue
        
        chunk_hash = hash(chunk)
        if chunk_hash in seen_chunks:
            continue
        seen_chunks.add(chunk_hash)
        
        # Semantic similarity
        chunk_embedding = semantic_model.encode([chunk])
        semantic_score = cosine_similarity(question_embedding, chunk_embedding)[0][0]
        
        # Keyword overlap
        question_words = set(question_lower.split())
        chunk_words = set(chunk.lower().split())
        keyword_overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
        
        # Combined relevance score
        relevance_score = 0.8 * semantic_score + 0.2 * keyword_overlap
        
        candidates.append({
            'node': node,
            'chunk': chunk,
            'semantic_score': semantic_score,
            'keyword_score': keyword_overlap,
            'relevance_score': relevance_score,
            'source_paper': node.get('metadata', {}).get('paper_id', 'unknown'),
            'section': node.get('metadata', {}).get('section', 'unknown')
        })
    
    # Filter and sort
    candidates = [c for c in candidates if c['relevance_score'] >= 0.3]
    candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Select top-k
    top_candidates = candidates[:top_k]
    retrieved_chunks = [c['chunk'] for c in top_candidates]
    
    return ' '.join(retrieved_chunks), top_candidates

def generate_answer_with_context(question, context, response_type="comprehensive"):
    """Generate answer using Gemini with retrieved context."""
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        if response_type == "short":
            length_instruction = "Answer in maximum 30 words. Be concise and factual."
            max_tokens = 128
        elif response_type == "long":
            length_instruction = "Provide a detailed technical answer (150-250 words). Include specific measurements, materials, and processes mentioned in the context."
            max_tokens = 768
        else:  # auto/comprehensive
            length_instruction = "Provide a comprehensive answer (60-120 words). Include specific details from the context."
            max_tokens = 512
        
        prompt = f"""You are an expert in carbon capture research. Answer the following question based ONLY on the provided context from research papers.

Context: {context}

Question: {question}

Instructions: {length_instruction}

Answer:"""
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I apologize, but I encountered an error while generating the answer: {str(e)}"

@app.route('/')
def index():
    """Main chatbot interface."""
    return render_template('chatbot_new.html')

@app.route('/static/knowledge_graph_visualization.html')
def knowledge_graph():
    """Serve the knowledge graph visualization."""
    kg_path = Path("output/knowledge_graph_visualization.html")
    if kg_path.exists():
        with open(kg_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Knowledge graph visualization not found!", 404

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.json
        question = data.get('message', '').strip()
        history = data.get('history', [])
        response_type = data.get('response_type', 'auto')
        
        if not question:
            return jsonify({'error': 'Please provide a question.'}), 400
        
        # Retrieve relevant context
        context, sources = retrieve_context_for_question(
            question, 
            top_k=5, 
            use_graph_expansion=True
        )
        
        if not context:
            return jsonify({
                'response': "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or asking about carbon capture materials, processes, or properties.",
                'sources': [],
                'context_found': False
            })
        
        # Generate answer
        answer = generate_answer_with_context(question, context, response_type)
        
        # Prepare source information
        source_info = []
        for source in sources[:3]:  # Top 3 sources
            source_info.append({
                'paper': source['source_paper'].replace('_', ' ').title(),
                'section': source['section'].title(),
                'relevance': f"{source['relevance_score']:.2f}",
                'snippet': source['chunk'][:200] + '...' if len(source['chunk']) > 200 else source['chunk']
            })
        
        return jsonify({
            'response': answer,
            'sources': source_info,
            'context_found': True,
            'num_sources': len(sources)
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/stats')
def stats():
    """Get knowledge base statistics."""
    return jsonify({
        'total_nodes': len(KNOWLEDGE_GRAPH.get('nodes', [])),
        'total_edges': len(KNOWLEDGE_GRAPH.get('edges', [])),
        'total_papers': len(CHUNK_DATA),
        'node_types': list(set(node.get('type', 'Unknown') for node in KNOWLEDGE_GRAPH.get('nodes', []))),
        'status': 'ready'
    })

if __name__ == '__main__':
    print("🚀 Starting RAG Chatbot...")
    load_data()
    print("🌐 Starting web server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)