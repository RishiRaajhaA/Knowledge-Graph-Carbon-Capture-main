"""
Evaluate Pre-RAG (Retrieval) and Post-RAG (Generation) performance.
Metrics: Precision, Recall, F1, Semantic Similarity, ROUGE-1, BLEU, Exact Match
Output: output/evaluations.json
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Suppress gRPC/ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# Load chunk data once for fast lookups
CHUNK_DATA = None

def load_chunk_data():
    """Load extracted_texts_chunked.json for chunk lookups."""
    global CHUNK_DATA
    if CHUNK_DATA is None:
        chunk_path = Path("output/extracted_texts_chunked.json")
        if chunk_path.exists():
            with open(chunk_path, 'r', encoding='utf-8') as f:
                CHUNK_DATA = json.load(f)
        else:
            print("⚠️ Warning: extracted_texts_chunked.json not found")
            CHUNK_DATA = {}
    return CHUNK_DATA


def get_chunk_by_id(chunk_id):
    """Fetch chunk text by chunk_id (format: paper_id#section#chunk_index).
    
    Args:
        chunk_id: String in format "paper_id#section#chunk_index"
        
    Returns:
        Chunk text or empty string if not found
    """
    chunk_data = load_chunk_data()
    
    try:
        # Parse chunk_id: "paper_id#section#chunk_index"
        parts = chunk_id.split('#')
        if len(parts) != 3:
            return ""
        
        paper_id, section, chunk_index = parts
        chunk_index = int(chunk_index)
        
        # Navigate to the chunk
        if paper_id in chunk_data and 'chunks' in chunk_data[paper_id]:
            chunks = chunk_data[paper_id]['chunks']
            for chunk in chunks:
                if chunk.get('section') == section and chunk.get('chunk_id') == chunk_index:
                    return chunk.get('text', '')
        
        return ""
    except Exception as e:
        print(f"Error fetching chunk {chunk_id}: {e}")
        return ""


def compute_semantic_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    emb1 = semantic_model.encode([text1])
    emb2 = semantic_model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]


def compute_rouge1_f1(reference, hypothesis):
    """Compute ROUGE-1 F1 score."""
    scores = rouge_scorer_obj.score(reference, hypothesis)
    return scores['rouge1'].fmeasure


def compute_bleu(reference, hypothesis):
    """Compute BLEU score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


def compute_exact_match(reference, hypothesis):
    """Compute exact match (1 if identical, 0 otherwise)."""
    return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0


def get_connected_nodes(node_id, kg, max_hops=2):
    """Get nodes connected to the given node via graph edges (multi-hop).
    
    Args:
        node_id: Starting node ID
        kg: Knowledge graph with nodes and edges
        max_hops: Maximum number of hops to traverse (default: 2)
    
    Returns:
        Set of connected node IDs
    """
    connected = set()
    current_level = {node_id}
    visited = {node_id}
    
    for hop in range(max_hops):
        next_level = set()
        
        for edge in kg.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            
            # Check if edge connects to current level
            if source in current_level and target not in visited:
                next_level.add(target)
                connected.add(target)
                visited.add(target)
            elif target in current_level and source not in visited:
                next_level.add(source)
                connected.add(source)
                visited.add(source)
        
        if not next_level:  # No more connections
            break
        
        current_level = next_level
    
    return connected


def retrieve_context_for_qa(qa_pair, kg, top_k=3, similarity_threshold=0.3, use_graph_expansion=False, max_hops=1, precomputed_embeddings=None):
    """Enhanced retrieval: First match entity names, then semantic search on chunks.
    
    This is much faster as it uses entity name matching as a filter before semantic search.
    
    Args:
        qa_pair: QA pair with question (no source_entities needed)
        kg: Knowledge graph with nodes and edges
        top_k: Maximum number of chunks to retrieve (default: 3)
        similarity_threshold: Minimum semantic similarity score (default: 0.3)
        use_graph_expansion: Whether to expand to connected nodes (default: False)
        max_hops: Maximum number of graph hops for expansion (default: 1)
        precomputed_embeddings: Dict of {node_id: embedding} for faster retrieval
    
    Returns:
        String of concatenated top-k relevant chunks
    """
    question = qa_pair.get('question', '')
    question_lower = question.lower()
    
    # Get question embedding once
    question_embedding = semantic_model.encode([question])
    
    # Step 1: FAST FILTER - Find nodes whose entity names appear in the question
    node_id_to_node = {node['id']: node for node in kg['nodes']}
    candidate_nodes = []
    
    for node_id, node in node_id_to_node.items():
        entity_name = node.get('name', '').lower()
        
        # Skip very short entity names (too generic)
        if len(entity_name) < 3:
            continue
        
        # Check if entity name appears in question (fuzzy match)
        if entity_name in question_lower:
            candidate_nodes.append(node)
    
    # If no entity name matches, fall back to top semantic candidates
    if len(candidate_nodes) < top_k:
        # Get top-k most similar nodes by name similarity
        name_candidates = []
        for node in kg['nodes']:
            entity_name = node.get('name', '')
            if not entity_name or len(entity_name) < 3:
                continue
            
            # Quick keyword overlap on entity name
            name_words = set(entity_name.lower().split())
            question_words = set(question_lower.split())
            overlap = len(name_words & question_words)
            
            if overlap > 0:
                name_candidates.append((node, overlap))
        
        # Sort by overlap and add top ones
        name_candidates.sort(key=lambda x: x[1], reverse=True)
        for node, _ in name_candidates[:10]:  # Top 10 by name match
            if node not in candidate_nodes:
                candidate_nodes.append(node)
    
    # Step 1.5: GRAPH EXPANSION - Add connected nodes via edges
    if use_graph_expansion and candidate_nodes:
        print(f"  [Graph Expansion] Starting with {len(candidate_nodes)} seed nodes...")
        expanded_nodes = []
        seed_node_ids = {node['id'] for node in candidate_nodes}
        
        for seed_node in candidate_nodes:
            # Get connected nodes (1-hop or 2-hop neighbors)
            connected_node_ids = get_connected_nodes(seed_node['id'], kg, max_hops=max_hops)
            
            # Add connected nodes to expansion list
            for conn_id in connected_node_ids:
                if conn_id not in seed_node_ids and conn_id in node_id_to_node:
                    expanded_nodes.append(node_id_to_node[conn_id])
        
        # Add expanded nodes to candidates (limit to avoid explosion)
        candidate_nodes.extend(expanded_nodes[:20])  # Add top 20 connected nodes
        print(f"  [Graph Expansion] Expanded to {len(candidate_nodes)} total nodes (added {len(expanded_nodes[:20])} neighbors)")
    
    # Step 2: Score candidate node chunks by semantic similarity
    candidates = []
    seen_chunks = set()
    
    # Track which nodes came from graph expansion (for boosting)
    seed_node_ids = set()
    if candidate_nodes:
        seed_node_ids = {candidate_nodes[i]['id'] for i in range(min(10, len(candidate_nodes)))}
    
    for node in candidate_nodes:
        # NEW: Get chunk using chunk_id instead of direct access
        chunk_id = node.get('chunk_id', '')
        if not chunk_id:
            continue
        
        chunk = get_chunk_by_id(chunk_id)
        if not chunk or len(chunk) < 50:
            continue
        
        # DEDUPLICATION
        chunk_hash = hash(chunk)
        if chunk_hash in seen_chunks:
            continue
        seen_chunks.add(chunk_hash)
        
        # Compute semantic similarity for this chunk
        chunk_embedding = semantic_model.encode([chunk])
        semantic_score = cosine_similarity(question_embedding, chunk_embedding)[0][0]
        
        # Keyword overlap
        question_words = set(question_lower.split())
        chunk_words = set(chunk.lower().split())
        keyword_overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
        
        # Graph relationship boost (prioritize direct matches over expanded nodes)
        graph_boost = 1.2 if node['id'] in seed_node_ids else 1.0
        
        # Combined score (increased semantic weight from 0.7 to 0.8)
        relevance_score = (0.8 * semantic_score + 0.2 * keyword_overlap) * graph_boost
        
        candidates.append({
            'node': node,
            'chunk': chunk,
            'semantic_score': semantic_score,
            'keyword_score': keyword_overlap,
            'relevance_score': relevance_score
        })
    
    # Filter by threshold and sort by relevance
    candidates = [c for c in candidates if c['relevance_score'] >= similarity_threshold]
    candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Select top-k chunks
    top_candidates = candidates[:top_k]
    
    # Concatenate top chunks (deduplicated)
    retrieved_chunks = [c['chunk'] for c in top_candidates]
    
    return ' '.join(retrieved_chunks)


def generate_answer_with_context(question, context, qa_type="short"):
    """Generate answer using Gemini with retrieved context.
    
    Args:
        question: The question to answer
        context: Retrieved context chunks
        qa_type: "short" (max 20 words) or "long" (60-90 words)
    """
    try:
        # Safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Adjust prompt based on question type
        if qa_type == "short":
            length_instruction = "Answer in MAX 20 words. Be concise and factual."
            max_tokens = 128
        else:  # long
            length_instruction = "Answer in 60-90 words. Be comprehensive but concise. Count carefully!"
            max_tokens = 512
        
        prompt = f"""Answer the following question based ONLY on the provided context.

Context: {context}

Question: {question}

{length_instruction}

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
        return ""


def evaluate_pre_rag(qa_pairs, kg):
    """Evaluate retrieval performance (Pre-RAG)."""
    print("\nEvaluating Pre-RAG (Retrieval)...")
    
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'semantic_similarity': [],
        'rouge1_f1': [],
        'bleu': [],
        'exact_match': []
    }
    
    detailed_results = []  # Store per-question results
    
    for qa in qa_pairs:  # Evaluate all questions
        # Retrieve context with graph expansion enabled
        retrieved_context = retrieve_context_for_qa(
            qa, kg, 
            top_k=5,                    # Retrieve more candidates
            use_graph_expansion=True,   # Enable graph traversal
            max_hops=1                  # 1-hop neighbor expansion
        )
        reference_answer = qa['answer']
        
        if not retrieved_context:
            continue
        
        # Compute metrics comparing retrieved context to reference answer
        sem_sim = compute_semantic_similarity(retrieved_context, reference_answer)
        rouge_f1 = compute_rouge1_f1(reference_answer, retrieved_context)
        bleu = compute_bleu(reference_answer, retrieved_context)
        
        metrics['semantic_similarity'].append(sem_sim)
        metrics['rouge1_f1'].append(rouge_f1)
        metrics['bleu'].append(bleu)
        
        # Precision/Recall/F1: simplified - check if key terms present
        ref_terms = set(reference_answer.lower().split())
        ret_terms = set(retrieved_context.lower().split())
        
        if len(ret_terms) > 0:
            precision = len(ref_terms & ret_terms) / len(ret_terms)
            metrics['precision'].append(precision)
        
        if len(ref_terms) > 0:
            recall = len(ref_terms & ret_terms) / len(ref_terms)
            metrics['recall'].append(recall)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            metrics['f1'].append(f1)
        
        # Exact match for retrieval
        exact_match = compute_exact_match(reference_answer, retrieved_context)
        metrics['exact_match'].append(exact_match)
        
        # Store detailed result for this question
        detailed_results.append({
            'question': qa['question'],
            'reference_answer': reference_answer,
            'retrieved_context': retrieved_context[:500] + '...' if len(retrieved_context) > 500 else retrieved_context,
            'retrieved_context_length': len(retrieved_context),
            'source_paper': qa.get('source_paper', 'unknown'),
            'source_section': qa.get('source_section', 'unknown'),
            'scores': {
                'semantic_similarity': float(sem_sim),
                'rouge1_f1': float(rouge_f1),
                'recall': float(recall) if len(ref_terms) > 0 else 0.0,
                'f1': float(f1) if precision + recall > 0 else 0.0,
                'precision': float(precision) if len(ret_terms) > 0 else 0.0,
                'bleu': float(bleu),
                'exact_match': float(exact_match)
            }
        })
    
    # Average all metrics - return in requested order
    return {
        'semantic_similarity': float(np.mean(metrics['semantic_similarity'])) if metrics['semantic_similarity'] else 0.0,
        'rouge1_f1': float(np.mean(metrics['rouge1_f1'])) if metrics['rouge1_f1'] else 0.0,
        'recall': float(np.mean(metrics['recall'])) if metrics['recall'] else 0.0,
        'f1': float(np.mean(metrics['f1'])) if metrics['f1'] else 0.0,
        'precision': float(np.mean(metrics['precision'])) if metrics['precision'] else 0.0,
        'bleu': float(np.mean(metrics['bleu'])) if metrics['bleu'] else 0.0,
        'exact_match': float(np.mean(metrics['exact_match'])) if metrics['exact_match'] else 0.0
    }, detailed_results


def evaluate_post_rag(qa_pairs, kg, qa_type="short"):
    """Evaluate generation performance (Post-RAG)."""
    print("\nEvaluating Post-RAG (Generation)...")
    
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'semantic_similarity': [],
        'rouge1_f1': [],
        'bleu': [],
        'exact_match': []
    }
    
    detailed_results = []  # Store per-question results
    
    for i, qa in enumerate(qa_pairs[:10]):  # Sample for evaluation
        print(f"  Generating answer {i+1}/10...")
        
        # Retrieve context with graph expansion and generate answer
        retrieved_context = retrieve_context_for_qa(
            qa, kg,
            top_k=5,                    # Retrieve more candidates
            use_graph_expansion=True,   # Enable graph traversal
            max_hops=1                  # 1-hop neighbor expansion
        )
        if not retrieved_context:
            continue
        
        generated_answer = generate_answer_with_context(qa['question'], retrieved_context, qa_type=qa_type)
        reference_answer = qa['answer']
        
        if not generated_answer:
            continue
        
        # Compute all metrics
        sem_sim = compute_semantic_similarity(generated_answer, reference_answer)
        rouge_f1 = compute_rouge1_f1(reference_answer, generated_answer)
        bleu = compute_bleu(reference_answer, generated_answer)
        exact = compute_exact_match(reference_answer, generated_answer)
        
        metrics['semantic_similarity'].append(sem_sim)
        metrics['rouge1_f1'].append(rouge_f1)
        metrics['bleu'].append(bleu)
        metrics['exact_match'].append(exact)
        
        # Precision/Recall/F1 for generation
        ref_terms = set(reference_answer.lower().split())
        gen_terms = set(generated_answer.lower().split())
        
        if len(gen_terms) > 0:
            precision = len(ref_terms & gen_terms) / len(gen_terms)
            metrics['precision'].append(precision)
        
        if len(ref_terms) > 0:
            recall = len(ref_terms & gen_terms) / len(ref_terms)
            metrics['recall'].append(recall)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            metrics['f1'].append(f1)
        else:
            f1 = 0.0
        
        # Store detailed result for this question
        detailed_results.append({
            'question': qa['question'],
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'retrieved_context': retrieved_context[:500] + '...' if len(retrieved_context) > 500 else retrieved_context,
            'retrieved_context_length': len(retrieved_context),
            'source_paper': qa.get('source_paper', 'unknown'),
            'source_section': qa.get('source_section', 'unknown'),
            'scores': {
                'semantic_similarity': float(sem_sim),
                'rouge1_f1': float(rouge_f1),
                'recall': float(recall) if len(ref_terms) > 0 else 0.0,
                'f1': float(f1),
                'precision': float(precision) if len(gen_terms) > 0 else 0.0,
                'bleu': float(bleu),
                'exact_match': float(exact)
            }
        })
    
    # Average all metrics - return in requested order
    return {
        'semantic_similarity': float(np.mean(metrics['semantic_similarity'])) if metrics['semantic_similarity'] else 0.0,
        'rouge1_f1': float(np.mean(metrics['rouge1_f1'])) if metrics['rouge1_f1'] else 0.0,
        'recall': float(np.mean(metrics['recall'])) if metrics['recall'] else 0.0,
        'f1': float(np.mean(metrics['f1'])) if metrics['f1'] else 0.0,
        'precision': float(np.mean(metrics['precision'])) if metrics['precision'] else 0.0,
        'bleu': float(np.mean(metrics['bleu'])) if metrics['bleu'] else 0.0,
        'exact_match': float(np.mean(metrics['exact_match'])) if metrics['exact_match'] else 0.0
    }, detailed_results


def main():
    """Main evaluation function."""
    # Load knowledge graph
    kg_path = Path("output/knowledge_graph.json")
    if not kg_path.exists():
        print("❌ Error: output/knowledge_graph.json not found.")
        return
    
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    # Load Q&A pairs
    qa_short_path = Path("output/qa_short.json")
    qa_long_path = Path("output/qa_long.json")
    
    if not qa_short_path.exists() or not qa_long_path.exists():
        print("❌ Error: Q&A files not found. Run generate_qa.py first.")
        return
    
    with open(qa_short_path, 'r', encoding='utf-8') as f:
        qa_short = json.load(f)
    
    with open(qa_long_path, 'r', encoding='utf-8') as f:
        qa_long = json.load(f)
    
    print(f"Loaded {len(qa_short)} short QA pairs and {len(qa_long)} long QA pairs")
    
    # Evaluate Pre-RAG for short QA
    print("\n" + "="*60)
    print("PRE-RAG SHORT (Retrieval Only)")
    print("="*60)
    pre_rag_short_metrics, pre_rag_short_details = evaluate_pre_rag(qa_short, kg)
    
    # Evaluate Pre-RAG for long QA
    print("\n" + "="*60)
    print("PRE-RAG LONG (Retrieval Only)")
    print("="*60)
    pre_rag_long_metrics, pre_rag_long_details = evaluate_pre_rag(qa_long, kg)
    
    # Evaluate Post-RAG for short answers
    print("\n" + "="*60)
    print("POST-RAG SHORT (Retrieval + Generation)")
    print("="*60)
    post_rag_short_metrics, post_rag_short_details = evaluate_post_rag(qa_short, kg, qa_type="short")
    
    # Evaluate Post-RAG for long answers
    print("\n" + "="*60)
    print("POST-RAG LONG (Retrieval + Generation)")
    print("="*60)
    post_rag_long_metrics, post_rag_long_details = evaluate_post_rag(qa_long, kg, qa_type="long")
    
    # Compile results
    results = {
        "pre_rag_short": pre_rag_short_metrics,
        "pre_rag_long": pre_rag_long_metrics,
        "post_rag_short": post_rag_short_metrics,
        "post_rag_long": post_rag_long_metrics
    }
    
    # Compile detailed results
    detailed_results = {
        "pre_rag_short": {
            "method": "Retrieval Only (compare raw chunks to reference)",
            "qa_type": "short",
            "num_questions": len(pre_rag_short_details),
            "results": pre_rag_short_details
        },
        "pre_rag_long": {
            "method": "Retrieval Only (compare raw chunks to reference)",
            "qa_type": "long",
            "num_questions": len(pre_rag_long_details),
            "results": pre_rag_long_details
        },
        "post_rag_short": {
            "method": "Retrieval + Generation (LLM generates answer from chunks)",
            "qa_type": "short",
            "num_questions": len(post_rag_short_details),
            "results": post_rag_short_details
        },
        "post_rag_long": {
            "method": "Retrieval + Generation (LLM generates answer from chunks)",
            "qa_type": "long",
            "num_questions": len(post_rag_long_details),
            "results": post_rag_long_details
        }
    }
    
    # Save aggregate results
    output_path = Path("output/evaluations.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed results
    detailed_output_path = Path("output/evaluation_details.json")
    with open(detailed_output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Evaluation complete")
    print(f"✓ Saved aggregate metrics to {output_path}")
    print(f"✓ Saved detailed results to {detailed_output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\nPre-RAG Short (Retrieval) Metrics:")
    for metric, value in pre_rag_short_metrics.items():
        print(f"  {metric:.<25} {value:.3f}")
    
    print("\nPre-RAG Long (Retrieval) Metrics:")
    for metric, value in pre_rag_long_metrics.items():
        print(f"  {metric:.<25} {value:.3f}")
    
    print("\nPost-RAG Short Answer Metrics:")
    for metric, value in post_rag_short_metrics.items():
        print(f"  {metric:.<25} {value:.3f}")
    
    print("\nPost-RAG Long Answer Metrics:")
    for metric, value in post_rag_long_metrics.items():
        print(f"  {metric:.<25} {value:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
