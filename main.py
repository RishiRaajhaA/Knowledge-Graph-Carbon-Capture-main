"""
Main pipeline runner - executes all steps sequentially.
Steps: Extract Text → Build Graph → Generate Q&A → Evaluate
"""

import os
import sys
import time
from pathlib import Path


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def check_prerequisites():
    """Check if prerequisites are met."""
    print_header("Checking Prerequisites")
    
    # Check for raw folder
    raw_dir = Path("raw")
    if not raw_dir.exists():
        print("❌ Error: 'raw/' folder not found")
        return False
    
    pdf_files = list(raw_dir.glob("*.pdf"))
    if len(pdf_files) == 0:
        print("❌ Error: No PDF files found in 'raw/' folder")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files in 'raw/' folder")
    
    # Check for .env file
    if not Path(".env").exists():
        print("❌ Error: .env file not found")
        print("   Please create .env file with: GEMINI_API_KEY=your_key_here")
        return False
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set in .env file")
        return False
    
    print("✓ Environment configured correctly")
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    print("✓ Output directory ready")
    
    return True


def run_step(step_name, script_name):
    """Run a pipeline step."""
    print_header(f"Step: {step_name}")
    start_time = time.time()
    
    # Import and run the script
    try:
        if script_name == "extract_text":
            import extract_text
            extract_text.main()
        elif script_name == "build_graph":
            import build_graph
            build_graph.main()
        elif script_name == "generate_qa":
            import generate_qa
            generate_qa.main()
        elif script_name == "evaluate":
            import evaluate
            evaluate.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print pipeline summary."""
    print_header("Pipeline Summary")
    
    output_files = {
        "Extracted Texts": "output/extracted_texts.json",
        "Knowledge Graph": "output/knowledge_graph.json",
        "Short Q&A Pairs": "output/qa_short.json",
        "Long Q&A Pairs": "output/qa_long.json",
        "Evaluations": "output/evaluations.json"
    }
    
    print("Generated files:")
    for name, path in output_files.items():
        if Path(path).exists():
            size = Path(path).stat().st_size
            print(f"  ✓ {name:.<30} {path} ({size:,} bytes)")
        else:
            print(f"  ✗ {name:.<30} {path} (not found)")
    
    # Load and display evaluation results
    eval_path = Path("output/evaluations.json")
    if eval_path.exists():
        import json
        with open(eval_path, 'r') as f:
            results = json.load(f)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print("\nPre-RAG Short (Retrieval):")
        for metric, value in results['pre_rag_short'].items():
            print(f"  {metric:.<25} {value:.3f}")
        
        print("\nPre-RAG Long (Retrieval):")
        for metric, value in results['pre_rag_long'].items():
            print(f"  {metric:.<25} {value:.3f}")
        
        print("\nPost-RAG Short (Generation):")
        for metric, value in results['post_rag_short'].items():
            print(f"  {metric:.<25} {value:.3f}")
        
        print("\nPost-RAG Long (Generation):")
        for metric, value in results['post_rag_long'].items():
            print(f"  {metric:.<25} {value:.3f}")


def main():
    """Run complete pipeline."""
    print("\n")
    print("=" + "="*68 + "=")
    print("|" + " "*10 + "Graph-based RAG Pipeline for Carbon Capture" + " "*15 + "|")
    print("=" + "="*68 + "=")
    
    pipeline_start = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Pipeline aborted due to missing prerequisites")
        return
    
    # Step 1: Extract text from PDFs
    if not run_step("Extract Text from PDFs", "extract_text"):
        return
    
    # Step 2: Build knowledge graph
    if not run_step("Build Knowledge Graph", "build_graph"):
        return
    
    # Step 3: Generate Q&A pairs
    if not run_step("Generate Q&A Pairs", "generate_qa"):
        return
    
    # Step 4: Evaluate
    if not run_step("Evaluate System", "evaluate"):
        return
    
    # Print summary
    print_summary()
    
    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"✓ Pipeline completed successfully in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
