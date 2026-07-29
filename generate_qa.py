"""
Generate Q&A pairs (short and long) from Abstracts and Conclusions using Gemini.
This generates 100-150 questions from original paper text, not graph entities.
Output: output/qa_short.json, output/qa_long.json
"""

import os
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm
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


SHORT_QA_PROMPT = """
Given this text from a carbon capture research paper, generate a SHORT factual question and answer.

Text:
{text}

Paper ID: {paper_id}
Section: {section}

Generate a SHORT Q&A pair (answer max 20 words):
- Question should be specific and answerable from the text
- Answer should be concise and factual
- Focus on key findings, materials, properties, or methods

Return ONLY valid JSON:
{{
  "question": "What is the CO2 capacity of 3D graphene?",
  "answer": "8.9 mmol/g at 298K and 1 bar",
  "source_paper": "{paper_id}",
  "source_section": "{section}"
}}
"""


LONG_QA_PROMPT = """
Given this text from a carbon capture research paper, generate a LONG detailed question and answer.

Text:
{text}

Paper ID: {paper_id}
Section: {section}

Generate a LONG Q&A pair with STRICT LENGTH CONTROL:
- Question should invite detailed explanation
- Answer MUST be 60-90 words (count carefully!)
- Include synthesis methods, properties, performance metrics, or mechanisms
- Be comprehensive but CONCISE - no unnecessary details

IMPORTANT: Answer must be between 60-90 words. Not more, not less.

Return ONLY valid JSON:
{{
  "question": "Describe the CO2 adsorption properties and synthesis of the material.",
  "answer": "The material demonstrates excellent CO2 adsorption capabilities, achieving capacities of 8.9 mmol/g under standard conditions (298K and 1 bar). It is synthesized through thermal reduction at 800°C, creating porous architectures that provide high surface areas for gas adsorption. The performance is attributed to the material's unique structural properties and high accessibility of adsorption sites.",
  "source_paper": "{paper_id}",
  "source_section": "{section}"
}}
"""


def generate_qa_with_gemini(text, paper_id, section, answer_type="short"):
    """Generate Q&A pair from text (abstract or conclusion)."""
    try:
        # Safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        # Select prompt
        prompt_template = SHORT_QA_PROMPT if answer_type == "short" else LONG_QA_PROMPT
        prompt = prompt_template.format(
            text=text,
            paper_id=paper_id,
            section=section
        )
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512 if answer_type == "short" else 1024,
            )
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        qa_pair = json.loads(response_text)
        qa_pair["answer_type"] = answer_type
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating QA: {e}")
        return None


def generate_qa_pairs(extracted_texts, answer_type="short", max_pairs=75):
    """Generate Q&A pairs from abstracts and conclusions."""
    qa_pairs = []
    
    print(f"\nGenerating {answer_type} Q&A pairs from abstracts and conclusions...")
    
    # Collect all abstracts and conclusions
    text_sources = []
    for paper_id, paper_data in extracted_texts.items():
        # Add abstract (direct string field, not list)
        if 'abstract' in paper_data and paper_data['abstract']:
            abstract_text = paper_data['abstract']
            if len(abstract_text) > 200:  # Only meaningful abstracts
                text_sources.append({
                    'text': abstract_text,
                    'paper_id': paper_id,
                    'section': 'abstract'
                })
        
        # Add conclusion (direct string field, not list)
        if 'conclusion' in paper_data and paper_data['conclusion']:
            conclusion_text = paper_data['conclusion']
            if len(conclusion_text) > 200:  # Only meaningful conclusions
                text_sources.append({
                    'text': conclusion_text,
                    'paper_id': paper_id,
                    'section': 'conclusion'
                })
    
    print(f"Found {len(text_sources)} abstract/conclusion texts")
    
    # Generate QA pairs from text sources
    selected_sources = text_sources[:max_pairs]
    
    for source in tqdm(selected_sources, desc=f"Generating {answer_type} QA"):
        qa_pair = generate_qa_with_gemini(
            text=source['text'],
            paper_id=source['paper_id'],
            section=source['section'],
            answer_type=answer_type
        )
        
        if qa_pair:
            qa_pairs.append(qa_pair)
        
        time.sleep(0.5)  # Rate limiting
    
    return qa_pairs


def main():
    """Main function to generate Q&A pairs."""
    # Load extracted texts
    texts_path = Path("output/extracted_texts.json")
    if not texts_path.exists():
        print("❌ Error: output/extracted_texts.json not found. Run extract_text.py first.")
        return
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        extracted_texts = json.load(f)
    
    print(f"Loaded extracted texts from {len(extracted_texts)} papers")
    
    # Generate short Q&A pairs (45 questions)
    qa_short = generate_qa_pairs(extracted_texts, answer_type="short", max_pairs=45)
    
    output_short = Path("output/qa_short.json")
    with open(output_short, 'w', encoding='utf-8') as f:
        json.dump(qa_short, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(qa_short)} short Q&A pairs")
    print(f"✓ Saved to {output_short}")
    
    # Generate long Q&A pairs (30 questions)
    qa_long = generate_qa_pairs(extracted_texts, answer_type="long", max_pairs=30)
    
    output_long = Path("output/qa_long.json")
    with open(output_long, 'w', encoding='utf-8') as f:
        json.dump(qa_long, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(qa_long)} long Q&A pairs")
    print(f"✓ Saved to {output_long}")
    
    print(f"\n✓ Total questions generated: {len(qa_short) + len(qa_long)} (Target: 75)")
    
    # Show examples
    print("\nExample short Q&A:")
    if qa_short:
        print(f"Q: {qa_short[0]['question']}")
        print(f"A: {qa_short[0]['answer']}")
        print(f"Source: {qa_short[0].get('source_paper', 'N/A')} - {qa_short[0].get('source_section', 'N/A')}")
    
    print("\nExample long Q&A:")
    if qa_long:
        print(f"Q: {qa_long[0]['question']}")
        print(f"A: {qa_long[0]['answer'][:150]}...")
        print(f"Source: {qa_long[0].get('source_paper', 'N/A')} - {qa_long[0].get('source_section', 'N/A')}")


if __name__ == "__main__":
    main()
