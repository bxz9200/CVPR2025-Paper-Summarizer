#!/usr/bin/env python3
import os
import re
import glob
import requests
import logging
from PyPDF2 import PdfReader
from pathlib import Path
import argparse
import json
import time
from bs4 import BeautifulSoup
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='paper_summary.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def extract_title_from_filename(filename):
    """Extract a readable title from the PDF filename."""
    basename = os.path.basename(filename)
    # Remove file extension
    title = os.path.splitext(basename)[0]
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Fix common patterns for better readability
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)  # Add space between camelCase
    return title

def extract_abstract_from_pdf(pdf_path):
    """Extract the abstract section from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Usually the abstract is in the first 2-3 pages
        for i in range(min(3, len(reader.pages))):
            text += reader.pages[i].extract_text()
        
        # Look for abstract section using common patterns
        patterns = [
            r'(?i)abstract\s*\n(.*?)(?:\n\s*\n|\n\s*introduction)',  # Common format: Abstract followed by Introduction
            r'(?i)abstract[:\.\s]+(.*?)(?:\n\s*\n|\n\s*1[\.\s]+introduction)',  # Format with numbered sections
            r'(?i)abstract[:\.\s]+(.*?)(?:\n\s*\n|\n\s*keywords)',   # Format with Keywords after Abstract
            r'(?i)abstract[:\.\s]+(.*?)(?:\n\s*\n|\n\s*\d+\.)',      # Format with numbered sections
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract text
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract
                
        # If no clear abstract section found, just return the first 1000 characters as a fallback
        logging.warning(f"Couldn't identify abstract section in {pdf_path}, using first text chunk")
        return text[:1000].strip()
    
    except Exception as e:
        logging.error(f"Error extracting abstract from {pdf_path}: {str(e)}")
        return None

def summarize_abstract_with_huggingface(abstract, api_key=None, title=None):
    """Summarize the abstract using HuggingFace's free inference API."""
    try:
        # Initialize NLTK if needed for the local summarization fallback
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # First try using HuggingFace API
        if api_key:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
        else:
            # The API can be used without authentication for a limited number of requests
            headers = {}
        
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        payload = {
            "inputs": f"Title: {title}\n\nAbstract: {abstract}",
            "parameters": {
                "max_length": 250,
                "min_length": 100,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", "").strip()
            
        logging.warning(f"HuggingFace API failed with status {response.status_code}. Using fallback method.")
    except Exception as e:
        logging.warning(f"Error with HuggingFace API: {str(e)}. Using fallback method.")
    
    # Fallback: Use SMMRY web service (without API)
    try:
        # SMMRY provides a simple free summarization service
        logging.info("Trying SMMRY web service as fallback...")
        
        smmry_url = "https://smmry.com/"
        
        # Send form data
        data = {
            'sm_api_input': abstract,
            'sm_length': 3  # Number of sentences in summary
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.post(smmry_url, data=data, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            summary_div = soup.find('div', {'class': 'sm_api_content'})
            
            if summary_div:
                return summary_div.get_text().strip()
    except Exception as e:
        logging.warning(f"SMMRY web service failed: {str(e)}. Using final local fallback.")
    
    # Final fallback: Use a very simple extraction-based summarization
    try:
        logging.info("Using local extraction-based summarization as final fallback...")
        from nltk.tokenize import sent_tokenize
        
        # Split into sentences
        sentences = sent_tokenize(abstract)
        
        # If there are only a few sentences, return all of them
        if len(sentences) <= 3:
            return abstract
        
        # Simple summarization: get first sentence, middle sentence, and last sentence
        summary_sentences = [
            sentences[0],
            sentences[len(sentences) // 2],
            sentences[-1]
        ]
        
        return " ".join(summary_sentences)
    except Exception as e:
        logging.error(f"All summarization methods failed: {str(e)}")
        # Return first 100 chars if all else fails
        return abstract[:200] + "..." if len(abstract) > 200 else abstract

def save_summaries_to_markdown(summaries, output_path):
    """Save the paper summaries in a markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# AI-Generated Paper Summaries\n\n")
        f.write("*Summaries of papers in the downloaded_papers folder*\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        for paper in summaries:
            f.write(f"## {paper['title']}\n\n")
            
            if paper.get('abstract'):
                f.write("### Original Abstract\n\n")
                f.write(f"{paper['abstract']}\n\n")
            
            if paper.get('summary'):
                f.write("### Summary\n\n")
                f.write(f"{paper['summary']}\n\n")
            else:
                f.write("*No summary available*\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Saved summaries to {output_path}")

def save_summaries_to_html(summaries, output_path):
    """Save the paper summaries in HTML format."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Generated Paper Summaries</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        h3 {{
            color: #2c3e50;
        }}
        .paper-container {{
            margin-bottom: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }}
        .abstract {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            font-style: italic;
        }}
        .summary {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
        }}
        .generation-info {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <h1>AI-Generated Paper Summaries</h1>
    <p class="generation-info">Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    for paper in summaries:
        html_content += f'<div class="paper-container">\n'
        html_content += f'    <h2>{paper["title"]}</h2>\n'
        
        if paper.get('abstract'):
            html_content += f'    <h3>Original Abstract</h3>\n'
            html_content += f'    <div class="abstract">{paper["abstract"]}</div>\n'
        
        if paper.get('summary'):
            html_content += f'    <h3>Summary</h3>\n'
            html_content += f'    <div class="summary">{paper["summary"]}</div>\n'
        else:
            html_content += f'    <p><em>No summary available</em></p>\n'
        
        html_content += '</div>\n'
    
    html_content += """</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Saved HTML summaries to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and summarize abstracts from academic papers.")
    parser.add_argument("--input-dir", default="downloaded_papers", help="Directory containing PDF files")
    parser.add_argument("--output", default="paper_summaries.md", help="Output file (markdown format)")
    parser.add_argument("--html", action="store_true", help="Generate HTML output instead of markdown")
    parser.add_argument("--hf-token", help="HuggingFace API token (optional, limited requests possible without it)")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_path = args.output
    api_key = args.hf_token
    
    # Handle HTML output option
    if args.html and output_path.endswith('.md'):
        output_path = output_path.replace('.md', '.html')
    
    # Get all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return
    
    logging.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    summaries = []
    
    for pdf_file in pdf_files:
        title = extract_title_from_filename(pdf_file)
        logging.info(f"Processing: {title}")
        
        abstract = extract_abstract_from_pdf(pdf_file)
        if not abstract:
            logging.warning(f"Could not extract abstract from {pdf_file}")
            summaries.append({"title": title})
            continue
        
        logging.info(f"Extracted abstract ({len(abstract)} chars)")
        
        summary = summarize_abstract_with_huggingface(abstract, api_key, title)
        if summary:
            logging.info(f"Generated summary ({len(summary)} chars)")
        else:
            logging.warning(f"Failed to generate summary for {title}")
        
        summaries.append({
            "title": title,
            "abstract": abstract,
            "summary": summary
        })
        
        # Add a small delay to avoid API rate limits
        time.sleep(2)
    
    if args.html:
        save_summaries_to_html(summaries, output_path)
    else:
        save_summaries_to_markdown(summaries, output_path)
    
    logging.info("Summary extraction complete!")

if __name__ == "__main__":
    main()
