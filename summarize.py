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
from typing import List, Dict, Any, Optional, Tuple
import textwrap
import difflib

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

# Maximum abstract length for a single API call (to avoid token limits)
MAX_ABSTRACT_LENGTH = 1500
# Maximum summary length (in characters) for a single chunk
MAX_SUMMARY_LENGTH = 250

def extract_title_from_filename(filename):
    """Extract a readable title from the PDF filename."""
    basename = os.path.basename(filename)
    # Remove file extension
    title = os.path.splitext(basename)[0]
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Fix common patterns for better readability
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)  # Add space between camelCase
    # Remove arxiv ID if present (often appended after an underscore)
    title = re.sub(r'_\d+\.\d+.*$', '', title)
    return title

def extract_abstract_from_pdf(pdf_path):
    """Extract the abstract section from a PDF file with improved accuracy."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Usually the abstract is in the first 2-3 pages
        for i in range(min(3, len(reader.pages))):
            text += reader.pages[i].extract_text()
        
        # Try to extract abstract from document metadata first
        try:
            info = reader.metadata
            if info and hasattr(info, 'subject') and info.subject and len(info.subject) > 100:
                # Some PDFs store the abstract in metadata
                abstract_candidate = info.subject
                cleaned = preprocess_abstract(abstract_candidate)
                if validate_abstract(cleaned):
                    logging.info(f"Using abstract from PDF metadata for {pdf_path}")
                    return cleaned
        except Exception as e:
            logging.debug(f"Error reading PDF metadata: {str(e)}")
        
        # Precompile regex patterns for better performance and readability
        abstract_section = re.compile(r'\bABSTRACT\b|\babstract\b', re.IGNORECASE)
        intro_section = re.compile(r'\b(INTRODUCTION|Introduction|1\.(\s+|)INTRODUCTION|I\.\s+INTRODUCTION)\b')
        section_headers = re.compile(r'(\n\s*\d{1,2}(\.\d{1,2})?[\.\s]+[A-Z][a-zA-Z\s]+|\n\s*[A-Z][A-Z\s]+\n)', re.MULTILINE)
        
        # Check for explicit abstract heading in the text
        try:
            abstract_match = re.search(r'(?i)(^|\n)\s*abstract\s*[:\.—-]?\s*\n+(.+?)(?=\n\s*\d?\.?\s*Introduction|\n\s*\d\.|$)', text, re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(2).strip()
                if 100 <= len(abstract_text) <= 3000:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
        except Exception as e:
            logging.debug(f"Error in primary abstract extraction: {str(e)}")
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Method 1: Find abstract section with explicit "Abstract" header
        # This is common in most academic papers
        abstract_found = False
        abstract_text = ""
        
        for i, para in enumerate(paragraphs):
            # Look for a paragraph that contains only the word "Abstract"
            if abstract_section.match(para) and len(para) < 30:
                abstract_found = True
                # The abstract is likely the next paragraph
                if i + 1 < len(paragraphs):
                    next_para = paragraphs[i + 1]
                    # Make sure it's not a section heading
                    if not intro_section.match(next_para) and len(next_para) > 100:
                        cleaned = preprocess_abstract(next_para)
                        if validate_abstract(cleaned):
                            return cleaned
            # Look for paragraph starting with "Abstract" followed by content
            elif abstract_section.match(para) and len(para) >= 30:
                # Extract everything after "Abstract"
                match = abstract_section.search(para)
                start_pos = match.end()
                abstract_text = para[start_pos:].strip()
                if len(abstract_text) > 100:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
        
        # Method 2: Find abstract by looking at text structure
        # Look for text between title/authors and introduction
        # First, try to find the introduction section
        intro_index = None
        for i, para in enumerate(paragraphs):
            if intro_section.match(para):
                intro_index = i
                break
        
        if intro_index is not None and intro_index > 0:
            # Look for the abstract before the introduction
            # We'll check the 3 paragraphs before the introduction
            for i in range(max(0, intro_index - 3), intro_index):
                # Potential abstract should be substantial and not contain author info
                if (len(paragraphs[i]) > 100 and 
                    not re.search(r'@|\buniversity\b|\binstitute\b|professor|faculty', paragraphs[i].lower()) and
                    not re.match(r'^\d+\.', paragraphs[i])):
                    cleaned = preprocess_abstract(paragraphs[i])
                    if validate_abstract(cleaned):
                        return cleaned
        
        # Method 3: ArXiv-style papers often have a specific structure
        # Title, authors, abstract, then sections
        # First, find potential title (short first paragraph)
        title_idx = None
        for i, para in enumerate(paragraphs[:3]):
            if 10 <= len(para) <= 150 and not re.search(r'@|abstract', para.lower()):
                title_idx = i
                break
        
        if title_idx is not None:
            # Skip author information (1-2 paragraphs after title)
            # Authors section typically contains emails, affiliations
            potential_abstract_idx = title_idx + 1
            while potential_abstract_idx < len(paragraphs):
                para = paragraphs[potential_abstract_idx]
                if re.search(r'@|\buniversity\b|\binstitute\b|department', para.lower()) or len(para) < 100:
                    potential_abstract_idx += 1
                else:
                    break
            
            if potential_abstract_idx < len(paragraphs):
                # Check if this paragraph has abstract-like content
                abstract_candidate = paragraphs[potential_abstract_idx]
                # Abstract should be substantial, not have section numbering, and come before introduction
                if (100 <= len(abstract_candidate) <= 3000 and 
                    not re.match(r'^\d+[\.\s]', abstract_candidate) and
                    not intro_section.match(abstract_candidate)):
                    
                    # Extra check: if the text contains "abstract:" at the start, remove it
                    abstract_candidate = re.sub(r'^abstract[\s\.:]+', '', abstract_candidate, flags=re.IGNORECASE).strip()
                    cleaned = preprocess_abstract(abstract_candidate)
                    if validate_abstract(cleaned):
                        return cleaned
        
        # Method 4: Look for content positioned like an abstract (after title, before sections)
        # Find all section headers in the text
        section_matches = list(section_headers.finditer(text))
        
        if section_matches and len(section_matches) >= 2:
            # Assume first section is after abstract
            first_section_pos = section_matches[0].start()
            # Look for text before the first section
            start_text = text[:first_section_pos].strip()
            # Split into paragraphs
            early_paragraphs = re.split(r'\n\s*\n', start_text)
            early_paragraphs = [p.strip() for p in early_paragraphs if p.strip()]
            
            # Skip title and authors, find first substantial paragraph
            for para in early_paragraphs:
                if (len(para) >= 200 and 
                    not re.search(r'@|\buniversity\b|\binstitute\b', para.lower()) and
                    not re.match(r'^\d+\.', para)):
                    cleaned = preprocess_abstract(para)
                    if validate_abstract(cleaned):
                        return cleaned
        
        # Method 5: Multi-paragraph abstract handling
        # Some papers have abstracts split into multiple paragraphs
        # Look for paragraphs that seem to continue from an abstract
        for i, para in enumerate(paragraphs):
            if abstract_section.search(para):
                # Check if this contains "Abstract:" followed by text
                abstract_pos = abstract_section.search(para).end()
                start_abstract = para[abstract_pos:].strip()
                
                # If this paragraph continues to the next, collect them
                if start_abstract and i + 1 < len(paragraphs):
                    # Check next paragraph - if it's short and doesn't look like a section header
                    next_para = paragraphs[i + 1]
                    if (len(next_para) < 300 and 
                        not intro_section.match(next_para) and
                        not re.match(r'^\d+\.', next_para) and
                        not re.search(r'keywords:|index terms:', next_para, re.IGNORECASE)):
                        
                        # This might be a continuation of the abstract
                        combined_abstract = start_abstract + " " + next_para
                        if len(combined_abstract) >= 200:
                            cleaned = preprocess_abstract(combined_abstract)
                            if validate_abstract(cleaned):
                                return cleaned
        
        # Method 6: CVPR-style papers often have abstract at the top
        # Try another format common in conference papers
        try:
            cvpr_abstract = re.search(r'(?i)(?<=\n\s*)\S.{100,2000}?(?=\n\s*\d?\.?\s*introduction|\n\s*\d\.\s)', text)
            if cvpr_abstract:
                abstract_text = cvpr_abstract.group(0).strip()
                if 150 <= len(abstract_text) <= 3000:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
        except Exception as e:
            logging.debug(f"Error in CVPR-style abstract extraction: {str(e)}")
        
        # Method 7: Generic abstract detection approach
        # Just find the first substantial paragraph that's not the title, authors, or a section heading
        logging.warning(f"Using generic abstract detection for {pdf_path}")
        for para in paragraphs[:15]:  # Check first 15 paragraphs only
            if (200 <= len(para) <= 3000 and 
                not re.search(r'@|\buniversity\b|\binstitute\b|department|figure|fig\.|table', para.lower()) and
                not re.match(r'^\d+\.', para) and 
                not intro_section.match(para) and
                abstract_section.search(para) is None):  # Not the "Abstract" label alone
                cleaned = preprocess_abstract(para)
                if validate_abstract(cleaned):
                    return cleaned
        
        # Fallback: If all else fails, just get first substantial paragraph
        logging.warning(f"Couldn't identify abstract section in {pdf_path}, using first substantial paragraph")
        for para in paragraphs:
            if len(para.strip()) >= 200 and len(para.strip()) <= 3000:
                cleaned = preprocess_abstract(para.strip())
                return cleaned
        
        return preprocess_abstract(text[:1000].strip())
    
    except Exception as e:
        logging.error(f"Error extracting abstract from {pdf_path}: {str(e)}")
        return None

def preprocess_abstract(abstract: str) -> str:
    """Clean and preprocess the abstract for better summarization quality."""
    if not abstract:
        return ""
        
    # Remove extra whitespace and line breaks
    cleaned = re.sub(r'\s+', ' ', abstract).strip()
    
    # Fix hyphenation in words that were split across lines in the PDF
    # This matches common patterns like "pro- pose" or "vulner- ability"
    cleaned = re.sub(r'(\w+)-\s+(\w+)', lambda m: m.group(1) + m.group(2), cleaned)
    
    # Fix common compound adjectives that should have spaces
    # Pattern: lowercase + uppercase or lowercase + digit
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    
    # Fix missing spaces between common compound words in PDF extraction
    common_compounds = [
        r'(high)(quality)', r'(state)(of)(the)(art)', r'(fine)(tuning)', 
        r'(pre)(trained)', r'(real)(world)', r'(end)(to)(end)', r'(open)(source)'
    ]
    for pattern in common_compounds:
        cleaned = re.sub(pattern, lambda m: ' '.join(m.groups()), cleaned, flags=re.IGNORECASE)
    
    # Fix hyphenation without spaces
    cleaned = re.sub(r'(\w+)-(\w+)distribution', r'\1-\2-distribution', cleaned)
    cleaned = re.sub(r'([a-z])-of([a-z])', r'\1-of-\2', cleaned)
    
    # Remove common artifacts
    cleaned = re.sub(r'et\s+al\.', 'et al.', cleaned)  # Fix "et al." spacing
    cleaned = re.sub(r'\s+\(\d{4}\)', '', cleaned)  # Remove citation years
    cleaned = re.sub(r'\[\d+\]', '', cleaned)  # Remove citation brackets
    cleaned = re.sub(r'\s+,', ',', cleaned)  # Fix comma spacing
    
    # Remove very long URLs that might waste tokens
    cleaned = re.sub(r'https?://\S+', '[URL]', cleaned)
    
    # Remove author listings, affiliations, and emails
    cleaned = re.sub(r'\d+University\s+of\s+[A-Za-z\s,]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?:\w+\d+[,†\*]*[,\s]*)+(?:University|Institute|College|Corporation)', '', cleaned)
    cleaned = re.sub(r'\w+@\w+\.\w+', '', cleaned)  # Remove emails
    cleaned = re.sub(r'\d+[,†\*][,\s]*', '', cleaned)  # Remove author references like "1,2,"
    
    # Remove section numbering and labels
    cleaned = re.sub(r'^\s*\d+\.\s+\w+', '', cleaned)
    cleaned = re.sub(r'^\s*abstract\s*[:\.]?\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Remove line/page numbers
    cleaned = re.sub(r'\s+\d+\s*$', '', cleaned)
    cleaned = re.sub(r'^\s*\d+\s+', '', cleaned)
    
    # Remove specific paper artifacts
    cleaned = re.sub(r'(?i)keywords[:\.]\s*.*?(?=\n|$)', '', cleaned)  # Remove keywords list if included
    cleaned = re.sub(r'(?i)index terms[:\.]\s*.*?(?=\n|$)', '', cleaned)  # Remove index terms
    
    # Remove footnotes, acknowledgements, and other non-abstract elements
    cleaned = re.sub(r'^\s*\d+\s*This work was supported by.*?(?=\n|$)', '', cleaned)
    cleaned = re.sub(r'(?i)(\n|^)\s*(acknowledgements?|funding)[:\.](.*?)(?=\n|$)', '', cleaned)
    cleaned = re.sub(r'\*\s*Corresponding author.*?(?=\n|$)', '', cleaned)
    cleaned = re.sub(r'(?i)equal contribution.*?(?=\n|$)', '', cleaned)
    
    # Remove paper/preprint information
    cleaned = re.sub(r'arXiv:\d+\.\d+v\d+', '', cleaned)
    cleaned = re.sub(r'\d{4}\.\d{5}v\d+', '', cleaned)
    
    # Remove any leading numbers or bullet points
    cleaned = re.sub(r'^\s*[\d\.\*\-•◦‣⁃]+\s+', '', cleaned)
    
    # Check if abstract starts with title-like text (often happens with arXiv papers)
    # If so, attempt to skip to actual abstract content
    title_and_abstract = re.match(r'^(.{20,150})\s+Abstract\s+(.+)$', cleaned, re.IGNORECASE)
    if title_and_abstract:
        # Skip the title and "Abstract" label, just keep abstract content
        cleaned = title_and_abstract.group(2)
    
    # Remove "Project page" mentions common in abstracts
    cleaned = re.sub(r'(?i)project page (is|at|available).*?(\.|$)', '', cleaned)
    
    # Final clean-up of any remaining artifacts
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^\s*abstract\s*[:\.]?\s*', '', cleaned, flags=re.IGNORECASE)  # Remove "Abstract:" again if still present
    
    return cleaned

def validate_abstract(abstract: str) -> bool:
    """
    Validate that the text is likely to be an abstract.
    
    Args:
        abstract: The text to validate
        
    Returns:
        True if the text appears to be an abstract, False otherwise
    """
    if not abstract:
        return False
        
    # Check length (abstracts are typically 150-500 words)
    word_count = len(abstract.split())
    if word_count < 30 or word_count > 600:
        logging.warning(f"Abstract has unusual length: {word_count} words")
        if word_count < 20 or word_count > 1000:
            return False
    
    # Check for suspicious content that suggests it's not an abstract
    suspicious_patterns = [
        r'(?i)fig(?:ure|\.)\s+\d{1,2}',        # Figure references
        r'(?i)table\s+\d{1,2}',                # Table references
        r'(?i)equation\s+\d{1,2}',             # Equation references
        r'(?i)section\s+\d{1,2}',              # Section references
        r'(?i)algorithm\s+\d{1,2}',            # Algorithm references
        r'(?i)\d{1,2}\.\s+experiments',        # Numbered sections
        r'(?i)\d{1,2}\.\s+introduction',       # Numbered sections
        r'(?i)\d{1,2}\.\s+related work',       # Numbered sections
        r'(?i)\d{1,2}\.\s+conclusion',         # Numbered sections
        r'(?i)submitted to',                  # Submission notes
        r'(?i)camera.?ready',                  # Camera ready notes
        r'(?i)under review',                   # Review notes
    ]
    
    suspicion_count = 0
    for pattern in suspicious_patterns:
        if re.search(pattern, abstract):
            suspicion_count += 1
            logging.debug(f"Suspicious pattern found: {pattern}")
            
    if suspicion_count >= 3:
        logging.warning(f"Text contains {suspicion_count} suspicious patterns, may not be an abstract")
        return False
        
    # Check for abstract-like beginnings
    # Abstracts often start with phrases about the paper's contributions
    abstract_beginning_patterns = [
        r'(?i)^(in|this) (paper|work|article|study)',
        r'(?i)^we (present|propose|introduce|describe|develop)',
        r'(?i)^(recent|current) (advances|developments|research|studies|work)',
        r'(?i)^this (paper|work|research) (presents|proposes|describes|focuses)',
        r'(?i)^(vision(-|\s)language|VL) models',
        r'(?i)^large (language|vision|multimodal) models',
    ]
    
    has_abstract_beginning = any(re.search(pattern, abstract[:100]) for pattern in abstract_beginning_patterns)
    
    # Check for abstract-like content (presence of keywords related to contributions)
    contribution_keywords = [
        r'propose', r'present', r'introduce', r'novel', r'approach', r'method',
        r'contribution', r'demonstrate', r'show', r'performance', r'experiments',
        r'results', r'outperform', r'state-of-the-art', r'state of the art'
    ]
    
    has_contribution_terms = sum(1 for keyword in contribution_keywords if re.search(r'\b' + keyword + r'\b', abstract.lower())) >= 2
    
    # Return overall assessment
    if has_abstract_beginning:
        return True
    elif has_contribution_terms and suspicion_count < 2:
        return True
    elif suspicion_count < 1 and 50 <= word_count <= 500:
        return True
    else:
        return False

def extract_and_validate_abstract(pdf_path):
    """
    Extract abstract from PDF and validate that it's actually an abstract.
    If not valid, try alternative extraction methods.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Validated abstract or None if extraction failed
    """
    # First extraction attempt
    abstract = extract_abstract_from_pdf(pdf_path)
    
    if not abstract:
        logging.warning(f"Failed to extract abstract from {pdf_path}")
        return None
        
    # Clean the abstract
    cleaned_abstract = preprocess_abstract(abstract)
    
    # Validate the abstract
    if validate_abstract(cleaned_abstract):
        logging.info(f"Successfully validated abstract from {pdf_path}")
        return cleaned_abstract
        
    # If validation failed, try a more aggressive approach
    logging.warning(f"First extracted abstract failed validation for {pdf_path}, trying alternative method")
    
    try:
        # Get raw text from the PDF - focusing on just first page
        reader = PdfReader(pdf_path)
        first_page_text = reader.pages[0].extract_text()
        
        # Sometimes we need to check second page too (if abstract continues)
        if len(reader.pages) > 1:
            second_page_text = reader.pages[1].extract_text()
            
            # Check if abstract might continue on second page
            # Look for typical start of introduction on second page
            if re.search(r'\b(INTRODUCTION|Introduction|1\.(\s+|)INTRODUCTION|I\.\s+INTRODUCTION)\b', second_page_text):
                # Get text from second page up to introduction
                intro_match = re.search(r'\b(INTRODUCTION|Introduction|1\.(\s+|)INTRODUCTION|I\.\s+INTRODUCTION)\b', second_page_text)
                if intro_match:
                    continuation = second_page_text[:intro_match.start()].strip()
                    if len(continuation) > 100 and len(continuation) < 1000:
                        # This might be continuation of abstract from first page
                        # Try to find a good ending point in first page text
                        end_of_first_text = first_page_text.strip()
                        # Combine the two parts
                        combined_text = end_of_first_text + " " + continuation
                        # Try to extract from this combined text
                        combined_paragraphs = re.split(r'\n\s*\n', combined_text)
                        for para in combined_paragraphs:
                            if len(para) > 200 and "abstract" in para.lower():
                                cleaned_para = preprocess_abstract(para)
                                if validate_abstract(cleaned_para):
                                    return cleaned_para
        
        # Try a structural approach based on page layout
        # Look for lines that contain "Abstract" explicitly
        lines = first_page_text.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'\babstract\b', line.lower()):
                # This line contains the abstract heading
                # Collect text after this line until a likely section heading
                abstract_text = []
                j = i + 1
                while j < len(lines) and not re.search(r'^\d+\.\s+\w+|^introduction|^keywords', lines[j].lower()):
                    if len(lines[j].strip()) > 0:
                        abstract_text.append(lines[j])
                    j += 1
                
                if abstract_text:
                    candidate = ' '.join(abstract_text)
                    if len(candidate) >= 150:
                        cleaned_candidate = preprocess_abstract(candidate)
                        if validate_abstract(cleaned_candidate):
                            return cleaned_candidate
        
        # Fallback: Extract substantial paragraph after title but before introduction
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', first_page_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Find likely title (short paragraph at the start, not containing emails)
        title_idx = None
        for i, para in enumerate(paragraphs[:3]):
            if 20 <= len(para) <= 200 and not re.search(r'@|abstract|introduction', para.lower()):
                title_idx = i
                break
        
        if title_idx is not None:
            # Skip author information (1-3 paragraphs with emails, affiliations)
            author_end = title_idx + 1
            while author_end < len(paragraphs) and (
                    re.search(r'@|\d{4}|university|institute|college', paragraphs[author_end].lower()) or 
                    len(paragraphs[author_end]) < 100):
                author_end += 1
            
            # Check next substantial paragraph - likely to be abstract
            if author_end < len(paragraphs):
                candidate = paragraphs[author_end]
                if len(candidate) >= 150:
                    cleaned_candidate = preprocess_abstract(candidate)
                    return cleaned_candidate
        
        # Last resort: return the original abstract even though it failed validation
        logging.warning(f"All extraction attempts failed for {pdf_path}, using best guess")
        return cleaned_abstract
        
    except Exception as e:
        logging.error(f"Error in alternative abstract extraction for {pdf_path}: {str(e)}")
        # Return the original cleaned abstract as fallback
        return cleaned_abstract

def split_abstract_into_chunks(abstract: str, max_length: int = MAX_ABSTRACT_LENGTH) -> List[str]:
    """
    Split a long abstract into manageable chunks for summarization.
    
    Args:
        abstract: The abstract text to split
        max_length: Maximum length of each chunk in characters
        
    Returns:
        List of abstract chunks
    """
    if not abstract or len(abstract) <= max_length:
        return [abstract]
        
    # Initialize NLTK for sentence tokenization
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Split by sentences to ensure we don't cut in the middle of a sentence
    sentences = nltk.tokenize.sent_tokenize(abstract)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def summarize_with_huggingface(text: str, api_key: Optional[str] = None) -> Optional[str]:
    """Call the Hugging Face API to summarize text with improved prompt engineering."""
    try:
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            # The API can be used without authentication for a limited number of requests
            headers = {}
        
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        # Enhanced prompt with method-focused instructions for better summarization
        enhanced_prompt = (
            "Summarize the following academic paper abstract into a coherent, logical paragraph "
            "that captures the paper's main contributions, with special emphasis on the proposed method or approach. "
            "Make sure to clearly explain WHAT method is proposed and HOW it works. "
            "Include key technical details about the methodology while maintaining readability. "
            "Also include significant results that validate the method's effectiveness.\n\n"
            f"{text}"
        )
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "max_length": MAX_SUMMARY_LENGTH,
                "min_length": min(100, MAX_SUMMARY_LENGTH//2),
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "").strip()
                # Clean up the summary to remove any mention of "This paper" at the beginning
                summary = re.sub(r'^(This paper|The paper|This abstract|The abstract)\s+', '', summary)
                return summary
        
        logging.warning(f"HuggingFace API returned status code {response.status_code}")
        return None
        
    except Exception as e:
        logging.warning(f"Error with HuggingFace API: {str(e)}")
        return None

def summarize_with_smmry(text: str) -> Optional[str]:
    """Use SMMRY web service to summarize text."""
    try:
        logging.info("Trying SMMRY web service...")
        
        smmry_url = "https://smmry.com/"
        
        data = {
            'sm_api_input': text,
            'sm_length': 3  # Number of sentences in summary
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.post(smmry_url, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            summary_div = soup.find('div', {'class': 'sm_api_content'})
            
            if summary_div:
                return summary_div.get_text().strip()
        
        return None
        
    except Exception as e:
        logging.warning(f"SMMRY web service failed: {str(e)}")
        return None

def extract_key_sentences(text: str, num_sentences: int = 3) -> str:
    """
    Extract key sentences from text using a sophisticated extractive approach 
    that considers sentence importance, content relevance, and logical flow.
    Gives special priority to sentences describing proposed methods or approaches.
    Generates a coherent summary by selecting the most informative sentences
    while maintaining the original ordering for readability.
    """
    try:
        # Ensure NLTK resources are available
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # If there are only a few sentences, return all of them
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on multiple criteria
        scored_sentences = []
        
        # Common academic terms indicating importance in research papers
        importance_words = [
            'novel', 'propose', 'present', 'introduce', 'develop', 'demonstrate',
            'contribute', 'important', 'significant', 'outperform', 'improve',
            'result', 'experiment', 'evaluate', 'show', 'achieve', 'state-of-the-art',
            'approach', 'method', 'framework', 'system', 'model', 'algorithm',
            'performance', 'accuracy', 'effectiveness', 'contribution', 'challenge', 
            'solution', 'implementation', 'analysis', 'benchmark', 'efficacy',
            'advancement', 'technique', 'innovative', 'empirically', 'validation'
        ]
        
        # Method-specific terms that strongly indicate description of methodology
        method_terms = [
            'method', 'approach', 'algorithm', 'technique', 'framework', 'architecture',
            'model', 'system', 'procedure', 'process', 'mechanism', 'strategy', 
            'paradigm', 'pipeline', 'formulation', 'formulate', 'propose', 'introduce', 
            'develop', 'design', 'construct', 'create', 'implement', 'leverage',
            'utilize', 'employ', 'based on', 'consists of', 'comprises', 'composed of',
            'presented', 'introduced', 'architecture', 'network', 'module', 'component'
        ]
        
        # Additional weight for sentences with specific content markers
        key_phrase_multiplier = 1.5
        method_phrase_multiplier = 2.0  # Higher multiplier for method descriptions
        key_phrases = [
            'we propose', 'this paper', 'our approach', 'our method', 
            'experimental results', 'we present', 'we introduce',
            'our experiments show', 'results demonstrate', 'we demonstrate',
            'our contribution', 'we achieve', 'we evaluate'
        ]
        
        # Phrases strongly indicating a method description (higher priority)
        method_phrases = [
            'we propose a', 'we introduce a', 'our method', 'our approach',
            'our framework', 'our system', 'our model', 'we develop a',
            'we present a', 'our proposed', 'we design a', 'we implement',
            'our technique', 'our algorithm', 'our architecture', 'our solution',
            'we formulate', 'the proposed method', 'the proposed approach',
            'the key idea', 'the main idea', 'the core of', 'at the heart of'
        ]
        
        total_words = sum(len(s.split()) for s in sentences)
        
        for i, sentence in enumerate(sentences):
            # Position-based score (first, second, and last sentences are important)
            if i == 0:  # First sentence usually introduces the problem/approach
                position_score = 3.0
            elif i == 1:  # Second sentence often provides context
                position_score = 2.0
            elif i == len(sentences) - 1:  # Last sentence often states conclusions
                position_score = 2.5
            elif i < len(sentences) / 3:  # Earlier sentences tend to be more important in abstracts
                position_score = 1.5
            else:
                position_score = 1.0
            
            # Content-based score
            lower_sentence = sentence.lower()
            
            # Count important terms
            term_count = sum(1 for word in importance_words if word.lower() in lower_sentence)
            content_score = (term_count / len(importance_words)) * 2.0
            
            # Count method-specific terms (weighted more heavily)
            method_term_count = sum(1 for term in method_terms if term.lower() in lower_sentence)
            method_score = (method_term_count / len(method_terms)) * 3.0  # Higher weight for method terms
            
            # Check for key phrases that strongly indicate important content
            phrase_multiplier = key_phrase_multiplier if any(phrase in lower_sentence for phrase in key_phrases) else 1.0
            
            # Check for method phrases that strongly indicate method descriptions
            if any(phrase in lower_sentence for phrase in method_phrases):
                phrase_multiplier = method_phrase_multiplier  # Higher priority for method descriptions
            
            # Length score - prefer sentences with substantial but not excessive information
            words = sentence.split()
            word_count = len(words)
            
            # Calculate ratio of this sentence's length to average sentence length
            sentence_length_ratio = word_count / (total_words / len(sentences))
            
            if 0.8 <= sentence_length_ratio <= 1.5:
                # Ideal length - neither too short nor too long compared to average
                length_score = 1.0
            elif sentence_length_ratio < 0.8:
                # Too short - may lack information
                length_score = sentence_length_ratio
            else:
                # Too long - may be too complex or contain too much information
                length_score = 1.5 / sentence_length_ratio
            
            # Information density score - sentences with numbers, percentages, measurements
            # often contain important results
            info_density_score = 0.0
            if re.search(r'\d+(?:\.\d+)?%|\d+\.\d+|accuracy|performance|results?|achieves?', lower_sentence):
                info_density_score = 1.0
            
            # Method description detection score
            # Look for structural indicators of a method description
            method_description_score = 0.0
            if (re.search(r'consists? of|contains?|includes?|based on|composed of|comprised of', lower_sentence) or
                re.search(r'implemented (using|with|via)|designed to|structured as', lower_sentence)):
                method_description_score = 2.0
            
            # Combined score with weighted factors
            total_score = (
                (position_score * 1.5) + 
                (content_score * 1.5) + 
                (method_score * 2.5) +  # Higher weight for method-related content
                (length_score * 1.0) +
                (info_density_score * 1.5) +
                (method_description_score * 2.0)
            ) * phrase_multiplier
            
            scored_sentences.append((i, sentence, total_score))
        
        # Sort sentences by score in descending order
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Ensure we have at least one sentence about methodology
        # If none of the top sentences describe methodology, replace the lowest scoring
        # sentence with the highest scoring methodology sentence
        method_included = False
        for _, sentence, _ in scored_sentences[:num_sentences]:
            if any(phrase in sentence.lower() for phrase in method_phrases):
                method_included = True
                break
        
        if not method_included:
            # Find the highest scoring sentence with method description
            for i, (idx, sentence, score) in enumerate(scored_sentences):
                if i >= num_sentences and any(phrase in sentence.lower() for phrase in method_phrases):
                    # Replace the lowest scoring sentence in our selection
                    scored_sentences[num_sentences-1] = (idx, sentence, score)
                    break
        
        # Get the top N highest-scoring sentences
        selected_sentences = scored_sentences[:num_sentences]
        
        # Sort selected sentences by their original position to maintain logical flow
        selected_sentences.sort(key=lambda x: x[0])
        
        # Join the selected sentences back together
        result = ' '.join(s[1] for s in selected_sentences)
        return result
        
    except Exception as e:
        logging.warning(f"Error in extractive summarization: {str(e)}")
        # Fallback to a simpler approach if advanced extractive summarization fails
        sentences = text.split('. ')
        if len(sentences) <= num_sentences:
            return text
            
        # Simple selection of first, last, and evenly distributed sentences
        selected = [sentences[0]]
        if num_sentences > 2:
            step = len(sentences) // (num_sentences - 2)
            for i in range(1, num_sentences - 1):
                idx = min(i * step, len(sentences) - 2)
                selected.append(sentences[idx])
        if len(sentences) > 1:
            selected.append(sentences[-1])
            
        return '. '.join(selected) + ('.' if not selected[-1].endswith('.') else '')

def keywords_from_abstract(abstract: str, num_keywords: int = 5) -> List[str]:
    """Extract potential keywords from the abstract."""
    try:
        # Try to use NLTK's parts of speech tagging
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
        
        from nltk import pos_tag, word_tokenize
        
        # Tokenize and tag words
        tokens = word_tokenize(abstract.lower())
        tagged = pos_tag(tokens)
        
        # Keep only nouns and adjectives as potential keywords
        keywords = []
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'to', 'for', 'of', 'with', 'on', 'at', 'from', 'by'])
        
        for word, tag in tagged:
            if tag.startswith('NN') and len(word) > 3 and word not in stopwords:  # Nouns
                keywords.append(word)
            elif tag.startswith('JJ') and len(word) > 4 and word not in stopwords:  # Adjectives
                keywords.append(word)
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Return most common keywords
        return [word for word, count in word_counts.most_common(num_keywords)]
        
    except Exception as e:
        logging.warning(f"Error extracting keywords: {str(e)}")
        return []

def post_process_summary(summary: str) -> str:
    """
    Enhance the readability and coherence of generated summaries by performing various 
    post-processing operations. Pays special attention to preserving method descriptions.
    
    Args:
        summary: Generated summary text to improve
        
    Returns:
        Enhanced summary with improved readability and coherence
    """
    if not summary:
        return summary
    
    # Clean up spacing and punctuation
    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'\s([.,;:!?])', r'\1', summary)
    
    # Method-related phrases that should not be removed even if they seem redundant
    method_phrases = [
        'proposed method', 'proposed approach', 'introduces', 'presents', 
        'proposes', 'develops', 'framework', 'architecture', 'technique', 
        'algorithm', 'methodology', 'we propose', 'our method', 'our approach'
    ]
    
    # Remove repetitive sentences by checking for high similarity
    # But preserve method descriptions
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    if len(sentences) > 1:
        unique_sentences = []
        for i, sentence in enumerate(sentences):
            is_duplicate = False
            is_method_sentence = False
            
            # Check if this sentence contains method-related content that should be preserved
            if any(phrase in sentence.lower() for phrase in method_phrases):
                is_method_sentence = True
            
            # Check similarity with previous sentences
            for prev_sent in unique_sentences:
                similarity = difflib.SequenceMatcher(None, sentence.lower(), prev_sent.lower()).ratio()
                if similarity > 0.7:  # High similarity threshold
                    is_duplicate = True
                    break
            
            # Keep sentence if it's not a duplicate or if it contains method info despite being similar
            if not is_duplicate or is_method_sentence:
                unique_sentences.append(sentence)
        
        summary = ' '.join(unique_sentences)
    
    # Ensure the summary doesn't end abruptly
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    # Fix common citation patterns that interrupt the flow
    summary = re.sub(r'\s*\[\d+\]\s*', ' ', summary)
    summary = re.sub(r'\s*\([A-Za-z\s]+,\s*\d{4}\)\s*', ' ', summary)
    
    # Address double words (common in summarization errors)
    summary = re.sub(r'\b(\w+)\s+\1\b', r'\1', summary, flags=re.IGNORECASE)
    
    # Fix spacing around punctuation
    summary = re.sub(r'\s+([.,;:!?])', r'\1', summary)
    summary = re.sub(r'([.,;:!?])(?=[^\s])', r'\1 ', summary)
    
    # Fix spacing after period
    summary = re.sub(r'\.(?=[A-Z0-9])', '. ', summary)
    
    # Ensure proper capitalization at beginning
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    # Improve transition from problem statement to method description
    # Often summaries jump abruptly from problem to method
    summary = re.sub(r'(\. )([Tt]o address this|[Tt]o solve this|[Tt]o tackle this)', r'. To address this', summary)
    
    # Add method transition words if method is mentioned without proper introduction
    if re.search(r'\. [A-Z][^.!?]*?(proposed|introduce|present)[^.!?]*?method', summary) and \
       not re.search(r'(proposed|introduce|present)[^.!?]*(method|approach|framework)', summary.split('.')[0]):
        # Add transition if method is mentioned later without setup
        summary = re.sub(r'(\. )([A-Z][^.!?]*?(proposed|introduce|present)[^.!?]*?method)', 
                        r'. As a solution, \2', summary)
    
    # Strengthen method description by ensuring use of strong verbs
    for weak, strong in [
        ('uses', 'leverages'), 
        ('has', 'utilizes'), 
        ('makes', 'employs'),
        ('with', 'through'),
        ('using', 'utilizing'),
        ('shows', 'demonstrates')
    ]:
        # Only replace if in context of method description
        method_context = r'(method|approach|framework|model|system|technique)(\s+[^.!?]*?)' + weak
        replacement = r'\1\2' + strong
        summary = re.sub(method_context, replacement, summary, flags=re.IGNORECASE)
    
    return summary

def summarize_abstract_with_huggingface(abstract, api_key=None, title=None):
    """
    A comprehensive approach to summarize academic paper abstracts using the HuggingFace API.
    Handles long abstracts by breaking them into chunks with appropriate context.
    Ensures method descriptions are preserved in the summarization process.
    
    Args:
        abstract: The paper abstract to summarize
        api_key: Optional HuggingFace API key
        title: Optional paper title to provide context
        
    Returns:
        A summarized version of the abstract
    """
    try:
        # Make sure we have a clean abstract
        cleaned_abstract = preprocess_abstract(abstract) if abstract else ""
        if not cleaned_abstract:
            return None
            
        # Try to identify method section in the abstract for special treatment
        method_section = None
        method_indicators = [
            r'we propose', r'we present', r'we introduce', r'our approach', 
            r'our method', r'we develop', r'our framework', r'our system',
            r'proposed method', r'is composed of', r'consists of', r'comprises of',
            r'is implemented'
        ]
        
        for indicator in method_indicators:
            # Find a sentence with method indicator
            match = re.search(f'([^.!?]*{indicator}[^.!?]*[.!?])', cleaned_abstract, re.IGNORECASE)
            if match:
                method_section = match.group(1).strip()
                break
                
        # Construct context with title if available
        context = f"Title: {title}\n\n" if title else ""
        context += cleaned_abstract
        
        # If abstract is short enough, summarize it directly
        if len(cleaned_abstract) <= MAX_ABSTRACT_LENGTH:
            # Add method section context if identified
            if method_section:
                context += f"\n\nMake sure to include the proposed method: {method_section}"
                
            summary = summarize_with_huggingface(context, api_key)
            
            if summary:
                return summary
                
            # Fallback to SMMRY web service
            logging.info("Using SMMRY web service as fallback...")
            smmry_summary = summarize_with_smmry(context)
            if smmry_summary:
                return smmry_summary
                
            # Final fallback: Use improved extractive summarization
            logging.info("Using extractive summarization as final fallback...")
            return extract_key_sentences(cleaned_abstract)
        
        # For longer abstracts, use a chunk-based approach
        logging.info(f"Abstract is long ({len(cleaned_abstract)} chars), using chunk-based summarization")
        
        # Split abstract into chunks
        chunks = split_abstract_into_chunks(cleaned_abstract)
        
        # Check if we've identified a method section
        method_chunk_idx = -1
        if method_section:
            # Find which chunk contains the method section
            for i, chunk in enumerate(chunks):
                if method_section in chunk:
                    method_chunk_idx = i
                    break
        
        # If we have only two chunks, summarize them separately with context
        if len(chunks) <= 2:
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                
                # Add special instructions based on chunk position
                chunk_type = ""
                if i == 0:
                    chunk_type = "This is the beginning of the abstract, covering the problem statement and context."
                elif i == len(chunks) - 1:
                    chunk_type = "This is the end of the abstract, likely covering results and conclusions."
                    
                # Add special treatment for method chunk
                if i == method_chunk_idx:
                    chunk_type += " This chunk contains the proposed method description, which should be preserved."
                    
                chunk_with_context = f"Title: {title}\n\nChunk {i+1}/{len(chunks)}: {chunk_type}\n\n{chunk}"
                
                # If this is the method chunk, add explicit instruction
                if i == method_chunk_idx:
                    chunk_with_context += f"\n\nMake sure to include details about the proposed method: {method_section}"
                
                # Try HuggingFace API first
                chunk_summary = summarize_with_huggingface(chunk_with_context, api_key)
                
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                else:
                    # Fallback to improved extractive summarization
                    # Add transitions based on position
                    extracted_summary = extract_key_sentences(chunk)
                    if i == 0 and len(chunks) > 1:
                        chunk_summaries.append(extracted_summary)
                    elif i == len(chunks) - 1:
                        chunk_summaries.append(extracted_summary)
                    else:
                        chunk_summaries.append(extracted_summary)
            
            # Combine chunk summaries with transitions
            combined_text = ""
            for i, summary in enumerate(chunk_summaries):
                if i == 0:
                    combined_text = summary
                elif i == 1 and len(chunk_summaries) == 2:
                    # Add a transition between problem statement and method/results
                    # Check if the second chunk likely contains method information
                    if method_chunk_idx == 1 or method_section in chunks[1]:
                        combined_text += " To address this challenge, " + summary.lower() if summary[0].isupper() else summary
                    else:
                        combined_text += " " + summary
                else:
                    combined_text += " Furthermore, " + summary.lower() if summary[0].isupper() else summary
            
            # Final pass to ensure coherent summary
            title_context = f"Title: {title}\n\n" if title else ""
            final_context = f"{title_context}Create a coherent, unified summary from these sections, emphasizing the proposed method:\n\n{combined_text}"
            
            final_summary = summarize_with_huggingface(final_context, api_key)
            if final_summary:
                return final_summary
            
            # If API fails for final summary, use the combined text
            return combined_text
        else:
            # First level: Summarize each chunk individually
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                
                # Add context about what this chunk represents and the paper title
                chunk_context = f"Title: {title}\n\n"
                
                # Add position context for the chunk
                if i == 0:
                    chunk_context += "This is the beginning of the abstract, likely introducing the problem.\n\n"
                elif i == len(chunks) - 1:
                    chunk_context += "This is the end of the abstract, likely covering results and conclusions.\n\n"
                else:
                    # Middle chunks might contain method descriptions
                    if i == method_chunk_idx or (method_chunk_idx == -1 and i == 1):  # If method chunk or likely method position
                        chunk_context += "This part likely describes the proposed method.\n\n"
                    else:
                        chunk_context += "This is a middle section of the abstract.\n\n"
                
                chunk_context += chunk
                
                # If this chunk contains the method section, add special instruction
                if i == method_chunk_idx or (method_section and method_section in chunk):
                    chunk_context += f"\n\nMake sure to include details about the proposed method: {method_section}"
                
                # Try HuggingFace API first
                chunk_summary = summarize_with_huggingface(chunk_context, api_key)
                
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                else:
                    # Fallback to improved extractive summarization
                    chunk_summaries.append(extract_key_sentences(chunk))
            
            # Second level: Combine all chunk summaries and summarize again
            combined_text = ""
            for i, summary in enumerate(chunk_summaries):
                if i == 0:
                    combined_text = summary
                elif i == method_chunk_idx or (method_chunk_idx == -1 and i == 1):  # Method chunk or likely method position
                    # Add a transition for the method chunk
                    combined_text += " To address this challenge, " + summary.lower() if summary[0].isupper() else summary
                else:
                    # Add appropriate transitions
                    if i == len(chunk_summaries) - 1:
                        combined_text += " Finally, " + summary.lower() if summary[0].isupper() else summary
                    else:
                        combined_text += " Furthermore, " + summary.lower() if summary[0].isupper() else summary
            
            meta_context = f"Title: {title}\n\nCreate a coherent, unified summary from these sections, with particular emphasis on the proposed method or approach:\n\n{combined_text}"
            
            if method_section:
                meta_context += f"\n\nMake sure to include this important method information: {method_section}"
            
            final_summary = summarize_with_huggingface(meta_context, api_key)
            if final_summary:
                return final_summary
            
            # If second-level summarization fails, use extractive approach on combined text
            return extract_key_sentences(combined_text)
    except Exception as e:
        logging.error(f"Error in summarize_abstract_with_huggingface: {str(e)}")
        return extract_key_sentences(abstract)
    
    # If HuggingFace API fails, try SMMRY web service
    logging.info("Using SMMRY web service as fallback...")
    smmry_summary = summarize_with_smmry(context)
    if smmry_summary:
        return smmry_summary
    
    # Final fallback: Use improved extractive summarization
    logging.info("Using extractive summarization as final fallback...")
    return extract_key_sentences(cleaned_abstract)

def save_summaries_to_markdown(summaries, output_path):
    """Save the paper summaries in a markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# AI-Generated Paper Summaries\n\n")
        f.write("*Summaries of papers in the downloaded_papers folder*\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        for paper in summaries:
            f.write(f"## {paper['title']}\n\n")
            
            # Add keywords if available
            if paper.get('keywords'):
                f.write("### Keywords\n\n")
                keywords_str = ", ".join(paper['keywords'])
                f.write(f"{keywords_str}\n\n")
            
            if paper.get('abstract'):
                f.write("### Original Abstract\n\n")
                # Format abstract with nice wrapping
                wrapped_abstract = textwrap.fill(paper['abstract'], width=80)
                f.write(f"{wrapped_abstract}\n\n")
            
            if paper.get('summary'):
                f.write("### Summary\n\n")
                # Format summary with nice wrapping
                wrapped_summary = textwrap.fill(paper['summary'], width=80)
                f.write(f"{wrapped_summary}\n\n")
            else:
                f.write("*No summary available*\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Saved summaries to {output_path}")

def save_summaries_to_html(summaries, output_path):
    """Save the paper summaries in an enhanced HTML file with improved typography and structure."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Generated Paper Summaries</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --light-bg: #f9f9f9;
            --medium-bg: #e8f4f8;
            --border-color: #eee;
            --text-color: #333;
            --light-text: #7f8c8d;
            --method-bg: #ebf5eb;
            --method-border: #28a745;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px;
            color: var(--text-color);
            background-color: #fff;
        }
        
        h1 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.2em;
            text-align: center;
        }
        
        h2 {
            color: var(--secondary-color);
            margin-top: 40px;
            font-size: 1.8em;
            border-left: 4px solid var(--secondary-color);
            padding-left: 15px;
        }
        
        h3 {
            color: var(--primary-color);
            font-size: 1.3em;
            margin-top: 25px;
            border-bottom: 1px dotted var(--border-color);
            padding-bottom: 5px;
        }
        
        .paper-container {
            margin-bottom: 60px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: box-shadow 0.3s ease;
        }
        
        .paper-container:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .abstract {
            background-color: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            font-style: italic;
            text-align: justify;
            margin-bottom: 20px;
            line-height: 1.7;
            border-left: 3px solid var(--secondary-color);
        }
        
        .summary {
            background-color: var(--medium-bg);
            padding: 20px;
            border-radius: 8px;
            text-align: justify;
            line-height: 1.8;
            font-size: 1.05em;
            border-left: 3px solid var(--accent-color);
        }
        
        .method-description {
            background-color: var(--method-bg);
            padding: 20px;
            border-radius: 8px;
            text-align: justify;
            line-height: 1.8;
            font-size: 1.05em;
            border-left: 3px solid var(--method-border);
            margin-bottom: 20px;
        }
        
        .method-description strong {
            color: var(--method-border);
        }
        
        .keywords {
            background-color: #f0f7fa;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .keywords span {
            display: inline-block;
            background-color: #e1e8ed;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            color: var(--primary-color);
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .keywords span:hover {
            background-color: var(--secondary-color);
            color: white;
        }
        
        .generation-info {
            color: var(--light-text);
            font-size: 0.9em;
            margin-bottom: 40px;
            text-align: center;
        }
        
        .section-title {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .section-title::before {
            content: '';
            width: 8px;
            height: 8px;
            background-color: var(--secondary-color);
            margin-right: 8px;
            display: inline-block;
            border-radius: 50%;
        }
        
        .method-section-title::before {
            background-color: var(--method-border);
        }
        
        details {
            margin-top: 15px;
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 5px 15px;
        }
        
        summary {
            cursor: pointer;
            font-weight: bold;
            padding: 8px 0;
        }
        
        summary:hover {
            color: var(--secondary-color);
        }
        
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }
            
            .paper-container {
                padding: 15px;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            h2 {
                font-size: 1.5em;
            }
        }
    </style>
</head>
<body>
    <h1>AI-Generated Paper Summaries</h1>
    <p class="generation-info">Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
    
    for paper in summaries:
        html_content += f'<div class="paper-container">\n'
        html_content += f'    <h2>{paper["title"]}</h2>\n'
        
        # Add keywords if available
        if paper.get('keywords'):
            html_content += f'    <div class="section-title"><h3>Keywords</h3></div>\n'
            html_content += f'    <div class="keywords">\n'
            for keyword in paper['keywords']:
                html_content += f'        <span>{keyword}</span>\n'
            html_content += f'    </div>\n'
        
        # Add method description if available
        if paper.get('abstract'):
            method_desc = extract_method_description(paper['abstract'])
            if method_desc:
                html_content += f'    <div class="section-title method-section-title"><h3>Proposed Method</h3></div>\n'
                # Convert markdown-style bold to HTML bold
                method_desc = method_desc.replace('**', '<strong>').replace('**', '</strong>')
                html_content += f'    <div class="method-description">{method_desc}</div>\n'
        
        # Add original abstract
        if paper.get('abstract'):
            html_content += f'    <div class="section-title"><h3>Original Abstract</h3></div>\n'
            html_content += f'    <div class="abstract">{paper["abstract"]}</div>\n'
        
        # Add summary
        if paper.get('summary'):
            html_content += f'    <div class="section-title"><h3>Summary</h3></div>\n'
            html_content += f'    <div class="summary">{paper["summary"]}</div>\n'
        
        html_content += '</div>\n'
    
    html_content += "</body>\n</html>"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Saved HTML summaries to {output_path}")

def get_huggingface_api_key():
    """Get the HuggingFace API key from environment variables."""
    hf_token = os.environ.get("HUGGINGFACE_API_KEY")
    if hf_token:
        logging.info("Found HuggingFace API key in environment variables")
        return hf_token
    return None

def extract_method_description(abstract: str) -> Optional[str]:
    """
    Extract and organize a detailed description of the method proposed in the abstract.
    
    Args:
        abstract: The paper abstract text
        
    Returns:
        A structured description of the proposed method, or None if no method is clearly described
    """
    if not abstract:
        return None
        
    # Clean the abstract
    cleaned_abstract = preprocess_abstract(abstract)
    
    # Method-related indicators to look for
    method_indicators = [
        r'we propose', r'we present', r'we introduce', r'our approach', 
        r'our method', r'we develop', r'our framework', r'our system',
        r'this paper proposes', r'this paper presents', r'this paper introduces',
        r'proposed method', r'novel method', r'new approach', r'new framework',
        r'we design', r'we create', r'we implement', r'we formulate',
        r'called', r'named', r'termed'
    ]
    
    # First try to find sentences with explicit method descriptions
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_abstract)
    method_sentences = []
    method_name = None
    
    # Look for the method name (often presented as "we propose X" or "called X")
    for indicator in method_indicators:
        # Different patterns for method name extraction
        if 'called' in indicator or 'named' in indicator or 'termed' in indicator:
            name_pattern = f'({indicator}\\s+)([A-Z]\\w*(?:-?\\w+)*)'
        else:
            name_pattern = f'({indicator}\\s+)(?:a|an|the)?\\s*(?:novel|new)?\\s*(?:method|approach|framework|system|model|technique)?(?:\\s+named|\\s+called)?\\s+([A-Z]\\w*(?:-?\\w+)*)'
            
        match = re.search(name_pattern, cleaned_abstract, re.IGNORECASE)
        if match:
            method_name = match.group(2).strip()
            break
    
    # Extract sentences that describe the method
    for i, sentence in enumerate(sentences):
        # Check if sentence contains method indicators or the method name
        if (any(re.search(f'{indicator}', sentence, re.IGNORECASE) for indicator in method_indicators) or
            (method_name and method_name in sentence)):
            method_sentences.append(sentence)
            
            # Also include the next sentence for context if it's not the last sentence
            if i < len(sentences) - 1 and len(sentences[i+1].split()) > 5:  # Only if next sentence is substantial
                if not any(indicator in sentences[i+1].lower() for indicator in ['we evaluate', 'we compare', 'experiment']):
                    method_sentences.append(sentences[i+1])
    
    # If we found method-related sentences, compile them into a description
    if method_sentences:
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for s in method_sentences:
            if s not in seen:
                seen.add(s)
                unique_sentences.append(s)
                
        method_description = ' '.join(unique_sentences)
        
        # If a method name was found, highlight it
        if method_name:
            header = f"The paper proposes a method called **{method_name}**.\n\n"
            return header + method_description
        else:
            return "The paper proposes the following method:\n\n" + method_description
    
    # If no explicit method sentences were found, use a more general approach
    if 'method' in cleaned_abstract.lower() or 'approach' in cleaned_abstract.lower():
        # Find the most relevant sentence containing 'method' or 'approach'
        method_sentence = None
        for sentence in sentences:
            if 'method' in sentence.lower() or 'approach' in sentence.lower():
                method_sentence = sentence
                break
                
        if method_sentence:
            return "Method information extracted from abstract:\n\n" + method_sentence
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract and summarize abstracts from academic papers.")
    parser.add_argument("--input-dir", default="downloaded_papers", help="Directory containing PDF files")
    parser.add_argument("--output", default="paper_summaries.md", help="Output file (markdown format)")
    parser.add_argument("--html", action="store_true", help="Generate HTML output instead of markdown")
    parser.add_argument("--hf-token", help="HuggingFace API token (optional, limited requests possible without it)")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_path = args.output
    
    # Check for API key in command line args first, then environment variables
    api_key = args.hf_token
    if not api_key:
        api_key = get_huggingface_api_key()
        
    # Handle HTML output option
    if args.html and output_path.endswith('.md'):
        output_path = output_path.replace('.md', '.html')
    
    # Get all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return
    
    # Limit number of papers if specified
    if args.max_papers and args.max_papers > 0 and args.max_papers < len(pdf_files):
        logging.info(f"Limiting to {args.max_papers} papers (out of {len(pdf_files)} found)")
        pdf_files = pdf_files[:args.max_papers]
    else:
        logging.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    summaries = []
    
    for pdf_file in pdf_files:
        title = extract_title_from_filename(pdf_file)
        logging.info(f"Processing: {title}")
        
        # Use improved abstract extraction with validation
        abstract = extract_and_validate_abstract(pdf_file)
        
        if not abstract:
            logging.warning(f"Could not extract abstract from {pdf_file}")
            summaries.append({"title": title})
            continue
        
        logging.info(f"Extracted abstract ({len(abstract)} chars)")
        
        # Extract potential keywords from abstract
        keywords = keywords_from_abstract(abstract)
        
        # Get the proposed method description
        method_description = extract_method_description(abstract)
        if method_description:
            logging.info(f"Extracted method description ({len(method_description)} chars)")
        
        # Get the summary with improved approach
        summary = summarize_abstract_with_huggingface(abstract, api_key, title)
        
        # Apply post-processing to enhance summary quality
        if summary:
            summary = post_process_summary(summary)
            logging.info(f"Generated summary ({len(summary)} chars)")
        
        # Store all the information
        paper_info = {
            "title": title,
            "abstract": abstract,
            "summary": summary,
            "keywords": keywords,
            "method_description": method_description
        }
        
        summaries.append(paper_info)
        
        # Add a small delay to avoid API rate limits
        time.sleep(2)
    
    if args.html:
        save_summaries_to_html(summaries, output_path)
    else:
        save_summaries_to_markdown(summaries, output_path)
    
    # Print summary statistics
    abstracts_found = sum(1 for s in summaries if s.get('abstract'))
    abstracts_validated = sum(1 for s in summaries if s.get('abstract') and validate_abstract(s.get('abstract')))
    summaries_generated = sum(1 for s in summaries if s.get('summary'))
    
    logging.info("Summary extraction complete!")
    print("\nSummary Extraction Report:")
    print(f"Papers processed: {len(summaries)}")
    print(f"Abstracts successfully identified: {abstracts_found} ({abstracts_found/len(summaries)*100:.1f}%)")
    print(f"Abstracts passing validation: {abstracts_validated} ({abstracts_validated/len(summaries)*100:.1f}%)")
    print(f"Summaries generated: {summaries_generated} ({summaries_generated/len(summaries)*100:.1f}%)")
    print(f"\nOutput saved to: {output_path}")
    
    if abstracts_found < len(summaries):
        print("\nTip: If some abstracts were not correctly identified, try running with a specific paper:")
        print("  python summarize.py --input-dir path/to/specific_paper_folder")
        
    print("\nThe script now uses enhanced abstract extraction techniques to better identify")
    print("and clean academic paper abstracts across various formats including arXiv and")
    print("conference papers. Abstracts are validated to ensure they contain actual abstract")
    print("content rather than other sections of the paper.")

if __name__ == "__main__":
    main()
