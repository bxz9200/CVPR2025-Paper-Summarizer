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

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Usually academic paper abstracts are in the first 3 pages
        for i in range(min(3, len(reader.pages))):
            text += reader.pages[i].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def extract_abstract_from_pdf(pdf_path):
    """
    Extract the abstract from a PDF file.
    
    This function focuses on finding the text between Abstract and Introduction sections,
    even when figures or other content appear before the abstract.
    """
    try:
        # Read the PDF file
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None
            
        # Get reader for metadata extraction later
        reader = None
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            logging.debug(f"Error creating PDF reader for metadata: {str(e)}")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Define patterns for section headings
        abstract_pattern = r'(?i)^\s*Abstract\s*$|^\s*Abstract\s*[.:]|^\s*ABSTRACT\s*$|^\s*ABSTRACT\s*[.:]'
        intro_pattern = r'(?i)^\s*1\.?\s*Introduction\s*$|^\s*I\.?\s*Introduction\s*$|^\s*Introduction\s*$'
        
        abstract_section = re.compile(abstract_pattern)
        intro_section = re.compile(intro_pattern)
        
        # Find Abstract and Introduction section indices
        abstract_index = None
        intro_index = None
        
        for i, para in enumerate(paragraphs):
            # Check for Abstract section
            if abstract_section.match(para):
                abstract_index = i
            # Check for Introduction section
            elif intro_section.match(para):
                intro_index = i
                # Once we found the Introduction, we can stop searching if we already found the Abstract
                if abstract_index is not None:
                    break
        
        # If we found both Abstract and Introduction sections in the correct order
        if abstract_index is not None and intro_index is not None and abstract_index < intro_index:
            # The abstract content starts from the paragraph after abstract_index
            # and continues until just before intro_index
            abstract_content = []
            
            # If the Abstract heading contains content (in-line with the heading)
            if len(paragraphs[abstract_index]) > 20:  # If it contains more than just "Abstract"
                # Extract everything after "Abstract"
                match = abstract_section.search(paragraphs[abstract_index])
                if match:
                    abstract_text = paragraphs[abstract_index][match.end():].strip()
                    if abstract_text:
                        abstract_content.append(abstract_text)
            
            # Add all paragraphs between Abstract and Introduction
            for i in range(abstract_index + 1, intro_index):
                # Skip any figure captions or very short paragraphs that might be noise
                if len(paragraphs[i]) > 20 and not re.match(r'^(Figure|Table|Fig\.)\s+\d+', paragraphs[i]):
                    abstract_content.append(paragraphs[i])
            
            if abstract_content:
                full_abstract = ' '.join(abstract_content)
                cleaned = preprocess_abstract(full_abstract)
                if validate_abstract(cleaned):
                    logging.info(f"Successfully extracted abstract between Abstract and Introduction sections for {pdf_path}")
                    return cleaned
        
        # Try alternative methods if the above approach didn't work
        
        # Look for abstract marked with a specific pattern
        abstract_match = re.search(r'(?i)(\n\s*abstract\s*\n)(.*?)(?=\n\s*\d?\.?\s*introduction|\n\s*\d\.|$)', text, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(2).strip()
            # Check if this is a substantial chunk of text
            if len(abstract_text) >= 150 and len(abstract_text) <= 3000:
                cleaned = preprocess_abstract(abstract_text)
                if validate_abstract(cleaned):
                    logging.info(f"Found abstract using pattern matching for {pdf_path}")
                    return cleaned
        
        # Try to extract from metadata as a last resort
        if reader:
            try:
                info = reader.metadata
                if info and hasattr(info, 'subject') and info.subject and len(info.subject) > 100:
                    abstract_candidate = info.subject
                    cleaned = preprocess_abstract(abstract_candidate)
                    if validate_abstract(cleaned):
                        logging.info(f"Using abstract from PDF metadata for {pdf_path}")
                        return cleaned
            except Exception as e:
                logging.debug(f"Error reading PDF metadata: {str(e)}")
        
        # Fallback: Use a generic approach to find abstract-like text
        for i, para in enumerate(paragraphs[:15]):  # Check first 15 paragraphs only
            if (200 <= len(para) <= 3000 and 
                not re.search(r'@|\buniversity\b|\binstitute\b|department|figure|fig\.|table', para.lower()) and
                not re.match(r'^\d+\.', para) and 
                not intro_section.match(para)):
                cleaned = preprocess_abstract(para)
                if validate_abstract(cleaned):
                    logging.info(f"Using generic abstract detection for {pdf_path}")
                    return cleaned
        
        # Last resort: Just use the first substantial paragraph
        logging.warning(f"Couldn't identify abstract section in {pdf_path}, using first substantial paragraph")
        for para in paragraphs:
            if len(para.strip()) >= 200 and len(para.strip()) <= 3000:
                cleaned = preprocess_abstract(para.strip())
                return cleaned
        
        return None
        
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
    
    # Remove any markdown formatting characters that could cause bold/italic styling
    cleaned = cleaned.replace('**', '').replace('__', '').replace('*', '').replace('_', ' ')
    
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
    
    # NEW: Additional patterns for abstracts that might not be at the beginning of papers
    # For papers like Magma where abstract appears after figures/illustrations
    midpoint_abstract_patterns = [
        r'(?i)(we|this paper) (present|propose|introduce)s? (a|an|the) (\w+|\w+\s\w+) (model|approach|method|framework|system)',
        r'(?i)(in this paper|this work|we) (present|propose|introduce|describe)',
        r'(?i)(this (work|paper|study)|we) (develop|present)s? (a|an|the)',
        r'(?i)we (tackle|address|solve|focus on) (the problem|the task|the challenge)',
        r'(?i)(is|are) (a (significant|major|important)|an important) (challenge|problem|task|issue)'
    ]
    
    # Check if abstract contains these typical abstract phrases anywhere in the text
    has_abstract_beginning = any(re.search(pattern, abstract[:100]) for pattern in abstract_beginning_patterns)
    
    # NEW: Check if abstract contains typical abstract content anywhere in the text (not just beginning)
    has_midpoint_abstract_pattern = any(re.search(pattern, abstract) for pattern in midpoint_abstract_patterns)
    
    # Check for abstract-like content (presence of keywords related to contributions)
    contribution_keywords = [
        r'propose', r'present', r'introduce', r'novel', r'approach', r'method',
        r'contribution', r'demonstrate', r'show', r'performance', r'experiments',
        r'results', r'outperform', r'state-of-the-art', r'state of the art',
        # NEW: Additional keywords common in academic abstracts
        r'framework', r'model', r'system', r'implementation', r'architecture',
        r'foundation model', r'multimodal', r'evaluation', r'benchmark', r'algorithm',
        r'technique', r'solution', r'paradigm', r'accuracy', r'effectiveness'
    ]
    
    has_contribution_terms = sum(1 for keyword in contribution_keywords if re.search(r'\b' + keyword + r'\b', abstract.lower())) >= 2
    
    # NEW: Check for abstract-specific structural patterns
    # Abstracts often have a specific structure: problem statement, approach, results
    has_problem_statement = re.search(r'(?i)(problem|challenge|task|issue|limitation)', abstract[:len(abstract)//2])
    has_approach_mention = re.search(r'(?i)(approach|method|model|framework|propose|present|introduce)', abstract)
    has_results_mention = re.search(r'(?i)(result|performance|evaluation|experiment|demonstrate|show|achieve)', abstract[len(abstract)//3:])
    
    has_abstract_structure = (has_problem_statement and has_approach_mention) or (has_approach_mention and has_results_mention)
    
    # NEW: Check for conventional end-of-abstract statements
    concluding_statements = [
        r'(?i)(demonstrate|show) (the|our) (effectiveness|performance|results)',
        r'(?i)(experimental|our) results (show|demonstrate)',
        r'(?i)(outperform|surpass|exceed) (previous|existing|current|state-of-the-art)',
        r'(?i)(achieve|attain) (state-of-the-art|sota|superior|better) (performance|results)'
    ]
    
    has_conclusion = any(re.search(pattern, abstract[len(abstract)//2:]) for pattern in concluding_statements)
    
    # Check for common academic paper abstract phrases that appear in the middle of the abstract
    # This helps identify abstracts in papers like Magma where the abstract might not start with typical patterns
    academic_phrases = [
        r'(?i)our (main|key) contribution',
        r'(?i)our (approach|method|model|framework)',
        r'(?i)we (evaluate|test|validate|assess)',
        r'(?i)experimental (results|evaluation)',
        r'(?i)(outperforms|improves upon)',
        r'(?i)(extensive|comprehensive) experiments'
    ]
    
    has_academic_phrases = any(re.search(pattern, abstract) for pattern in academic_phrases)
    
    # Return overall assessment with enhanced logic to handle abstracts in various positions
    if has_abstract_beginning:
        return True
    elif has_midpoint_abstract_pattern and suspicion_count < 2:
        return True
    elif has_contribution_terms and suspicion_count < 2:
        return True
    elif has_abstract_structure and suspicion_count < 2:
        return True
    elif has_conclusion and has_approach_mention and suspicion_count < 2:
        return True
    elif has_academic_phrases and suspicion_count < 2 and 50 <= word_count <= 500:
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
        # Try our specialized two-column conference paper extraction
        abstract = extract_conference_paper_abstract(pdf_path)
        if abstract:
            logging.info(f"Successfully extracted abstract using conference paper extractor for {pdf_path}")
            return abstract
        return None
        
    # Clean the abstract
    cleaned_abstract = preprocess_abstract(abstract)
    
    # Validate the abstract
    if validate_abstract(cleaned_abstract):
        logging.info(f"Successfully validated abstract from {pdf_path}")
        return cleaned_abstract
        
    # If validation failed, try a more aggressive approach
    logging.warning(f"First extracted abstract failed validation for {pdf_path}, trying alternative method")
    
    # Try specialized conference paper layout extraction
    conference_abstract = extract_conference_paper_abstract(pdf_path)
    if conference_abstract:
        logging.info(f"Successfully extracted abstract using conference paper extractor after validation failed")
        return conference_abstract
    
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

def extract_conference_paper_abstract(pdf_path):
    """
    Specialized function to extract abstracts from conference-style papers with 
    two-column layouts, where the abstract might appear after figures or other content.
    This is common in papers like the Magma paper where figures appear at the top.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted abstract or None if extraction failed
    """
    try:
        reader = PdfReader(pdf_path)
        
        # Conference papers often have abstracts on the first page, sometimes extending to second
        first_page_text = reader.pages[0].extract_text()
        full_text = first_page_text
        
        # Also check second page in case abstract continues
        if len(reader.pages) > 1:
            second_page_text = reader.pages[1].extract_text()
            # Only include text before introduction if it exists on second page
            intro_match = re.search(r'\b(INTRODUCTION|Introduction|1\.(\s+|)INTRODUCTION|I\.\s+INTRODUCTION)\b', second_page_text)
            if intro_match:
                continuation = second_page_text[:intro_match.start()].strip()
                full_text += "\n\n" + continuation
            else:
                full_text += "\n\n" + second_page_text
        
        # Look for abstract header with specific patterns common in conference papers
        # Pattern 1: Abstract appears after author affiliations with a clear header
        abstract_header_match = re.search(r'(?i)(\n\s*Abstract\s*\n|\n\s*ABSTRACT\s*\n)', full_text)
        if abstract_header_match:
            header_pos = abstract_header_match.end()
            # Look for text after the header until introduction or keywords section
            abstract_end_match = re.search(r'(?i)(\n\s*\d?\.?\s*Introduction|\n\s*\d\.\s|\n\s*Keywords|\n\s*Index Terms)', full_text[header_pos:])
            if abstract_end_match:
                abstract_text = full_text[header_pos:header_pos + abstract_end_match.start()].strip()
                if len(abstract_text) >= 150:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
            else:
                # Try to extract a reasonable chunk of text (conference abstracts are typically 150-350 words)
                # Take up to 3000 characters after the abstract header
                abstract_text = full_text[header_pos:header_pos + 3000].strip()
                # Try to find a logical endpoint (end of paragraph)
                paragraph_end = re.search(r'\n\s*\n', abstract_text)
                if paragraph_end:
                    abstract_text = abstract_text[:paragraph_end.start()].strip()
                    
                if len(abstract_text) >= 150:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
        
        # Pattern 2: Look for abstract in a column-based layout (common in IEEE/ACM/CVPR papers)
        # First identify if there are author affiliations with superscripts/numbers
        affiliation_pattern = re.search(r'(?i)(\d{1,2}|\*|†){1,3}(University|Institute|Research|Lab|Corporation|Inc\.|LLC)', full_text)
        if affiliation_pattern:
            # Find the end of affiliations section
            lines = full_text.split('\n')
            affiliation_end_idx = None
            
            for i, line in enumerate(lines):
                if re.search(r'(?i)(\d{1,2}|\*|†){1,3}(University|Institute|Research|Lab|Corporation)', line):
                    affiliation_end_idx = i
            
            if affiliation_end_idx is not None and affiliation_end_idx + 1 < len(lines):
                # Look for abstract after affiliations
                for i in range(affiliation_end_idx + 1, min(affiliation_end_idx + 10, len(lines))):
                    if re.search(r'(?i)\b(abstract|ABSTRACT)\b', lines[i]):
                        # Found the abstract header, collect text until next section
                        abstract_lines = []
                        j = i + 1
                        while j < len(lines) and not re.search(r'(?i)(\d?\.?\s*Introduction|\d\.\s|Keywords|Index Terms)', lines[j]):
                            abstract_lines.append(lines[j])
                            j += 1
                        
                        if abstract_lines:
                            abstract_text = ' '.join(abstract_lines).strip()
                            if len(abstract_text) >= 150:
                                cleaned = preprocess_abstract(abstract_text)
                                if validate_abstract(cleaned):
                                    return cleaned
        
        # Pattern 3: Look for a properly formatted abstract in a two-column layout
        # This handles papers like the Magma paper where figures/diagrams appear before the abstract
        # The abstract often appears in a specific formatted way in the left column
        column_abstract_pattern = re.search(r'(?i)(\n\s*Abstract\s*[\.:—-]?\s*\n|\n\s*ABSTRACT\s*[\.:—-]?\s*\n)(.*?)(?=\n\s*\d?\.?\s*Introduction|\n\s*\d\.\s|\n\s*Keywords|\n\s*Index Terms|\n\s*\d\.\s*\n)', full_text, re.DOTALL)
        if column_abstract_pattern:
            abstract_text = column_abstract_pattern.group(2).strip()
            if len(abstract_text) >= 150:
                cleaned = preprocess_abstract(abstract_text)
                if validate_abstract(cleaned):
                    return cleaned
        
        # Pattern 4: Try a common conference paper format where Abstract appears on one line
        # and the actual content follows on subsequent lines
        abstract_line_pattern = re.search(r'(?i)(^|\n)(\s*)Abstract(\s*)\n', full_text)
        if abstract_line_pattern:
            start_pos = abstract_line_pattern.end()
            # Get the indentation level of the abstract heading
            abstract_indentation = len(abstract_line_pattern.group(2))
            
            # Find the next section which might have similar indentation
            lines = full_text[start_pos:].split('\n')
            abstract_lines = []
            
            for line in lines:
                if line.strip() == '':  # Skip empty lines
                    continue
                    
                line_indent = len(line) - len(line.lstrip())
                
                # If we find a line that looks like a section heading with similar indentation,
                # we've reached the end of the abstract
                if (line_indent <= abstract_indentation and 
                    re.search(r'(?i)(\d?\.?\s*Introduction|\d\.\s|Keywords|Index Terms|Related Work)', line.strip())):
                    break
                
                abstract_lines.append(line)
                
                # Don't collect too many lines - abstracts are typically not more than 20 lines
                if len(abstract_lines) >= 20:
                    break
            
            if abstract_lines:
                abstract_text = ' '.join(abstract_lines).strip()
                if len(abstract_text) >= 150:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned):
                        return cleaned
        
        # If we couldn't find the abstract with specific patterns, try a generic approach
        # Look for any substantial paragraph near author affiliations that looks like an abstract
        paragraphs = re.split(r'\n\s*\n', full_text)
        for i, para in enumerate(paragraphs):
            # Check if this paragraph contains affiliation info
            if re.search(r'(?i)(\d{1,2}|\*|†){1,3}(University|Institute|Research|Lab)', para):
                # Check if the next substantial paragraph might be the abstract
                for j in range(i+1, min(i+5, len(paragraphs))):
                    candidate = paragraphs[j]
                    # Skip paragraphs that look like section headings or figure captions
                    if (re.search(r'(?i)^(Introduction|Keywords|Index Terms|\d\.\s)', candidate) or 
                        re.search(r'(?i)^(Fig|Figure|Table)\s*\d', candidate) or
                        len(candidate) < 150):
                        continue
                    
                    # This could be an abstract - check if it contains abstract-like content
                    cleaned = preprocess_abstract(candidate)
                    if validate_abstract(cleaned):
                        return cleaned
                        
                # If we haven't found an abstract yet, check if one of the prior paragraphs
                # might be the abstract (sometimes it appears before affiliations)
                for j in range(max(0, i-3), i):
                    candidate = paragraphs[j]
                    # Skip short paragraphs or those that look like titles
                    if len(candidate) < 150 or len(candidate.split()) < 30:
                        continue
                    
                    cleaned = preprocess_abstract(candidate)
                    if validate_abstract(cleaned):
                        return cleaned
        
        return None
        
    except Exception as e:
        logging.error(f"Error in conference paper abstract extraction for {pdf_path}: {str(e)}")
        return None

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

def summarize_with_huggingface(text: str, api_key: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
    """Call the Hugging Face API to summarize text with improved prompt engineering and retry logic."""
    retries = 0
    while retries < max_retries:
        try:
            if api_key:
                headers = {"Authorization": f"Bearer {api_key}"}
            else:
                # The API can be used without authentication for a limited number of requests
                headers = {}
            
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            
            # Extract any potential method description for separate inclusion
            method_description = None
            method_indicators = [
                r'we propose', r'we present', r'we introduce', r'our approach', 
                r'our method', r'we develop', r'our framework', r'our system'
            ]
            
            for indicator in method_indicators:
                # Find a sentence with method indicator
                match = re.search(f'([^.!?]*{indicator}[^.!?]*[.!?])', text, re.IGNORECASE)
                if match:
                    method_description = match.group(1).strip()
                    break
            
            # Place the instructions as a clear system directive, separated from the content
            enhanced_prompt = (
                "### INSTRUCTIONS ###\n"
                "Summarize the following academic paper abstract into a coherent, logical paragraph.\n"
                "Focus on explaining the proposed method/approach and how it works.\n"
                "Include key technical details and significant results.\n\n"
                "### ABSTRACT TO SUMMARIZE ###\n"
                f"{text}\n\n"
            )
            
            # Add the method description as a separate section if found
            if method_description:
                enhanced_prompt += f"### KEY METHOD INFORMATION ###\n{method_description}\n\n"
            
            enhanced_prompt += "### SUMMARY OUTPUT BELOW ###\n"
            
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
                    
                    # Remove any instruction markers that might appear in the output
                    summary = re.sub(r'###.*?###', '', summary)
                    
                    # Clean up the summary to remove any mention of "This paper" at the beginning
                    summary = re.sub(r'^(This paper|The paper|This abstract|The abstract)\s+', '', summary)
                    
                    # Remove any instructions that might have leaked into the output
                    instruction_artifacts = [
                        r'Make sure to include the proposed method',
                        r'Include details about the proposed method',
                        r'Include the method',
                        r'Include significant results',
                        r'Emphasize the method',
                        r'Create a coherent summary',
                        r'Make sure to clearly explain',
                        r'Summarize the following academic paper abstract',
                        r'Summarize this abstract',
                        r'Provide a summary',
                        r'Focus on explaining',
                        r'Include key technical details',
                        r'SUMMARY OUTPUT BELOW',
                        r'INSTRUCTIONS',
                        r'ABSTRACT TO SUMMARIZE',
                        r'KEY METHOD INFORMATION'
                    ]
                    
                    for artifact in instruction_artifacts:
                        # Remove the artifact and anything following it if it appears at the end
                        summary = re.sub(f'{artifact}.*?$', '', summary, flags=re.IGNORECASE)
                        # Also try to remove it if it appears in the middle with a period or other delimiter
                        summary = re.sub(f'(?:[.,;:!?] ){artifact}[^.!?]*?[.!?]', '. ', summary, flags=re.IGNORECASE)
                    
                    # Remove any sentences that look like they're part of the prompt
                    summary_sentences = re.split(r'(?<=[.!?]) ', summary)
                    filtered_sentences = []
                    for sentence in summary_sentences:
                        # Skip sentences that look like prompt instructions
                        if (re.search(r'^summarize', sentence, re.IGNORECASE) or
                            re.search(r'^provide a summary', sentence, re.IGNORECASE) or
                            re.search(r'^focus on', sentence, re.IGNORECASE) or
                            re.search(r'^include key', sentence, re.IGNORECASE) or
                            re.search(r'^make sure', sentence, re.IGNORECASE) or
                            re.search(r'the following academic paper', sentence, re.IGNORECASE)):
                            continue
                        filtered_sentences.append(sentence)
                    
                    # Rejoin the sentences
                    summary = ' '.join(filtered_sentences)
                    
                    # Remove any arXiv identifiers and version numbers
                    summary = re.sub(r'\s*\d{4}\.\d{5}v\d+', '', summary)
                    summary = re.sub(r'\s*arXiv:\s*\d+\.\d+v\d+', '', summary)
                    
                    # Clean up any trailing artifacts
                    summary = re.sub(r'\s*:\s*\.?\s*$', '.', summary)
                    
                    # Apply post-processing to clean any remaining issues
                    summary = post_process_summary(summary)
                    
                    return summary
            elif response.status_code == 503:
                logging.warning(f"HuggingFace API returned status code 503 (attempt {retries+1}/{max_retries})")
                # Exponential backoff
                wait_time = 2 ** retries
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                logging.warning(f"HuggingFace API returned status code {response.status_code}")
                return None
                
        except Exception as e:
            logging.warning(f"Error with HuggingFace API: {str(e)}")
            return None
    
    logging.warning(f"HuggingFace API unavailable after {max_retries} attempts")
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

def fix_ml_acronyms(text: str) -> str:
    """Fix common machine learning acronyms that might have been broken up with spaces."""
    if not text:
        return text
        
    # This ensures terminology like "VLMs" doesn't become "VL Ms"
    ml_acronym_fixes = [
        (r'V\s+L\s+Ms', 'VLMs'),
        (r'V\s+L\s+M', 'VLM'),
        (r'L\s+V\s+L\s+Ms', 'LVLMs'),
        (r'L\s+V\s+L\s+M', 'LVLM'),
        (r'V\s+Q\s+A', 'VQA'),
        (r'G\s+P\s+T', 'GPT'),
        (r'C\s+N\s+Ns', 'CNNs'),
        (r'C\s+N\s+N', 'CNN'),
        (r'L\s+L\s+Ms', 'LLMs'),
        (r'L\s+L\s+M', 'LLM'),
        (r'R\s+N\s+Ns', 'RNNs'),
        (r'R\s+N\s+N', 'RNN'),
        (r'G\s+A\s+Ns', 'GANs'),
        (r'G\s+A\s+N', 'GAN'),
        (r'M\s+L\s+Ps', 'MLPs'),
        (r'M\s+L\s+P', 'MLP'),
        (r'N\s+L\s+P', 'NLP'),
        (r'C\s+V\s+P\s+R', 'CVPR'),
        (r'I\s+C\s+C\s+V', 'ICCV'),
        (r'E\s+C\s+C\s+V', 'ECCV'),
        (r'N\s+e\s+u\s+r\s+I\s+P\s+S', 'NeurIPS'),
        (r'B\s+E\s+R\s+T', 'BERT'),
        (r'Meta\s+V\s+Q\s+A', 'Meta VQA'),  # Specific to the paper mentioned
    ]
    
    for wrong, correct in ml_acronym_fixes:
        text = re.sub(wrong, correct, text)
        
    return text

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
    
    # First, remove prompt instruction leakage that sometimes appears in the output
    # Remove common instructional phrases like "Make sure to include the proposed method"
    instruction_patterns = [
        # Method inclusion instructions
        r'\s*(?:As a solution,)?\s*Make sure to include (?:the )?proposed method:?\s*\.?\s*$',
        r'\s*(?:Make sure to)?\s*include details about the proposed method:?\s*\.?\s*$',
        r'\s*Please include (?:the )?(?:details about )?(?:the )?(?:proposed )?method:?\s*\.?\s*$',
        r'\s*Include (?:the )?(?:proposed )?method:?\s*\.?\s*$',
        r'\s*(?:Also )?(?:include|mention) (?:significant|key|important) results:?\s*\.?\s*$',
        
        # Emphasis instructions
        r'\s*(?:Be sure to )?emphasize the (?:proposed )?method:?\s*\.?\s*$',
        r'\s*(?:Please )?(?:explain|describe) (?:the |how )?(?:proposed )?method works:?\s*\.?\s*$',
        r'\s*(?:Please )?(?:ensure|make sure) (?:the |that )?(?:summary is|text is) (?:coherent|clear|concise):?\s*\.?\s*$',
        r'\s*Create a (?:coherent|concise|clear|unified) summary:?\s*\.?\s*$',
        
        # Full prompt instructions that might have leaked into the output
        r'\s*Summarize the following academic paper abstract into a(?:n?)? (?:coherent|logical|concise)? ?(?:paragraph|summary)\.?\s*$',
        r'\s*(?:Please )?summarize this abstract\.?\s*$',
        r'\s*(?:Please )?provide a summary of this abstract\.?\s*$',
        r'\s*(?:Focus|Focusing) on explaining the proposed method\/approach and how it works\.?\s*$',
        r'\s*Include key technical details and significant results\.?\s*$',
        r'\s*Capture the paper\'s main contributions\.?\s*$',
        r'\s*Summarize the (?:above|following) (?:text|abstract|paper)\.?\s*$',
        
        # Artifacts from Hugging Face structure
        r'\s*\#\#\#.*?\#\#\#\s*$',
        r'\s*SUMMARY OUTPUT BELOW\s*$',
        r'\s*INSTRUCTIONS\s*$',
    ]
    
    for pattern in instruction_patterns:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
    
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
            # Skip sentences that look like prompt instructions
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in instruction_patterns):
                continue
                
            # Also check for sentences that look like they're instructing the model
            if re.search(r'^(?:please|make sure to|be sure to|ensure that|remember to)\b', sentence, re.IGNORECASE):
                continue
                
            # Check for sentences that look like they're asking to summarize
            if re.search(r'^(?:summarize|provide a summary|create a summary)\b', sentence, re.IGNORECASE):
                continue
                
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
    
    # Fix common machine learning acronyms that might have been broken up with spaces
    summary = fix_ml_acronyms(summary)
    
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
    
    # Double-check for any remaining instruction-like artifacts
    final_check_patterns = [
        r'\s*Make sure to include.*?$',
        r'\s*Include (?:the |details about )?.*?$',
        r'\s*Emphasize (?:the |how )?.*?$',
        r'\s*Create a (?:coherent|concise|clear|unified) summary.*?$',
        r'\s*with particular emphasis on.*?$',
        r'\s*Make sure to.*?$',
        r'\s*Summarize (?:the |this |following )?(?:abstract|paper|text).*?$',
        r'\s*Focus on (?:explaining|describing).*?$',
        r'(?:, |. |; )Summarize the following academic paper abstract.*?$',
        r'(?:, |. |; )Provide a summary.*?$',
    ]
    
    for pattern in final_check_patterns:
        summary = re.sub(pattern, '.', summary, flags=re.IGNORECASE)
    
    # Clean up potential double periods that might occur from replacements
    summary = re.sub(r'\.+', '.', summary)
    
    # Remove any text that appears after "arXiv:" as it might be paper identifiers that leaked in
    summary = re.sub(r'\s*arXiv:.*?$', '.', summary, flags=re.IGNORECASE)
    
    # Remove any numbers that look like arXiv IDs at the end
    summary = re.sub(r'\s*\d{4}\.\d{5}v\d+\.?$', '.', summary)
    
    # Ensure there's a period at the end
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
        
    return summary.strip()

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
            # Instead of adding method as an instruction, just include it directly in the text
            # This helps prevent instruction leakage
            input_text = context
            if method_section:
                # Include method information as part of the text, not as an instruction
                if "we propose" not in input_text.lower() and "proposed method" not in input_text.lower():
                    input_text = input_text + " " + method_section
                
            summary = summarize_with_huggingface(input_text, api_key)
            
            if summary:
                # Apply post-processing to clean up any remaining issues
                return post_process_summary(summary)
                
            # Fallback to SMMRY web service
            logging.info("Using SMMRY web service as fallback...")
            smmry_summary = summarize_with_smmry(context)
            if smmry_summary:
                return post_process_summary(smmry_summary)
                
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
                
                # Add position context to the chunk, not as instructions
                chunk_with_context = f"Title: {title}\n\n"
                
                # Add positional context as part of the text, not as instructions
                if i == 0:
                    chunk_with_context += chunk
                elif i == len(chunks) - 1:
                    chunk_with_context += chunk
                else:
                    chunk_with_context += chunk
                    
                # If this contains the method section, make sure it's included directly
                if i == method_chunk_idx and method_section and method_section not in chunk_with_context:
                    chunk_with_context += " " + method_section
                
                # Try HuggingFace API first
                chunk_summary = summarize_with_huggingface(chunk_with_context, api_key)
                
                if chunk_summary:
                    chunk_summaries.append(post_process_summary(chunk_summary))
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
            
            # Final pass to create a coherent summary from the combined text
            # Avoiding any instructional language that might leak
            final_context = f"{title}\n\n{combined_text}"
            
            final_summary = summarize_with_huggingface(final_context, api_key)
            if final_summary:
                return post_process_summary(final_summary)
            
            # If API fails for final summary, use the combined text with post-processing
            return post_process_summary(combined_text)
        else:
            # First level: Summarize each chunk individually
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                
                # Add context about what this chunk represents and the paper title
                chunk_context = f"Title: {title}\n\n"
                
                # Direct inclusion of text
                chunk_context += chunk
                
                # If this chunk contains the method section, ensure it's included
                if i == method_chunk_idx and method_section and method_section not in chunk_context:
                    chunk_context += " " + method_section
                
                # Try HuggingFace API first
                chunk_summary = summarize_with_huggingface(chunk_context, api_key)
                
                if chunk_summary:
                    chunk_summaries.append(post_process_summary(chunk_summary))
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
            
            # Final summarization with combined text
            meta_context = f"Title: {title}\n\n{combined_text}"
            
            # Include method information directly in the text
            if method_section and method_section not in meta_context:
                meta_context += " " + method_section
            
            final_summary = summarize_with_huggingface(meta_context, api_key)
            if final_summary:
                return post_process_summary(final_summary)
            
            # If second-level summarization fails, use extractive approach on combined text
            return post_process_summary(extract_key_sentences(combined_text))
    except Exception as e:
        logging.error(f"Error in summarize_abstract_with_huggingface: {str(e)}")
        return extract_key_sentences(abstract)
    
    # If HuggingFace API fails, try SMMRY web service
    logging.info("Using SMMRY web service as fallback...")
    smmry_summary = summarize_with_smmry(context)
    if smmry_summary:
        return post_process_summary(smmry_summary)
    
    # Final fallback: Use improved extractive summarization
    logging.info("Using extractive summarization as final fallback...")
    return post_process_summary(extract_key_sentences(cleaned_abstract))

def save_summaries_to_markdown(summaries, output_path):
    """Save the paper summaries in a markdown file."""
    # Track papers with non-standard layouts
    non_standard_layout_papers = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# AI-Generated Paper Summaries\n\n")
        f.write("*Summaries of papers in the downloaded_papers folder*\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        for paper in summaries:
            f.write(f"## {paper['title']}")
            
            # Add layout badge for papers with figures before abstract
            if paper.get('has_figures_before_abstract'):
                f.write(" 📊\n\n")
                f.write("*This paper has figures appearing before the abstract*\n\n")
                non_standard_layout_papers.append(paper["title"])
            else:
                f.write("\n\n")
            
            # Add method description if available
            if paper.get('method_description'):
                f.write("### Proposed Method\n\n")
                # Ensure method description doesn't contain any markdown formatting
                # except for the method name which should stay bold
                method_desc = paper['method_description']
                # Keep only the specific bold method name if it exists
                if "**" in method_desc:
                    # The method name might be in bold, preserve just that instance
                    name_pattern = re.compile(r'\*\*(.*?)\*\*')
                    method_name_match = name_pattern.search(method_desc)
                    if method_name_match:
                        method_name = method_name_match.group(1)
                        # Replace with non-bold version first
                        method_desc = method_desc.replace(f"**{method_name}**", method_name)
                else:
                    # Remove all bold/emphasis markers
                    method_desc = method_desc.replace('**', '').replace('__', '')
                
                # Fix ML acronyms
                method_desc = fix_ml_acronyms(method_desc)
                
                # Format with nice wrapping
                wrapped_method = textwrap.fill(method_desc, width=80)
                # Create a blockquote for the method description to visually distinguish it
                # Prefix each line with "> " to create a blockquote in markdown
                quoted_method = '\n'.join([f"> {line}" for line in wrapped_method.split('\n')])
                f.write(f"{quoted_method}\n\n")
            
            # Add keywords if available
            if paper.get('keywords'):
                f.write("### Keywords\n\n")
                keywords_str = ", ".join(paper['keywords'])
                f.write(f"{keywords_str}\n\n")
            
            if paper.get('abstract'):
                f.write("### Original Abstract\n\n")
                # Ensure abstract doesn't contain any markdown formatting characters
                clean_abstract = paper['abstract']
                # Remove any markdown formatting completely
                clean_abstract = clean_abstract.replace('**', '').replace('__', '').replace('*', '').replace('_', ' ')
                # Fix ML acronyms
                clean_abstract = fix_ml_acronyms(clean_abstract)
                # Format abstract with nice wrapping
                wrapped_abstract = textwrap.fill(clean_abstract, width=80)
                # Create a blockquote for the abstract to visually distinguish it
                # Prefix each line with "> " to create a blockquote in markdown
                quoted_abstract = '\n'.join([f"> {line}" for line in wrapped_abstract.split('\n')])
                f.write(f"{quoted_abstract}\n\n")
            
            if paper.get('summary'):
                f.write("### Summary\n\n")
                # Ensure summary doesn't contain any unintended markdown formatting
                clean_summary = paper['summary']
                # Remove any markdown formatting completely
                clean_summary = clean_summary.replace('**', '').replace('__', '').replace('*', '').replace('_', ' ')
                # Fix ML acronyms
                clean_summary = fix_ml_acronyms(clean_summary)
                # Format summary with nice wrapping
                wrapped_summary = textwrap.fill(clean_summary, width=80)
                # Create a blockquote for the summary to visually distinguish it
                quoted_summary = '\n'.join([f"> {line}" for line in wrapped_summary.split('\n')])
                f.write(f"{quoted_summary}\n\n")
            else:
                f.write("*No summary available*\n\n")
            
            # Add paper layout information if available
            if paper.get('abstract_position') or paper.get('has_figures_before_abstract'):
                f.write("### Paper Layout\n\n")
                layout_info = []
                
                if paper.get('abstract_position') and paper.get('abstract_position') != "unknown":
                    layout_info.append(f"Abstract position: {paper.get('abstract_position')}")
                    
                if paper.get('has_figures_before_abstract'):
                    layout_info.append("Has figures before abstract: Yes")
                
                for item in layout_info:
                    f.write(f"- {item}\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # Add summary statistics at the end
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total papers processed: {len(summaries)}\n")
        f.write(f"- Papers with abstracts: {sum(1 for s in summaries if s.get('abstract'))}\n")
        f.write(f"- Papers with summaries: {sum(1 for s in summaries if s.get('summary'))}\n")
        
        # Add non-standard layout statistics
        if non_standard_layout_papers:
            f.write(f"- Papers with figures before abstract: {len(non_standard_layout_papers)}\n\n")
            
            # List non-standard layout papers
            f.write("### Papers with Non-Standard Layouts\n\n")
            for paper_title in non_standard_layout_papers:
                f.write(f"- {paper_title}\n")
            f.write("\n")
    
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
            --layout-badge: #9b59b6;
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
            font-weight: bold;
        }
        
        h2 {
            color: var(--secondary-color);
            margin-top: 40px;
            font-size: 1.8em;
            border-left: 4px solid var(--secondary-color);
            padding-left: 15px;
            font-weight: bold;
        }
        
        h3 {
            color: var(--primary-color);
            font-size: 1.3em;
            margin-top: 25px;
            border-bottom: 1px dotted var(--border-color);
            padding-bottom: 5px;
            font-weight: bold;
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
        
        .paper-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        
        .paper-title {
            flex: 1;
            margin: 0;
        }
        
        .layout-badge {
            display: inline-block;
            background-color: var(--layout-badge);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
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
            font-weight: normal;
        }
        
        .summary {
            background-color: var(--medium-bg);
            padding: 20px;
            border-radius: 8px;
            text-align: justify;
            line-height: 1.8;
            font-size: 1.05em;
            border-left: 3px solid var(--accent-color);
            font-weight: normal;
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
            font-weight: normal;
        }
        
        .method-description strong {
            color: var(--method-border);
            font-weight: bold;
        }
        
        .keywords {
            background-color: #f0f7fa;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            font-weight: normal;
        }
        
        .keywords span {
            display: inline-block;
            background-color: #e1e8ed;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            color: var(--primary-color);
            font-weight: normal;
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
            font-weight: normal;
        }
        
        .section-title {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            font-weight: bold;
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
        
        .paper-metadata {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 10px 15px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #666;
        }
        
        .paper-metadata ul {
            list-style-type: none;
            padding: 0;
            margin: 5px 0;
        }
        
        .paper-metadata li {
            margin: 3px 0;
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
        
        /* Fix for ML acronyms to prevent them from being broken */
        .no-break {
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <h1>AI-Generated Paper Summaries</h1>
    <p class="generation-info">Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
    
    # Track papers with non-standard layouts
    non_standard_layout_papers = []
    
    for paper in summaries:
        html_content += f'<div class="paper-container">\n'
        
        # Create header with title and possibly layout badge
        html_content += f'    <div class="paper-header">\n'
        html_content += f'        <h2 class="paper-title">{paper["title"]}</h2>\n'
        
        # Add layout badge if this paper has figures before abstract
        if paper.get('has_figures_before_abstract'):
            html_content += f'        <span class="layout-badge">Figures before Abstract</span>\n'
            non_standard_layout_papers.append(paper["title"])
            
        html_content += f'    </div>\n'
        
        # Add keywords if available
        if paper.get('keywords'):
            html_content += f'    <div class="section-title"><h3>Keywords</h3></div>\n'
            html_content += f'    <div class="keywords">\n'
            for keyword in paper['keywords']:
                html_content += f'        <span>{keyword}</span>\n'
            html_content += f'    </div>\n'
        
        # Add method description if available
        if paper.get('method_description'):
            html_content += f'    <div class="section-title method-section-title"><h3>Proposed Method</h3></div>\n'
            # Clean any existing formatting
            method_desc = paper['method_description']
            # Only keep bold for the method name if it exists
            if "**" in method_desc:
                # Replace markdown bold with HTML strong
                method_desc = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', method_desc)
            else:
                # Remove any remaining markdown formatting
                method_desc = method_desc.replace('**', '').replace('__', '')
                
            # Fix ML acronyms
            method_desc = fix_ml_acronyms(method_desc)
                
            html_content += f'    <div class="method-description">{method_desc}</div>\n'
        # Fallback to extracting method from abstract if not found directly
        elif paper.get('abstract'):
            method_desc = extract_method_description(paper['abstract'])
            if method_desc:
                html_content += f'    <div class="section-title method-section-title"><h3>Proposed Method</h3></div>\n'
                # Replace markdown bold with HTML strong only for method name
                if "**" in method_desc:
                    method_desc = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', method_desc)
                else:
                    method_desc = method_desc.replace('**', '').replace('__', '')
                
                # Fix ML acronyms
                method_desc = fix_ml_acronyms(method_desc)
                
                html_content += f'    <div class="method-description">{method_desc}</div>\n'
        
        # Add original abstract
        if paper.get('abstract'):
            html_content += f'    <div class="section-title"><h3>Original Abstract</h3></div>\n'
            # Clean abstract of any potential formatting
            clean_abstract = paper['abstract'].replace('**', '').replace('__', '').replace('*', '').replace('_', ' ')
            # Fix ML acronyms
            clean_abstract = fix_ml_acronyms(clean_abstract)
            
            html_content += f'    <div class="abstract">{clean_abstract}</div>\n'
        
        # Add summary
        if paper.get('summary'):
            html_content += f'    <div class="section-title"><h3>Summary</h3></div>\n'
            # Clean summary of any potential formatting
            clean_summary = paper['summary'].replace('**', '').replace('__', '').replace('*', '').replace('_', ' ')
            # Fix ML acronyms
            clean_summary = fix_ml_acronyms(clean_summary)
            
            html_content += f'    <div class="summary">{clean_summary}</div>\n'
        
        # Add paper metadata including layout information
        if paper.get('abstract_position') or paper.get('has_figures_before_abstract'):
            html_content += f'    <div class="paper-metadata">\n'
            html_content += f'        <details>\n'
            html_content += f'            <summary>Paper Layout Information</summary>\n'
            html_content += f'            <ul>\n'
            
            if paper.get('abstract_position'):
                html_content += f'                <li>Abstract position: {paper.get("abstract_position", "unknown")}</li>\n'
                
            if paper.get('has_figures_before_abstract'):
                html_content += f'                <li>Has figures before abstract: Yes</li>\n'
                
            html_content += f'            </ul>\n'
            html_content += f'        </details>\n'
            html_content += f'    </div>\n'
        
        html_content += '</div>\n'
    
    # Add summary statistics at the bottom
    html_content += '<div class="paper-metadata">\n'
    html_content += '    <h3>Summary Statistics</h3>\n'
    html_content += '    <ul>\n'
    html_content += f'        <li>Total papers processed: {len(summaries)}</li>\n'
    html_content += f'        <li>Papers with abstracts: {sum(1 for s in summaries if s.get("abstract"))}</li>\n'
    html_content += f'        <li>Papers with summaries: {sum(1 for s in summaries if s.get("summary"))}</li>\n'
    
    # Add non-standard layout statistics
    if non_standard_layout_papers:
        html_content += f'        <li>Papers with figures before abstract: {len(non_standard_layout_papers)}</li>\n'
        
    html_content += '    </ul>\n'
    
    # List non-standard layout papers if any
    if non_standard_layout_papers:
        html_content += '    <details>\n'
        html_content += '        <summary>Papers with Non-Standard Layouts</summary>\n'
        html_content += '        <ul>\n'
        for paper_title in non_standard_layout_papers:
            html_content += f'            <li>{paper_title}</li>\n'
        html_content += '        </ul>\n'
        html_content += '    </details>\n'
        
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

def extract_method_description(text):
    """
    Extract a possible method description from the abstract or full text.
    Returns None if no method description can be confidently extracted.
    """
    # Don't process if text is None or empty
    if not text:
        return None
        
    # Look for common patterns that indicate method descriptions
    method_indicators = [
        r'we propose (.*?)(\.|\n)',
        r'we present (.*?)(\.|\n)',
        r'we introduce (.*?)(\.|\n)',
        r'our approach (.*?)(\.|\n)',
        r'our method (.*?)(\.|\n)',
        r'this paper proposes (.*?)(\.|\n)',
        r'this paper presents (.*?)(\.|\n)',
        r'this paper introduces (.*?)(\.|\n)',
        r'in this paper,?\s+we propose (.*?)(\.|\n)',
        r'in this paper,?\s+we present (.*?)(\.|\n)',
        r'in this paper,?\s+we introduce (.*?)(\.|\n)',
        r'we develop (.*?)(\.|\n)'
    ]
    
    for pattern in method_indicators:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            method_description = match.group(0)
            # Check if the description is substantial enough
            if len(method_description.split()) > 5:
                # Fix ML acronyms in the method description
                method_description = fix_ml_acronyms(method_description)
                return method_description
                
    # Look for a longer method description
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        # Check if this paragraph talks about the method
        method_words = ['method', 'approach', 'framework', 'system', 'technique', 'algorithm']
        if any(word in paragraph.lower() for word in method_words) and len(paragraph.split()) > 10:
            # Fix ML acronyms in the paragraph
            paragraph = fix_ml_acronyms(paragraph)
            return paragraph
            
    return None

def detect_figures_before_abstract(pdf_path):
    """
    Detect if a paper has figures that appear before the abstract section.
    This checks if the paper follows the "Figures before Abstract" pattern
    which is common in some conferences and journals.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if figures likely appear before abstract, False otherwise
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return False
            
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Define patterns
        abstract_pattern = r'(?i)^\s*Abstract\s*$|^\s*Abstract\s*[.:]|^\s*ABSTRACT\s*$|^\s*ABSTRACT\s*[.:]'
        figure_pattern = r'(?i)^(Figure|Fig\.)\s+\d+|^(Table)\s+\d+'
        
        abstract_section = re.compile(abstract_pattern)
        figure_section = re.compile(figure_pattern)
        
        # Find Abstract and any Figure/Table references in the first few paragraphs
        abstract_index = None
        figure_index = None
        
        # Only check the first 15 paragraphs
        for i, para in enumerate(paragraphs[:15]):
            if abstract_section.match(para):
                abstract_index = i
                break
            if figure_section.match(para) or re.search(r'(?i)(figure|fig\.|table)\s+\d+', para):
                figure_index = i
        
        # If we found a figure reference before the abstract
        if figure_index is not None and (abstract_index is None or figure_index < abstract_index):
            logging.info(f"Detected figures before abstract in {pdf_path}")
            return True
            
        # Alternative approach: check for keywords that suggest figures at the top
        first_paragraph = paragraphs[0] if paragraphs else ""
        if re.search(r'(?i)(overview|architecture|framework|pipeline|approach|model)\s+(figure|diagram|fig\.)', first_paragraph):
            logging.info(f"Detected likely figure reference in the first paragraph of {pdf_path}")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Error detecting figures before abstract in {pdf_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and summarize abstracts from academic papers.")
    parser.add_argument("--input-dir", default="downloaded_papers", help="Directory containing PDF files")
    parser.add_argument("--output", default="paper_summaries.md", help="Output file (markdown format)")
    parser.add_argument("--html", action="store_true", help="Generate HTML output instead of markdown")
    parser.add_argument("--hf-token", help="HuggingFace API token (optional, limited requests possible without it)")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()
    
    # Set up more detailed logging if debug is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Add more detailed formatting
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
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
        try:
            title = extract_title_from_filename(pdf_file)
            logging.info(f"Processing: {title}")
            
            # Check if this paper has figures before abstract
            has_figures_before_abstract = detect_figures_before_abstract(pdf_file)
            
            # Extract the abstract
            abstract = extract_abstract_from_pdf(pdf_file)
            if not abstract:
                logging.warning(f"Failed to extract abstract from {pdf_file}")
                continue
                
            # If paper has figures before abstract, include this information
            summary_info = {
                "title": title,
                "abstract": abstract,
                "pdf_path": pdf_file,
                "has_figures_before_abstract": has_figures_before_abstract,
            }
            
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
                "method_description": method_description,
                "has_figures_before_abstract": has_figures_before_abstract,
            }
            
            summaries.append(paper_info)
            
            # Add a small delay to avoid API rate limits
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    if args.html:
        save_summaries_to_html(summaries, output_path)
    else:
        save_summaries_to_markdown(summaries, output_path)
    
    # Print summary statistics
    abstracts_found = sum(1 for s in summaries if s.get('abstract'))
    abstracts_validated = sum(1 for s in summaries if s.get('abstract') and validate_abstract(s.get('abstract')))
    summaries_generated = sum(1 for s in summaries if s.get('summary'))
    papers_with_figures_before_abstract = sum(1 for s in summaries if s.get('has_figures_before_abstract'))
    
    logging.info("Summary extraction complete!")
    print("\nSummary Extraction Report:")
    print(f"Papers processed: {len(summaries)}")
    print(f"Abstracts successfully identified: {abstracts_found} ({abstracts_found/len(summaries)*100:.1f}%)")
    print(f"Abstracts passing validation: {abstracts_validated} ({abstracts_validated/len(summaries)*100:.1f}%)")
    print(f"Summaries generated: {summaries_generated} ({summaries_generated/len(summaries)*100:.1f}%)")
    print(f"Papers with figures before abstract: {papers_with_figures_before_abstract} ({papers_with_figures_before_abstract/len(summaries)*100:.1f}%)")
    
    # Report on papers with non-standard layouts
    if papers_with_figures_before_abstract > 0:
        print("\nPapers with figures before abstract:")
        for s in summaries:
            if s.get('has_figures_before_abstract'):
                print(f"  - {s['title']}")
    
    print(f"\nOutput saved to: {output_path}")
    
    if abstracts_found < len(summaries):
        print("\nTip: If some abstracts were not correctly identified, try running with a specific paper:")
        print("  python summarize.py --input-dir path/to/specific_paper_folder")
        
    print("\nThe script now uses enhanced abstract extraction techniques to better identify")
    print("and clean academic paper abstracts across various formats including arXiv and")
    print("conference papers. Abstracts are validated to ensure they contain actual abstract")
    print("content rather than other sections of the paper.")
    print("\nThe script can now handle papers with figures appearing before the abstract,")
    print("such as in the Magma paper and other conference paper formats with non-standard layouts.")

if __name__ == "__main__":
    main()
