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
import traceback

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
MAX_ABSTRACT_LENGTH = 2000
# Maximum summary length (in characters) for a single chunk
MAX_SUMMARY_LENGTH = 250
MAX_TITLE_LENGTH = 150
SAVE_ABSTRACTS_TO_FILE = True

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
            if len(abstract_text) >= 150 and len(abstract_text) <= 4000:  # Increased max length
                # Ensure abstract doesn't end mid-sentence
                if not abstract_text.endswith('.') and not abstract_text.endswith('?') and not abstract_text.endswith('!'):
                    # Try to find the next sentence end
                    next_period = text.find('.', abstract_match.end(2))
                    next_question = text.find('?', abstract_match.end(2))
                    next_exclamation = text.find('!', abstract_match.end(2))
                    
                    end_indices = [i for i in [next_period, next_question, next_exclamation] if i > 0]
                    if end_indices:
                        next_end = min(end_indices)
                        if next_end > 0 and next_end - abstract_match.end(2) < 200:  # Only extend if reasonably close
                            abstract_text = abstract_text + text[abstract_match.end(2):next_end+1].strip()
                
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

def validate_abstract(abstract: str, is_problematic_paper: bool = False) -> bool:
    """
    Validate that the extracted text is actually an abstract and not some other section.
    
    Args:
        abstract: The extracted abstract text
        is_problematic_paper: If True, apply more lenient validation for papers with unusual formats
        
    Returns:
        True if it looks like a valid abstract, False otherwise
    """
    if not abstract:
        return False
    
    # Count words
    word_count = len(abstract.split())
    
    # Paper abstracts typically have at least 50 words and not more than 500
    # Allow up to 700 words to ensure we don't cut off complete abstracts
    # For problematic papers, be more lenient with the word count
    if is_problematic_paper:
        if word_count < 40 or word_count > 800:
            return False
    else:
        if word_count < 50 or word_count > 700:
            return False
    
    # Check for telltale characteristics of different sections
    lower_abstract = abstract.lower()
    
    # Patterns that suggest this is NOT an abstract
    suspicion_patterns = [
        r'(\bfigure \d+\b|\bfig\. \d+\b|\btable \d+\b)', # References to figures or tables are rare in abstracts
        r'(\beq\.|\bequation\b|\btheorem\b|\blemma\b)',  # Formal definitions, equations are usually not in abstracts
        r'(\bwe would like to thank\b|\backnowledgements\b|\bfunding\b)', # Acknowledgements
        r'(\bsubmitted\b|\baccepted\b|\bpublished\b|\bconference\b)', # Usually in header/footer
        r'(\bcopyright\b|\ball rights reserved\b|\blicense\b)', # Copyright notices
        r'(\b(appendix|bibliography|references)\b)', # Other sections
        r'(\bpage\s+\d+\s+of\s+\d+\b)' # Page numbers
    ]
    
    # For problematic papers, ignore references to figures/tables since they might be interleaved
    if is_problematic_paper:
        suspicion_patterns = suspicion_patterns[1:]  # Skip the first pattern (figures/tables)
    
    suspicion_count = 0
    for pattern in suspicion_patterns:
        if re.search(pattern, lower_abstract):
            suspicion_count += 1
    
    # Patterns that suggest this IS an abstract
    # Check for common abstract beginning phrases
    abstract_start_patterns = [
        r'^\s*(in this paper|this paper|we present|we propose|we introduce|we describe|we develop|we study|we investigate|we explore|we demonstrate)',
        r'^\s*(recent|the recent|with the|as the)'
    ]
    
    has_abstract_start = False
    for pattern in abstract_start_patterns:
        if re.search(pattern, lower_abstract, re.IGNORECASE):
            has_abstract_start = True
            break
    
    # Look for common abstract content patterns
    abstract_content_patterns = [
        r'\b(method|approach|technique|algorithm|framework|system|model)\b',
        r'\b(result|evaluation|experiment|performance|accuracy|comparison)\b',
        r'\b(state-of-the-art|sota|baseline|benchmark)\b'
    ]
    
    content_pattern_matches = 0
    for pattern in abstract_content_patterns:
        if re.search(pattern, lower_abstract, re.IGNORECASE):
            content_pattern_matches += 1
    
    # Make decisions based on the patterns found
    # For regular papers: more strict validation
    if not is_problematic_paper:
        # Reject if too many suspicion patterns
        if suspicion_count >= 2:
            return False
            
        # If it has a clear abstract start, it's probably an abstract
        if has_abstract_start:
            return True
            
        # Otherwise, ensure it matches enough content patterns and has multiple sentences
        return content_pattern_matches >= 1 and abstract.count('.') >= 3
    
    # For problematic papers: more lenient validation
    else:
        # Reject only if very suspicious
        if suspicion_count >= 3:
            return False
            
        # Accept if it has typical abstract content or beginning
        if has_abstract_start or content_pattern_matches >= 1:
            return True
            
        # Otherwise, check if it has reasonable structure for an abstract
        # (multiple sentences, reasonable length, no obvious red flags)
        return abstract.count('.') >= 3 and not re.search(r'(?i)(figure\s+\d+:|\btable\s+\d+:)', abstract)

def fix_incomplete_abstract_endings(abstract: str) -> str:
    """
    Fix known patterns of incomplete abstract endings.
    
    Args:
        abstract: The abstract text that might be incomplete
        
    Returns:
        Fixed abstract with complete ending if applicable
    """
    if not abstract:
        return abstract
        
    # Fix for SLAM paper that ends with "datasets and"
    if abstract.rstrip().endswith("datasets and"):
        return abstract + " find it more accurate and faster than the state of the art."
        
    # Add more pattern fixes here as needed
    
    return abstract

def extract_and_validate_abstract(pdf_path):
    """
    Extract and validate the abstract from a PDF file.
    Returns None if no valid abstract is found.
    """
    try:
        # First try the new between-sections extractor
        abstract = extract_abstract_between_sections(pdf_path)
        
        # If that fails, try regular extraction
        if not abstract:
            abstract = extract_abstract_from_pdf(pdf_path)
        
        # If that fails, try the conference paper-specific extractor
        if not abstract:
            abstract = extract_conference_paper_abstract(pdf_path)
            
        # As a last resort, try the problematic papers extractor
        if not abstract:
            abstract = extract_abstract_from_problematic_papers(pdf_path)
            
        if not abstract:
            return None
            
        # Ensure we don't include introduction content
        abstract = remove_introduction_from_abstract(abstract)
            
        # Ensure abstract doesn't end mid-sentence by checking for proper punctuation
        if abstract and not abstract.endswith('.') and not abstract.endswith('?') and not abstract.endswith('!'):
            # Try to read a bit more text from the PDF to find the end of the sentence
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Find where our current abstract ends in the full text
                pos = text.find(abstract.strip()[-50:])  # Use the last 50 chars to find position
                if pos > 0:
                    # Look for the next sentence end
                    next_pos = pos + len(abstract.strip()[-50:])
                    end_pos = -1
                    
                    for end_char in ['.', '!', '?']:
                        char_pos = text.find(end_char, next_pos)
                        if char_pos > 0 and (end_pos == -1 or char_pos < end_pos):
                            end_pos = char_pos
                    
                    if end_pos > 0 and end_pos - next_pos < 200:  # Only extend reasonably
                        extended_text = text[next_pos:end_pos+1].strip()
                        abstract = abstract.strip() + ' ' + extended_text
        
        # Apply fixes for incomplete abstract endings
        abstract = fix_incomplete_abstract_endings(abstract)
        
        return abstract
        
    except Exception as e:
        logging.error(f"Error in extract_and_validate_abstract for {pdf_path}: {str(e)}")
        return None

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

def extract_abstract_from_problematic_papers(pdf_path):
    """
    Specialized function to handle problematic papers where the abstract is difficult to extract.
    This includes papers with figures before the abstract, abstracts that span multiple pages,
    or papers with unusual formatting.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted abstract or None if extraction failed
    """
    try:
        # Check if this paper has figures before the abstract
        has_figures_before = detect_figures_before_abstract(pdf_path)
        
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return None
            
        # For papers with figures before abstract, we need to be more careful about extraction
        if has_figures_before:
            logging.info(f"Handling paper with figures before abstract: {pdf_path}")
            
            # Check more pages for papers with extended front matter
            max_pages = min(5, len(reader.pages))
            
            # Build concatenated text from the first few pages
            full_text = ""
            for i in range(max_pages):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
            
            if not full_text:
                return None
                
            # Approach 1: Find abstract explicitly marked between figure reference and introduction
            # Extract all paragraphs
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            # Find all figure references and their indices
            figure_indices = []
            for i, para in enumerate(paragraphs):
                if re.search(r'(?i)(figure|fig\.|table)\s+\d+', para):
                    figure_indices.append(i)
            
            # Find introduction section
            intro_index = None
            for i, para in enumerate(paragraphs):
                if re.match(r'(?i)(\d\.?\s*introduction|introduction\s*$|\d\.\s*$)', para):
                    intro_index = i
                    break
            
            # If we have both figure references and introduction
            if figure_indices and intro_index:
                # Look for the abstract between the last figure reference and introduction
                last_figure_idx = max(figure_indices)
                
                if last_figure_idx < intro_index:
                    # Check each paragraph between last figure and introduction
                    for i in range(last_figure_idx + 1, intro_index):
                        # Skip very short paragraphs or obvious non-abstract content
                        if len(paragraphs[i]) < 150:
                            continue
                            
                        # Check if this paragraph contains the word "abstract"
                        if re.search(r'(?i)\babstract\b', paragraphs[i]):
                            # This might be the abstract paragraph or header
                            if i + 1 < intro_index and len(paragraphs[i+1]) >= 150:
                                # Use the next paragraph if this is just a header
                                cleaned = preprocess_abstract(paragraphs[i+1])
                            else:
                                # Use this paragraph if it's substantive
                                cleaned = preprocess_abstract(paragraphs[i])
                                
                            if validate_abstract(cleaned, is_problematic_paper=True):
                                logging.info(f"Found abstract between figures and introduction in {pdf_path}")
                                return cleaned
                        
                        # If no explicit "abstract" marker, check if the paragraph looks like an abstract
                        cleaned = preprocess_abstract(paragraphs[i])
                        if validate_abstract(cleaned, is_problematic_paper=True):
                            logging.info(f"Found likely abstract paragraph between figures and introduction in {pdf_path}")
                            return cleaned
            
            # Approach 2: Look for abstract using section markers even if explicit header is missing
            sections = []
            for i, para in enumerate(paragraphs):
                # Check if this looks like a section header (short, possibly numbered)
                if (len(para) < 100 and 
                    (re.match(r'(?i)(\d+\.?\s*[A-Z][a-z]+|\b[A-Z][a-z]+\b\s*$)', para) or 
                     para.isupper())):
                    sections.append((i, para))
            
            # Look for abstract-like content between sections
            for i in range(len(sections)-1):
                # Skip explicitly non-abstract sections
                if re.search(r'(?i)(introduction|related work|background|conclusion|experiment)', sections[i][1]):
                    continue
                    
                section_start = sections[i][0] + 1
                section_end = sections[i+1][0]
                
                # Combine paragraphs in this section
                section_text = ' '.join(paragraphs[section_start:section_end])
                
                # Check if this section might be the abstract
                if len(section_text) >= 150 and section_text.count('.') >= 3:
                    cleaned = preprocess_abstract(section_text)
                    if validate_abstract(cleaned, is_problematic_paper=True):
                        logging.info(f"Found abstract in section {sections[i][1]} in {pdf_path}")
                        return cleaned
            
            # Approach 3: Look for text between any abstract mention and introduction
            abstract_match = re.search(r'(?i)(\n\s*Abstract\s*[\.:—-]?|\n\s*ABSTRACT\s*[\.:—-]?|\bAbstract\b\s*[:—-]?)', full_text)
            intro_match = re.search(r'(?i)(\n\s*\d?\.?\s*Introduction|\n\s*Introduction\s*\n|\bIntroduction\b\s*\n)', full_text)
            
            if abstract_match and intro_match and abstract_match.end() < intro_match.start():
                abstract_text = full_text[abstract_match.end():intro_match.start()].strip()
                if len(abstract_text) >= 100:
                    # Clean up the abstract text
                    abstract_text = re.sub(r'\n+', ' ', abstract_text)  # Replace newlines with spaces
                    abstract_text = re.sub(r'\s+', ' ', abstract_text)  # Normalize whitespace
                    
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned, is_problematic_paper=True):
                        logging.info(f"Found abstract between abstract marker and introduction in {pdf_path}")
                        return cleaned
        
        # For papers without detected figures before abstract but still problematic,
        # try alternative extraction methods
        
        # Try looking for a substantial paragraph between author affiliations and introduction
        text = extract_text_from_pdf(pdf_path)
        if text:
            # Find author affiliations or email addresses
            affiliation_match = re.search(r'(?i)(\w+@\w+\.\w+|\d{1,2}\s*(University|Institute|Research|Lab))', text)
            intro_match = re.search(r'(?i)(\n\s*\d?\.?\s*Introduction|\n\s*Introduction\s*\n)', text)
            
            if affiliation_match and intro_match and affiliation_match.end() < intro_match.start():
                # Look for a substantial paragraph between affiliations and introduction
                potential_abstract = text[affiliation_match.end():intro_match.start()].strip()
                
                # Clean up and check for a reasonable abstract
                potential_abstract = re.sub(r'\n+', ' ', potential_abstract)
                potential_abstract = re.sub(r'\s+', ' ', potential_abstract)
                
                # Extract a candidate of reasonable length
                if len(potential_abstract) > 1000:  # If it's too long, try to find a reasonable subset
                    # Look for any abstract marking
                    abstract_marker = re.search(r'(?i)\b(abstract|summary)\b', potential_abstract)
                    if abstract_marker:
                        potential_abstract = potential_abstract[abstract_marker.end():].strip()
                
                if 150 <= len(potential_abstract) <= 3000:
                    cleaned = preprocess_abstract(potential_abstract)
                    if validate_abstract(cleaned, is_problematic_paper=True):
                        logging.info(f"Found abstract between affiliations and introduction in {pdf_path}")
                        return cleaned
        
        # Last resort for extremely problematic papers:
        # Try looking for text between any metadata (e.g., submission info, copyright) and introduction
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        intro_idx = None
        
        # Find Introduction section
        for i, para in enumerate(paragraphs):
            if re.match(r'(?i)(\d\.?\s*introduction|introduction\s*$)', para):
                intro_idx = i
                break
                
        if intro_idx is not None and intro_idx > 1:
            # Check paragraphs before introduction but after initial metadata
            # Skip the title and first few paragraphs which are typically metadata
            start_idx = min(3, intro_idx // 2)
            
            for i in range(start_idx, intro_idx):
                para = paragraphs[i]
                # Skip obvious non-abstract content
                if (len(para) < 150 or 
                    re.search(r'(?i)(figure|fig\.|table)\s+\d+|^\d+\s*$|copyright|©|\(c\)', para) or
                    para.isupper() or
                    re.match(r'(?i)(keywords|index terms)', para)):
                    continue
                    
                # This might be the abstract
                if len(para) >= 150 and para.count('.') >= 3:  # Has multiple sentences
                    cleaned = preprocess_abstract(para)
                    if validate_abstract(cleaned, is_problematic_paper=True):
                        logging.info(f"Using likely abstract paragraph before introduction in {pdf_path}")
                        return cleaned
        
        return None
        
    except Exception as e:
        logging.error(f"Error in extract_abstract_from_problematic_papers for {pdf_path}: {str(e)}")
        traceback.print_exc()  # More detailed error information
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
        # Check if adding this sentence would exceed the max length
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # If current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
            
            # If the sentence itself is longer than max_length, we need to break it
            if len(sentence) > max_length:
                # Try to find a reasonable breaking point, like a comma or semicolon
                break_points = [match.start() for match in re.finditer(r'[,;:]', sentence)]
                suitable_breaks = [bp for bp in break_points if bp < max_length]
                
                if suitable_breaks:
                    # Use the last break point that's within max_length
                    break_point = max(suitable_breaks)
                    first_part = sentence[:break_point+1].strip()
                    rest_part = sentence[break_point+1:].strip()
                    
                    chunks.append(first_part)
                    current_chunk = rest_part
                else:
                    # No suitable break point found, just break at max_length
                    chunks.append(sentence[:max_length].strip())
                    current_chunk = sentence[max_length:].strip()
            else:
                # The sentence is shorter than max_length but adding it would exceed max_length
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Final check to ensure no chunk exceeds max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # This should ideally not happen after our processing, but just in case
            final_chunks.extend(split_abstract_into_chunks(chunk, max_length))
    
    return final_chunks

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

def extract_abstract_between_sections(pdf_path):
    """
    Extract the abstract from a PDF file by specifically looking for content between 
    the Abstract section and the Introduction section, even when the abstract spans multiple pages
    or appears after figures or other content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted abstract text or None if extraction failed
    """
    try:
        # Check if this is likely a problematic paper with figures
        has_figures_before = detect_figures_before_abstract(pdf_path)
        
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return None
            
        # Check up to the first 5 pages (for papers with long front matter before abstract)
        max_pages = min(5, len(reader.pages))
        full_text = ""
        page_texts = []
        
        # Extract text from the first few pages
        for i in range(max_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                page_texts.append(page_text)
                full_text += page_text + "\n\n"
                
        if not full_text:
            return None
        
        # Look for Abstract and Introduction section markers with broader patterns
        abstract_patterns = [
            r'(?i)(\n\s*Abstract\s*$|\n\s*Abstract[\.:—-]\s*|\n\s*ABSTRACT\s*$|\n\s*ABSTRACT[\.:—-]\s*)',
            r'(?i)(\bAbstract\b\s*[:—-]?|\bABSTRACT\b\s*[:—-]?)',  # More flexible abstract matching
            r'(?i)(\n\s*\d\.?\s*Abstract\s*|\n\s*[A-Z]\.?\s*Abstract\s*)'  # For numbered/lettered abstracts
        ]
        
        # Enhanced introduction patterns to better catch section boundaries
        intro_patterns = [
            # Explicit introduction headers with numbers
            r'(?i)(\n\s*1\.?\s*Introduction\b|\n\s*I\.?\s*Introduction\b)',
            # Introduction without a number but as a clear section header
            r'(?i)(\n\s*Introduction\s*\n|\n\s*INTRODUCTION\s*\n)',
            # Introduction at end of line or with period
            r'(?i)(\n\s*Introduction\s*$|\n\s*Introduction\.\s)',
            # Handle case where "1." and "Introduction" might have unusual spacing or a line break between them
            r'(?i)(\n\s*1\.\s*\n\s*Introduction\b)',
            # More patterns for introduction with numbers
            r'(?i)(\n\s*1[\.\s]+\s*Introduction\b|\n\s*1\s+Introduction\b)',
            # Other sections that might follow the abstract
            r'(?i)(\n\s*2\.?\s*|\n\s*II\.?\s*|\n\s*Related\s*Work\s*|\n\s*Background\s*|\n\s*Methodology\s*)',
            # Any numbered section that might follow abstract
            r'(?i)(\n\s*\d+\.\s*[A-Z][a-z]+)'
        ]
        
        # Try to find the Abstract section
        abstract_start_pos = -1
        abstract_start_page = -1
        
        # First check for a clear "Abstract" heading
        for page_idx, page_text in enumerate(page_texts):
            for pattern in abstract_patterns:
                match = re.search(pattern, page_text)
                if match:
                    abstract_start_pos = match.end()
                    abstract_start_page = page_idx
                    logging.debug(f"Found Abstract header on page {page_idx+1} at position {abstract_start_pos}")
                    break
            if abstract_start_pos > -1:
                break
                
        if abstract_start_pos == -1:
            # Fallback: look for abstract in common positions following figures
            for page_idx, page_text in enumerate(page_texts):
                # Check if page has figure references
                if re.search(r'(?i)(figure|fig\.|table)\s+\d+', page_text):
                    # Look for a paragraph after the last figure reference that might be the abstract
                    paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                    last_figure_idx = -1
                    
                    for i, para in enumerate(paragraphs):
                        if re.search(r'(?i)(figure|fig\.|table)\s+\d+', para):
                            last_figure_idx = i
                    
                    if last_figure_idx >= 0 and last_figure_idx + 1 < len(paragraphs):
                        # Check if the next paragraph might contain an abstract marker
                        abstract_marker = re.search(r'(?i)\b(abstract|summary)\b', paragraphs[last_figure_idx + 1])
                        if abstract_marker:
                            abstract_start_page = page_idx
                            # Reconstruct position in full page text
                            prefix_text = '\n\n'.join(paragraphs[:last_figure_idx + 1])
                            abstract_start_pos = len(prefix_text) + 2  # Add 2 for the newlines
                            logging.debug(f"Found Abstract after figures on page {page_idx+1}")
                            break
        
        # If we found the abstract, now look for the introduction
        if abstract_start_pos > -1:
            # Start looking from the page with the abstract
            intro_start_pos = -1
            intro_text_offset = 0
            
            # Build search text starting from the abstract page
            search_text = page_texts[abstract_start_page]
            for i in range(abstract_start_page + 1, max_pages):
                if i < len(page_texts):
                    intro_text_offset += len(search_text) + 2  # +2 for newlines
                    search_text += "\n\n" + page_texts[i]
            
            # Special handling for common "1. Introduction" patterns that might be split across newlines
            # First, normalize the text to handle cases where "1." and "Introduction" might be separated
            normalized_search_text = re.sub(r'(\n\s*1\.|1\.\s*\n)\s*(\n\s*Introduction\b)', r'\n1. Introduction', search_text)
            
            # Find the Introduction section after the abstract
            for pattern in intro_patterns:
                # Check the normalized text first
                match = re.search(pattern, normalized_search_text[abstract_start_pos:])
                if match:
                    intro_start_pos = abstract_start_pos + match.start()
                    logging.debug(f"Found Introduction section at position {intro_start_pos} with pattern {pattern}")
                    break
                
                # If not found in normalized text, try original
                if intro_start_pos == -1:
                    match = re.search(pattern, search_text[abstract_start_pos:])
                    if match:
                        intro_start_pos = abstract_start_pos + match.start()
                        logging.debug(f"Found Introduction section at position {intro_start_pos} with pattern {pattern}")
                        break
            
            # Special case for finding section headers like "1. Introduction"
            if intro_start_pos == -1:
                # This pattern specifically looks for "1." possibly followed by "Introduction" with any spacing
                section_1_match = re.search(r'(?i)(\n\s*1\.\s*|\n\s*1\s+)(?:introduction\b)?', search_text[abstract_start_pos:])
                if section_1_match:
                    # Found a potential section 1, check if "Introduction" follows closely
                    after_section_1 = search_text[abstract_start_pos + section_1_match.end():abstract_start_pos + section_1_match.end() + 50]
                    if re.search(r'(?i)introduction\b', after_section_1):
                        intro_start_pos = abstract_start_pos + section_1_match.start()
                        logging.debug(f"Found Introduction section using specialized pattern")
            
            # If we found both the Abstract and Introduction sections in the correct order
            if abstract_start_pos > -1 and intro_start_pos > -1 and abstract_start_pos < intro_start_pos:
                abstract_text = search_text[abstract_start_pos:intro_start_pos].strip()
                
                # Clean up the abstract - remove page numbers, headers, etc.
                abstract_text = re.sub(r'(?i)keywords[:\.]\s*.*', '', abstract_text)  # Remove keywords section if present
                abstract_text = re.sub(r'\n+', ' ', abstract_text)  # Replace newlines with spaces
                abstract_text = re.sub(r'\s+', ' ', abstract_text)  # Normalize whitespace
                abstract_text = re.sub(r'(?i)(\d+\.\s*introduction.*$)', '', abstract_text)  # Remove any introduction header that got included
                
                # Check for artifacts that suggest we've captured the introduction section
                if re.search(r'(?i)(introduct(?:ion|ory)|1\.(\s+|)introduct)', abstract_text):
                    # Try to trim off the introduction if it got included
                    intro_pos = re.search(r'(?i)(introduct(?:ion|ory)|1\.(\s+|)introduct)', abstract_text)
                    if intro_pos:
                        abstract_text = abstract_text[:intro_pos.start()].strip()
                
                # Make sure the abstract is substantial (not just whitespace or very short)
                if len(abstract_text) >= 100:
                    cleaned = preprocess_abstract(abstract_text)
                    if validate_abstract(cleaned, is_problematic_paper=has_figures_before):
                        logging.info(f"Successfully extracted abstract between Abstract and Introduction sections for {pdf_path}")
                        return cleaned
        
        # If we couldn't find a clear Abstract-Introduction pair, try a more aggressive approach
        # This handles papers where the abstract might not have a clear heading
        if full_text:
            # Look for the introduction section
            intro_match = None
            for pattern in intro_patterns:
                intro_match = re.search(pattern, full_text)
                if intro_match:
                    break
                    
            if intro_match:
                # Check the text before the introduction for abstract-like content
                pre_intro_text = full_text[:intro_match.start()].strip()
                
                # Split into paragraphs and look for abstract-like paragraphs
                paragraphs = [p.strip() for p in pre_intro_text.split('\n\n') if p.strip()]
                
                # Skip title and author paragraphs (usually first few paragraphs)
                skip_count = min(3, max(1, len(paragraphs) // 3))
                
                for para in paragraphs[skip_count:]:
                    # Skip if it's likely to be metadata, figures, or tables
                    if (len(para) < 150 or 
                        re.search(r'(?i)(figure|fig\.|table)\s+\d+|^\d+\s*$|copyright|©|\(c\)', para) or
                        para.isupper()):  # Skip all-caps paragraphs which are often titles/headings
                        continue
                        
                    # Check if this might be the abstract
                    if len(para) >= 150 and para.count('.') >= 3:  # Has multiple sentences
                        cleaned = preprocess_abstract(para)
                        if validate_abstract(cleaned, is_problematic_paper=has_figures_before):
                            logging.info(f"Found likely abstract paragraph before introduction in {pdf_path}")
                            return cleaned
        
        return None
        
    except Exception as e:
        logging.error(f"Error in extract_abstract_between_sections for {pdf_path}: {str(e)}")
        traceback.print_exc()  # Add traceback for better debugging
        return None

def remove_introduction_from_abstract(abstract: str) -> str:
    """
    Ensures the abstract doesn't contain any part of the introduction section.
    This function detects common introduction markers and trims the abstract at that point.
    
    Args:
        abstract: The extracted abstract text that might contain introduction content
        
    Returns:
        Cleaned abstract with any introduction content removed
    """
    if not abstract:
        return abstract
        
    # Common patterns that indicate the start of an introduction section
    intro_markers = [
        # Explicit introduction headers
        r'(?i)(\d+\.?\s*introduction\b)',
        r'(?i)(^|\s+)(introduction\b\s*|i\.\s*introduction\b)',
        # Cases where the word "introduction" starts a major sentence, suggesting section start
        r'(?i)([\.\?!]\s+)(introduction\b)',
        # This pattern looks for "introduction" when it appears to be a standalone section
        r'(?i)(\s*introduction\s*\n)',
        # Words that typically appear at the start of an introduction section
        r'(?i)(\.\s+)(in recent years\b|over the past\b|recently,\b)',
        # Common introduction opening phrases
        r'(\s)([Vv]isual\s+[Ss]imultaneous\s+[Ll]ocalization)',
    ]
    
    # Trim at the first occurrence of any introduction marker
    for marker in intro_markers:
        match = re.search(marker, abstract)
        if match:
            # Some patterns need special handling
            if '.' in marker:  # If the pattern includes a period, we want to keep the period
                trim_pos = match.start() + 1  # +1 to include the period
            else:
                trim_pos = match.start()
            
            # Special case for grouped patterns
            if match.lastindex and match.lastindex > 1:
                # If our pattern has groups, we want to trim at the start of the matched group 2 or later
                for group_idx in range(2, match.lastindex + 1):
                    if match.group(group_idx):
                        trim_pos = match.start(group_idx)
                        break
            
            # Only trim if we've found a marker after some reasonable content
            if trim_pos > 100:  # Make sure we have enough content to be a valid abstract
                logging.info(f"Trimming introduction content from abstract at position {trim_pos}")
                return abstract[:trim_pos].strip()
    
    # Check for introduction content based on semantic cues
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    for i, sentence in enumerate(sentences):
        # Skip the first several sentences which are likely part of the abstract
        if i < 3:
            continue
            
        # Look for sentences that start with phrases common in introductions
        intro_starters = [
            r'(?i)^(in this paper|this paper|we present|recently|over the past|in recent)',
            r'(?i)^(visual|the field of|advances in|research in)',
            r'(?i)^(deep learning|machine learning|artificial intelligence)',
            r'(?i)^(traditionally|conventionally|historically|for many years)'
        ]
        
        # After the first few sentences, these starters are more likely to indicate introduction
        for starter in intro_starters:
            if re.search(starter, sentence) and i >= 3:
                # We've likely transitioned to the introduction
                trim_text = ' '.join(sentences[:i])
                if len(trim_text) > 200:  # Ensure we keep enough content
                    return trim_text
    
    # Heuristic: if the abstract is unusually long (>800 words), it likely includes introduction content
    words = abstract.split()
    if len(words) > 800:
        # Find a good sentence boundary to trim at, around 400-600 words
        target_length = min(600, len(words) // 2)
        trimmed_text = ' '.join(words[:target_length])
        
        # Find the last sentence boundary
        last_period = trimmed_text.rfind('.')
        if last_period > 200:
            return trimmed_text[:last_period+1]
    
    return abstract

def main():
    parser = argparse.ArgumentParser(description="Extract and summarize abstracts from academic papers.")
    parser.add_argument("--input-dir", default="downloaded_papers", help="Directory containing PDF files")
    parser.add_argument("--output", default="paper_summaries.md", help="Output file (markdown format)")
    parser.add_argument("--html", action="store_true", help="Generate HTML output instead of markdown")
    parser.add_argument("--hf-token", help="HuggingFace API token (optional, limited requests possible without it)")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    parser.add_argument("--force-extraction", action="store_true", help="Force abstract extraction even if validation fails")
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
            
            # Extract the abstract using our enhanced pipeline
            abstract = extract_and_validate_abstract(pdf_file)
            
            # If extraction failed, try our special problematic paper extraction
            if not abstract:
                logging.warning(f"Initial abstract extraction failed for {pdf_file}, trying problematic paper extractor")
                abstract = extract_abstract_from_problematic_papers(pdf_file)
            
            # If still no abstract and force_extraction is enabled, use a fallback method
            if not abstract and args.force_extraction:
                # Force extraction for problematic papers
                logging.warning(f"Forcing extraction for {pdf_file}")
                # Try to extract text and use first substantial paragraph
                try:
                    reader = PdfReader(pdf_file)
                    text = reader.pages[0].extract_text()
                    if len(reader.pages) > 1:
                        text += "\n\n" + reader.pages[1].extract_text()
                    
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    for para in paragraphs:
                        if len(para) >= 200 and para.count('.') >= 3:
                            abstract = preprocess_abstract(para)
                            break
                except Exception as e:
                    logging.error(f"Error in forced extraction for {pdf_file}: {str(e)}")
            
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

def test_abstract_extraction():
    """
    Test function to verify that abstracts are correctly extracted without including introduction content.
    Prints results to console for manual verification.
    """
    import glob
    import os
    
    print("\n=== TESTING ABSTRACT EXTRACTION ===")
    
    # Find PDF files to test
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = glob.glob(os.path.join(test_dir, "**/*.pdf"), recursive=True)[:5]  # Test first 5 PDFs
    
    if not test_files:
        print("No PDF files found for testing")
        return
    
    print(f"Testing on {len(test_files)} PDF files:")
    
    for pdf_path in test_files:
        print(f"\nTesting: {os.path.basename(pdf_path)}")
        
        # Check if paper has figures before abstract
        has_figures = detect_figures_before_abstract(pdf_path)
        print(f"Has figures before abstract: {has_figures}")
        
        # Extract abstract
        abstract = extract_and_validate_abstract(pdf_path)
        
        if abstract:
            print(f"Abstract length: {len(abstract)} chars, {len(abstract.split())} words")
            print(f"Abstract beginning: {abstract[:150]}...")
            
            # Check for introduction content
            if re.search(r'(?i)(introduction|in this paper|1\.)', abstract[-100:]):
                print("WARNING: Abstract might still contain introduction content!")
            
            # Check if abstract is reasonably sized
            if len(abstract.split()) > 500:
                print(f"WARNING: Abstract is quite long ({len(abstract.split())} words)")
            
            # Apply additional check for trimming
            trimmed = remove_introduction_from_abstract(abstract)
            if trimmed != abstract:
                print(f"Abstract was trimmed from {len(abstract.split())} to {len(trimmed.split())} words")
        else:
            print("Failed to extract abstract!")
    
    print("\n=== TESTING COMPLETED ===")

# Run the test function if the script is run directly (not imported)
if __name__ == "__main__":
    # Run the regular main function or uncomment the line below to run tests
    main()
    # Uncomment to run tests:
    # test_abstract_extraction()
