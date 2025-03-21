"""
CVPR Paper Downloader

A tool to download CVPR papers based on keywords from the titles.

Usage examples:
    # Run interactively (prompt for keywords)
    python download.py --interactive
    
    # Download papers with specific keywords
    python download.py --keywords "Diffusion" "Text-to-Image"
    
    # Specify download directory and conference URL
    python download.py -k "Vision Language Model" -d "my_papers" -u "https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers"
"""

import requests
from bs4 import BeautifulSoup
import time
import os
import re
import difflib
from urllib.parse import quote
import logging
from typing import List, Tuple, Dict, Optional
import PyPDF2
import io
import random
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_download.log"),
        logging.StreamHandler()
    ]
)

# Create a session with retry logic
def create_robust_session(
    retries=5, 
    backoff_factor=0.5, 
    status_forcelist=[429, 500, 502, 503, 504],
    proxy=None
):
    """Create a requests session with retry capabilities."""
    session = requests.Session()
    
    # Configure proxy if provided
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy,
        }
    
    # Configure retry strategy
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
    )
    
    # Mount the adapter to both http and https
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Add a user agent to avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    return session

def setup_download_directory(directory: str = 'downloaded_papers') -> str:
    """Create and return the download directory path."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def normalize_title(title: str) -> str:
    """Normalize title for comparison by removing special characters and extra spaces."""
    if not title:
        return ""
    # Remove special characters, convert to lowercase, and remove extra spaces
    normalized = re.sub(r'[^\w\s]', ' ', title.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using sequence matcher."""
    if not title1 or not title2:
        return 0.0
    
    # Normalize titles
    norm_title1 = normalize_title(title1)
    norm_title2 = normalize_title(title2)
    
    # Use sequence matcher for similarity calculation
    return difflib.SequenceMatcher(None, norm_title1, norm_title2).ratio()

def get_cvpr_paper_titles(url: str) -> Tuple[List[str], List[str]]:
    """
    Scrape paper titles and authors from the CVPR webpage with improved error handling.
    
    Args:
        url: URL of the CVPR papers page
        
    Returns:
        Tuple of (titles list, authors list)
    """
    logging.info(f"Fetching papers from {url}")
    titles = []
    authors = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple potential HTML structures
        paper_rows = soup.find_all('tr')
        
        if not paper_rows:
            # Alternative structure: might be in divs or other containers
            paper_containers = soup.find_all('div', class_=['paper', 'paper-container', 'publication'])
            
            if paper_containers:
                for container in paper_containers:
                    title_elem = container.find(['h2', 'h3', 'strong', 'div'], class_=['title', 'paper-title'])
                    authors_elem = container.find(['div', 'p', 'span'], class_=['authors', 'paper-authors'])
                    
                    if title_elem:
                        titles.append(title_elem.text.strip())
                        if authors_elem:
                            authors.append(authors_elem.text.strip())
                        else:
                            authors.append("")
        else:
            # Original structure (table rows)
            for row in paper_rows:
                title_cell = row.find('td')
                if title_cell and title_cell.find('strong'):
                    title_text = title_cell.find('strong').text.strip()
                    titles.append(title_text)
                    
                    # Extract authors if available
                    author_div = title_cell.find('div', class_='indented')
                    if author_div and author_div.find('i'):
                        authors.append(author_div.find('i').text.strip())
                    else:
                        authors.append("")
        
        logging.info(f"Found {len(titles)} papers")
        if len(titles) == 0:
            logging.warning("No papers found. The webpage structure might have changed.")
        
        return titles, authors
    
    except Exception as e:
        logging.error(f"Error fetching papers: {str(e)}")
        return [], []

def has_keyword_in_title(title: str, keywords: List[str]) -> bool:
    """Check if the paper title contains any of the keywords (case insensitive)."""
    title_lower = title.lower()
    return any(keyword.lower() in title_lower for keyword in keywords)

def parse_arxiv_response(content) -> List[Dict]:
    """
    Parse arXiv API response content using different methods if XML parser fails.
    
    Args:
        content: Response content from arXiv API
        
    Returns:
        List of entries with title, pdf_url, and id
    """
    entries = []
    
    # Try multiple parsing approaches
    try:
        # First try to use lxml parser (most efficient for XML)
        try:
            soup = BeautifulSoup(content, 'xml')
        except:
            # If xml parser is not available, try lxml parser
            soup = BeautifulSoup(content, 'lxml')
            
        # Find all entry elements
        entry_elements = soup.find_all('entry')
        
        for entry in entry_elements:
            try:
                title_elem = entry.find('title')
                id_elem = entry.find('id')
                pdf_link = entry.find('link', {'title': 'pdf'})
                
                if title_elem and id_elem and pdf_link:
                    entries.append({
                        'title': title_elem.text.strip(),
                        'id': id_elem.text.strip().split('/')[-1],
                        'pdf_url': pdf_link['href']
                    })
            except Exception as e:
                logging.warning(f"Error parsing an entry: {e}")
                continue
                
    except Exception as e:
        # If all parsers fail, try a simple regex approach as last resort
        logging.warning(f"XML parsing failed: {e}. Trying regex fallback.")
        
        try:
            # Simple regex to extract basic info
            title_pattern = r'<title>(.*?)</title>'
            id_pattern = r'<id>(.*?)</id>'
            pdf_pattern = r'<link[^>]*title="pdf"[^>]*href="([^"]*)"'
            
            # Find all titles, ids, and pdf links
            titles = re.findall(title_pattern, content.decode('utf-8'))
            ids = re.findall(id_pattern, content.decode('utf-8'))
            pdf_urls = re.findall(pdf_pattern, content.decode('utf-8'))
            
            # Filter out the namespace entry
            entry_titles = [t for t in titles if "http://" not in t]
            
            # Create entries from matched patterns
            for i in range(min(len(entry_titles), len(ids), len(pdf_urls))):
                entries.append({
                    'title': entry_titles[i],
                    'id': ids[i].split('/')[-1],
                    'pdf_url': pdf_urls[i]
                })
        except Exception as e:
            logging.error(f"Regex fallback also failed: {e}")
    
    return entries

def search_arxiv_paper(title: str, authors: str = "", proxy=None, max_retries=5) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Search for papers on arXiv using the title and authors with improved matching.
    Uses retry logic to handle connection issues and rate limiting.
    
    Args:
        title: Paper title
        authors: Paper authors (optional)
        proxy: Optional proxy server to use for the request
        max_retries: Maximum number of manual retries (beyond session retries)
        
    Returns:
        Tuple of (pdf_url, arxiv_title, arxiv_id) or (None, None, None) if not found
    """
    logging.info(f"Searching arXiv for: {title}")
    
    # Clean the title for search
    query = normalize_title(title)
    query = '+'.join(query.split())
    
    # Add first author if available to improve search accuracy
    first_author = ""
    if authors:
        author_parts = re.split(r'[,;&]', authors)
        if author_parts:
            first_author = author_parts[0].strip()
            if first_author:
                query += f"+author:{quote(first_author)}"
    
    arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    
    # Create a session with retry capabilities
    session = create_robust_session(proxy=proxy)
    
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid hitting rate limits
            if attempt > 0:
                jitter = random.uniform(2, 5) * attempt
                logging.info(f"Retry attempt {attempt+1}/{max_retries} after {jitter:.2f}s")
                time.sleep(jitter)
            
            response = session.get(arxiv_url, timeout=30)
            response.raise_for_status()
            
            # Parse the response with our more robust parser
            entries = parse_arxiv_response(response.content)
            
            if not entries:
                logging.warning("No entries found in arXiv response")
                return None, None, None
            
            best_match = None
            best_similarity = 0.0
            
            for entry in entries:
                arxiv_title = entry['title']
                # Calculate similarity between original title and arXiv title
                similarity = title_similarity(title, arxiv_title)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            # Only consider it a match if similarity is above threshold
            if best_match and best_similarity > 0.6:
                pdf_url = best_match['pdf_url']
                arxiv_id = best_match['id']
                arxiv_title = best_match['title']
                logging.info(f"Found paper on arXiv (similarity: {best_similarity:.2f}): {arxiv_title}")
                return pdf_url, arxiv_title, arxiv_id
            else:
                logging.warning(f"No matching paper found on arXiv (best similarity: {best_similarity:.2f})")
                return None, None, None
                
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Connection error on attempt {attempt+1}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to connect to ArXiv after {max_retries} attempts")
                return None, None, None
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error on attempt {attempt+1}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to search ArXiv after {max_retries} attempts")
                return None, None, None
                
        except Exception as e:
            logging.error(f"Error searching arXiv: {str(e)}")
            logging.error("If you're getting 'Couldn't find a tree builder with the features you requested: xml', install lxml with: pip install lxml")
            return None, None, None

def sanitize_filename(title: str, arxiv_id: Optional[str] = None) -> str:
    """
    Create a clean, filesystem-safe filename from the paper title and arXiv ID.
    
    Args:
        title: Paper title
        arxiv_id: arXiv ID (optional)
        
    Returns:
        A sanitized filename
    """
    # First clean the title
    # Replace problematic characters with underscores
    clean_title = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', title)
    # Replace multiple spaces/underscores with a single underscore
    clean_title = re.sub(r'[\s_]+', '_', clean_title)
    # Trim to reasonable length to avoid too long filenames
    if len(clean_title) > 100:
        clean_title = clean_title[:100]
    
    # Add arXiv ID if available for uniqueness
    if arxiv_id:
        return f"{clean_title}_{arxiv_id}.pdf"
    else:
        return f"{clean_title}.pdf"

def download_paper(pdf_url: str, save_path: str, title: str, proxy=None) -> bool:
    """
    Download the paper PDF from arXiv with improved error handling.
    
    Args:
        pdf_url: URL of the PDF to download
        save_path: Path where to save the PDF
        title: Paper title (for logging)
        proxy: Optional proxy server to use for the request
        
    Returns:
        True if download was successful, False otherwise
    """
    if not pdf_url:
        logging.error(f"No PDF URL provided for paper: {title}")
        return False
    
    logging.info(f"Downloading paper: {title}")
    
    # Create a session with retry capabilities
    session = create_robust_session(
        retries=3,  # More retries for downloads
        backoff_factor=1.0,  # Longer backoff for downloads
        proxy=proxy
    )
    
    try:
        response = session.get(pdf_url, timeout=120)  # Even longer timeout for PDF downloads
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        # Verify the file was downloaded correctly (check if it's a valid PDF)
        if os.path.getsize(save_path) < 1000:  # Too small to be a valid PDF
            os.remove(save_path)
            logging.error(f"Downloaded file too small, likely not a valid PDF: {save_path}")
            return False
            
        logging.info(f"Successfully downloaded paper to {save_path}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to download paper: {str(e)}")
        # Clean up partial downloads if they exist
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def extract_title_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract the actual title from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        The extracted title or None if extraction failed
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # First try to extract from metadata
            if reader.metadata and reader.metadata.get('/Title'):
                return reader.metadata.get('/Title')
            
            # If metadata doesn't have title, try to extract from first page
            first_page_text = reader.pages[0].extract_text()
            
            # Look for title patterns (usually at the top of the first page)
            # Most academic papers have title as the first substantial line
            lines = first_page_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            # Simple heuristic: First substantial line that's not a header/footer
            for line in non_empty_lines[:5]:  # Check first 5 non-empty lines
                # Skip header/footer lines (usually short or contain page numbers)
                if len(line) > 10 and not re.search(r'page \d+|\d+/\d+|©|copyright', line.lower()):
                    return line
            
            # If still not found, look for the largest font text on first page
            # (This would require PyMuPDF which is more complex)
            
            # Fall back to first 2-3 lines combined if nothing else works
            if len(non_empty_lines) >= 2:
                return ' '.join(non_empty_lines[:2])
            
            return None
                
    except Exception as e:
        logging.error(f"Error extracting title from PDF {pdf_path}: {str(e)}")
        return None

def verify_downloaded_papers(download_dir: str, expected_titles: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Verify that downloaded papers match their expected titles.
    
    Args:
        download_dir: Directory containing downloaded papers
        expected_titles: Dictionary mapping file paths to expected titles
        
    Returns:
        Tuple of (verified_papers, deleted_papers)
    """
    verified_papers = []
    deleted_papers = []
    
    for filepath, expected_title in expected_titles.items():
        if not os.path.exists(filepath):
            logging.warning(f"Paper file not found: {filepath}")
            continue
            
        logging.info(f"Verifying paper: {os.path.basename(filepath)}")
        
        # Extract actual title from PDF
        actual_title = extract_title_from_pdf(filepath)
        
        if not actual_title:
            logging.warning(f"Could not extract title from {filepath}, keeping file")
            verified_papers.append(filepath)
            continue
            
        # Compare titles
        similarity = title_similarity(expected_title, actual_title)
        
        # Set a threshold for acceptable similarity
        if similarity >= 0.5:  # 50% similarity should be enough for title verification
            logging.info(f"Title verification passed (similarity: {similarity:.2f}): {filepath}")
            verified_papers.append(filepath)
        else:
            logging.warning(f"Title mismatch detected (similarity: {similarity:.2f}):")
            logging.warning(f"  Expected: {expected_title}")
            logging.warning(f"  Actual: {actual_title}")
            logging.warning(f"Deleting mismatched paper: {filepath}")
            
            try:
                os.remove(filepath)
                deleted_papers.append(filepath)
                logging.info(f"Deleted mismatched paper: {filepath}")
            except Exception as e:
                logging.error(f"Error deleting file {filepath}: {str(e)}")
    
    return verified_papers, deleted_papers

def download_cvpr_papers(keywords: List[str], cvpr_url: str = "https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers", proxy=None, download_dir: str = 'downloaded_papers'):
    """
    Main function to download CVPR papers related to specified keywords.
    
    Args:
        keywords: List of keywords to filter papers
        cvpr_url: URL of the CVPR accepted papers page
        proxy: Optional proxy server to use for requests
        download_dir: Directory to save downloaded papers
    """
    download_dir = setup_download_directory(download_dir)
    log_file = os.path.join(download_dir, 'download_log.txt')
    
    # Create a log file
    with open(log_file, 'w') as f:
        f.write(f"Download started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n\n")
    
    # Check for lxml dependency and warn if not installed
    try:
        from bs4 import BeautifulSoup
        BeautifulSoup("<test/>", "xml")
    except Exception:
        logging.warning("XML parser not found. For better performance, install lxml: pip install lxml")
    
    # Create a session with retry logic for the initial page fetch
    session = create_robust_session(proxy=proxy)
    
    # Scrape paper titles from CVPR
    try:
        logging.info(f"Fetching papers from {cvpr_url}")
        print(f"Fetching papers from CVPR website...")
        response = session.get(cvpr_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = []
        authors = []
        
        # Try multiple potential HTML structures (reusing previous logic)
        paper_rows = soup.find_all('tr')
        
        if not paper_rows:
            # Alternative structure: might be in divs or other containers
            paper_containers = soup.find_all('div', class_=['paper', 'paper-container', 'publication'])
            
            if paper_containers:
                for container in paper_containers:
                    title_elem = container.find(['h2', 'h3', 'strong', 'div'], class_=['title', 'paper-title'])
                    authors_elem = container.find(['div', 'p', 'span'], class_=['authors', 'paper-authors'])
                    
                    if title_elem:
                        titles.append(title_elem.text.strip())
                        if authors_elem:
                            authors.append(authors_elem.text.strip())
                        else:
                            authors.append("")
        else:
            # Original structure (table rows)
            for row in paper_rows:
                title_cell = row.find('td')
                if title_cell and title_cell.find('strong'):
                    title_text = title_cell.find('strong').text.strip()
                    titles.append(title_text)
                    
                    # Extract authors if available
                    author_div = title_cell.find('div', class_='indented')
                    if author_div and author_div.find('i'):
                        authors.append(author_div.find('i').text.strip())
                    else:
                        authors.append("")
        
        logging.info(f"Found {len(titles)} papers")
        print(f"Found {len(titles)} papers on the CVPR website")
        
    except Exception as e:
        logging.error(f"Error fetching papers: {str(e)}")
        print(f"Error fetching papers: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"Error fetching papers: {str(e)}\n")
        return
    
    if not titles:
        logging.error("No papers found. Please check if the webpage structure has changed.")
        print("No papers found. Please check if the webpage structure has changed.")
        with open(log_file, 'a') as f:
            f.write("No papers found. Please check if the webpage structure has changed.\n")
        return
    
    papers_with_keywords = 0
    downloaded_papers = 0
    paper_info = []  # To collect information about each paper
    expected_titles = {}  # Map file paths to expected titles
    
    # Print header for progress display
    print(f"\n{'='*80}")
    print(f"Searching for papers with keywords: {', '.join(keywords)}")
    print(f"{'='*80}")
    
    # Iterate over each title and search for the paper on arXiv
    for i, title in enumerate(titles):
        author = authors[i] if i < len(authors) else ""
        
        if has_keyword_in_title(title, keywords):
            papers_with_keywords += 1
            logging.info(f"\nFound paper with keyword: {title}")
            print(f"\n[{papers_with_keywords}] Found paper with keyword:")
            print(f"Title: {title}")
            print(f"Authors: {author}")
            
            paper_data = {
                "title": title,
                "authors": author,
                "status": "Not found on arXiv"
            }
            
            with open(log_file, 'a') as f:
                f.write(f"\n{title}\n")
                f.write(f"Authors: {author}\n")
            
            # Use our improved search function with retry logic
            print(f"Searching on arXiv... ", end="", flush=True)
            pdf_url, arxiv_title, arxiv_id = search_arxiv_paper(title, author, proxy=proxy)
            
            if pdf_url:
                print(f"Found! (arXiv ID: {arxiv_id})")
                # Define a path to save the downloaded paper
                filename = sanitize_filename(title, arxiv_id)
                save_path = os.path.join(download_dir, filename)
                
                # Use our improved download function
                print(f"Downloading paper... ", end="", flush=True)
                success = download_paper(pdf_url, save_path, title, proxy=proxy)
                if success:
                    downloaded_papers += 1
                    paper_data["status"] = "Downloaded"
                    paper_data["arxiv_title"] = arxiv_title
                    paper_data["arxiv_id"] = arxiv_id
                    paper_data["file_path"] = save_path
                    
                    # Store the expected title for verification
                    expected_titles[save_path] = arxiv_title or title
                    
                    print(f"Success!")
                    with open(log_file, 'a') as f:
                        f.write(f"Downloaded: {save_path}\n")
                        f.write(f"arXiv Title: {arxiv_title}\n")
                        f.write(f"arXiv ID: {arxiv_id}\n")
                else:
                    paper_data["status"] = "Download failed"
                    print(f"Failed!")
                    with open(log_file, 'a') as f:
                        f.write(f"Failed to download\n")
            else:
                print(f"Not found on arXiv")
                with open(log_file, 'a') as f:
                    f.write(f"Not found on arXiv\n")
            
            paper_info.append(paper_data)
            
            # Print progress information
            print(f"Progress: {papers_with_keywords} papers found, {downloaded_papers} downloaded")
            
            # More intelligent delay between requests to avoid rate limiting
            delay = random.uniform(3.0, 7.0)  # Randomized delay between 3-7 seconds
            print(f"Waiting {delay:.1f}s before next request...")
            time.sleep(delay)
    
    # Verify downloaded papers
    if downloaded_papers > 0:
        logging.info("\nVerifying downloaded papers...")
        print(f"\n{'='*80}")
        print(f"Verifying {downloaded_papers} downloaded papers...")
        verified_papers, deleted_papers = verify_downloaded_papers(download_dir, expected_titles)
        
        with open(log_file, 'a') as f:
            f.write("\n\nVerification Results:\n")
            f.write(f"Verified papers: {len(verified_papers)}\n")
            f.write(f"Deleted mismatched papers: {len(deleted_papers)}\n")
            
            if deleted_papers:
                f.write("\nDeleted Papers:\n")
                for paper in deleted_papers:
                    f.write(f"- {os.path.basename(paper)}\n")
        
        # Update paper_info with verification status
        for paper in paper_info:
            if "file_path" in paper:
                if paper["file_path"] in deleted_papers:
                    paper["status"] = "Deleted (title mismatch)"
                elif paper["file_path"] in verified_papers:
                    paper["status"] = "Verified"
    
    # Log the summary
    logging.info(f"\nSummary: Found {papers_with_keywords} papers with keywords out of {len(titles)} total papers")
    logging.info(f"Successfully downloaded {downloaded_papers} papers")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total CVPR papers: {len(titles)}")
    print(f"Papers matching keywords: {papers_with_keywords}")
    print(f"Papers successfully downloaded: {downloaded_papers}")
    
    if downloaded_papers > 0:
        logging.info(f"Verified papers: {len(verified_papers)}")
        logging.info(f"Deleted mismatched papers: {len(deleted_papers)}")
        print(f"Verified papers: {len(verified_papers)}")
        print(f"Deleted mismatched papers: {len(deleted_papers)}")
        print(f"\nAll papers have been saved to: {os.path.abspath(download_dir)}")
    
    with open(log_file, 'a') as f:
        f.write(f"\n\nSummary: Found {papers_with_keywords} papers with keywords out of {len(titles)} total papers\n")
        f.write(f"Successfully downloaded {downloaded_papers} papers\n")
        if downloaded_papers > 0:
            f.write(f"Verified papers: {len(verified_papers)}\n")
            f.write(f"Deleted mismatched papers: {len(deleted_papers)}\n")
        f.write(f"Download completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create a detailed CSV report
    try:
        import csv
        csv_path = os.path.join(download_dir, 'paper_details.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'authors', 'status', 'arxiv_title', 'arxiv_id', 'file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for paper in paper_info:
                writer.writerow({k: paper.get(k, '') for k in fieldnames})
        logging.info(f"Detailed paper information saved to {csv_path}")
        print(f"Detailed paper information saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error creating CSV report: {str(e)}")
        print(f"Error creating CSV report: {str(e)}")
    
    print(f"\nLog file saved to: {log_file}")
    print(f"{'='*80}")

def main():
    """Main entry point with command-line argument parsing for keywords and other options."""
    # Create description with examples
    description = """
    Download CVPR papers based on keywords in the paper titles.
    
    Examples:
      python download.py --interactive
      python download.py --keywords "Diffusion" "Text-to-Image"
      python download.py -k "Vision Language Model" -d "my_papers"
    """
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--keywords', '-k', nargs='+',
                      help='Keywords to search for in paper titles (multiple keywords can be provided)')
    parser.add_argument('--url', '-u', default="https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers",
                      help='URL of the CVPR accepted papers page')
    parser.add_argument('--proxy', '-p', default=None,
                      help='Proxy server to use for requests (e.g., "http://your-proxy-server:port")')
    parser.add_argument('--dir', '-d', default='downloaded_papers',
                      help='Directory to save downloaded papers')
    parser.add_argument('--interactive', '-i', action='store_true',
                      help='Enable interactive mode to enter keywords')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle keywords input
    keywords = args.keywords
    if keywords is None or args.interactive:
        # If no keywords provided or interactive mode enabled, prompt user
        print("Enter keywords to search for in paper titles (comma-separated):")
        print("Examples: VLM, Vision Language Model, Diffusion, etc.")
        user_input = input("> ")
        if user_input.strip():
            # Split by comma and strip whitespace
            keywords = [k.strip() for k in user_input.split(',') if k.strip()]
    
    # If still no keywords, use defaults
    if not keywords:
        keywords = ["VLM", "Vision Language Model"]
        print(f"Using default keywords: {', '.join(keywords)}")
    
    # Print configuration information
    logging.info(f"Starting download with keywords: {', '.join(keywords)}")
    logging.info(f"Using CVPR URL: {args.url}")
    logging.info(f"Saving papers to: {args.dir}")
    if args.proxy:
        logging.info(f"Using proxy: {args.proxy}")
    
    download_cvpr_papers(keywords, args.url, proxy=args.proxy, download_dir=args.dir)

if __name__ == "__main__":
    main()
