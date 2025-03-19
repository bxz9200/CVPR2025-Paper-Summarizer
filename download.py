import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Function to scrape the paper titles from the CVPR webpage
def get_cvpr_paper_titles(url):
    print(f"Fetching papers from {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        titles = []
        authors = []
        
        # Updated to match the actual HTML structure (table format)
        paper_rows = soup.find_all('tr')
        
        if not paper_rows:
            print("Warning: No paper rows found. The webpage structure might have changed.")
            return [], []
            
        for row in paper_rows:
            title_cell = row.find('td')
            if title_cell and title_cell.find('strong'):
                title_text = title_cell.find('strong').text.strip()
                titles.append(title_text)
                
                # Also extract authors if available
                author_div = title_cell.find('div', class_='indented')
                if author_div and author_div.find('i'):
                    authors.append(author_div.find('i').text.strip())
                else:
                    authors.append("")
        
        print(f"Found {len(titles)} papers")
        return titles, authors
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return [], []

# Function to check if the paper title contains the keyword
def has_keyword_in_title(title, keywords):
    return any(keyword.lower() in title.lower() for keyword in keywords)

# Function to search for papers on arXiv using the title
def search_arxiv_paper(title, authors=""):
    print(f"Searching arXiv for: {title}")
    try:
        # Clean the title for search
        query = re.sub(r'[^\w\s]', ' ', title)
        query = '+'.join(query.split())
        
        # Add first author if available to improve search accuracy
        if authors:
            first_author = authors.split('&middot;')[0].strip()
            if first_author:
                query += f"+author:{first_author.replace(' ', '+')}"
        
        arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
        
        response = requests.get(arxiv_url)
        response.raise_for_status()
        
        # Parse the XML response
        soup = BeautifulSoup(response.content, 'xml')
        
        # Find the first entry in the XML response
        entry = soup.find('entry')
        if entry:
            # Extract the PDF URL from the arXiv entry
            pdf_url = entry.find('link', {'title': 'pdf'})
            if pdf_url:
                return pdf_url['href']
            else:
                print("No PDF link found in the arXiv entry")
        else:
            print("No matching paper found on arXiv")
        return None
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None

# Function to download the paper from arXiv
def download_paper(pdf_url, save_path, title):
    if pdf_url:
        print(f"Downloading paper: {title}")
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded paper to {save_path}")
            return True
        except Exception as e:
            print(f"Failed to download paper: {e}")
    else:
        print(f"No PDF URL found for the paper: {title}")
    return False

# Main function to coordinate the process
def download_cvpr_papers():
    cvpr_url = "https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers"
    keywords = ["VLM", "Vision Language Model"]
    
    # Create a directory to store downloaded papers
    if not os.path.exists('downloaded_papers'):
        os.makedirs('downloaded_papers')
    
    # Create a log file to keep track of results
    log_file = os.path.join('downloaded_papers', 'download_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Download started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n\n")
    
    # Scrape paper titles from CVPR
    titles, authors = get_cvpr_paper_titles(cvpr_url)
    
    if not titles:
        print("No papers found. Please check if the webpage structure has changed.")
        with open(log_file, 'a') as f:
            f.write("No papers found. Please check if the webpage structure has changed.\n")
        return
    
    papers_with_keywords = 0
    downloaded_papers = 0
    
    # Iterate over each title and search for the paper on arXiv
    for i, title in enumerate(titles):
        author = authors[i] if i < len(authors) else ""
        
        if has_keyword_in_title(title, keywords):
            papers_with_keywords += 1
            print(f"\nFound paper with keyword: {title}")
            
            with open(log_file, 'a') as f:
                f.write(f"\n{title}\n")
                f.write(f"Authors: {author}\n")
            
            arxiv_pdf_url = search_arxiv_paper(title, author)
            
            if arxiv_pdf_url:
                # Define a path to save the downloaded paper
                safe_title = "".join([c if c.isalnum() else "_" for c in title[:50]])  # Sanitize filename
                save_path = os.path.join('downloaded_papers', f"{safe_title}.pdf")
                
                success = download_paper(arxiv_pdf_url, save_path, title)
                if success:
                    downloaded_papers += 1
                    with open(log_file, 'a') as f:
                        f.write(f"Downloaded: {save_path}\n")
                else:
                    with open(log_file, 'a') as f:
                        f.write(f"Failed to download\n")
            else:
                print(f"Paper not found on arXiv: {title}")
                with open(log_file, 'a') as f:
                    f.write(f"Not found on arXiv\n")
            
            # Avoid overloading the server with requests
            time.sleep(3)
    
    # Log the summary
    print(f"\nSummary: Found {papers_with_keywords} papers with keywords out of {len(titles)} total papers")
    print(f"Successfully downloaded {downloaded_papers} papers")
    
    with open(log_file, 'a') as f:
        f.write(f"\n\nSummary: Found {papers_with_keywords} papers with keywords out of {len(titles)} total papers\n")
        f.write(f"Successfully downloaded {downloaded_papers} papers\n")
        f.write(f"Download completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Run the script
if __name__ == "__main__":
    download_cvpr_papers()
