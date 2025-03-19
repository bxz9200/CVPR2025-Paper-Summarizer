# Paper Abstract Summarizer and Downloader

This repository contains tools for downloading and summarizing academic papers from CVPR 2025 conference.

## Components

### 1. Paper Downloader (`download.py`)

Automatically downloads CVPR papers related to Vision Language Models from arXiv.

- Scrapes the CVPR 2025 accepted papers list
- Filters papers using keywords (currently "VLM", "Vision Language Model")
- Searches for matching papers on arXiv
- Downloads the PDFs to the `downloaded_papers` folder

### 2. Abstract Summarizer (`summarize.py`)

This script extracts abstracts from PDF papers in the `downloaded_papers` folder, summarizes them using free methods, and saves the results in markdown or HTML format.

## Features

- **Paper Download Pipeline**:
  - Automatic scraping of CVPR conference papers
  - Keyword-based filtering to find relevant papers
  - arXiv API integration for retrieving paper PDFs
  - Organized downloading with logging

- **Abstract Summarization**:
  - Extracts abstracts from PDF papers in a specified folder
  - Summarizes abstracts using multiple fallback methods:
    1. Hugging Face's free inference API (no API key required, but one can be provided)
    2. SMMRY web service as a fallback
    3. Local NLTK-based extraction summarization as a final fallback
  - Saves summaries in either markdown or HTML format
  - Handles multiple papers in batch with logging

## Prerequisites

- Python 3.6+
- No paid API key required!

## Installation

1. Clone this repository or download the files
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download papers (optional)

If you don't already have papers in the `downloaded_papers` folder, you can run:

```bash
python download.py
```

This will:
- Scrape the CVPR 2025 website for accepted papers
- Filter them based on VLM-related keywords
- Search for these papers on arXiv
- Download them to the `downloaded_papers` folder
- Create a log file with download results

### Step 2: Summarize papers

Once you have papers in the `downloaded_papers` folder (either downloaded with `download.py` or added manually), run:

```bash
python summarize.py
```

#### Command-line options for summarize.py:

- `--input-dir`: Directory containing PDF files (default: "downloaded_papers")
- `--output`: Output file path (default: "paper_summaries.md")
- `--html`: Generate HTML output instead of markdown
- `--hf-token`: HuggingFace API token (optional, limited requests possible without it)

Example with all options:

```bash
python summarize.py --input-dir="my_papers" --output="summaries.md" --html --hf-token="your-huggingface-token"
```

## How it works

### Paper download process

1. The script scrapes the CVPR 2025 website for accepted papers
2. It filters papers containing keywords related to Vision Language Models
3. For each matching paper, it searches arXiv using the paper title
4. When a match is found, it downloads the PDF to the `downloaded_papers` folder
5. A detailed log is created with information about each paper

### Abstract summarization process

1. The script scans the input directory for PDF files
2. For each PDF, it extracts the title from the filename and the abstract from the content
3. The abstract is summarized using:
   - First attempt: HuggingFace's free inference API
   - Second attempt: SMMRY web service
   - Final fallback: Local NLTK-based extractive summarization
4. All summaries are compiled into a single markdown or HTML file with proper formatting

## Output

The summarization output is a markdown or HTML file with sections for each paper, including:
- The paper title
- The original abstract
- A concise AI-generated summary

The HTML output includes styling for better readability.
