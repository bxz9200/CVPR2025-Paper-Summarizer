# Paper Abstract Summarizer

This script extracts abstracts from PDF papers in the `downloaded_papers` folder, summarizes them using free methods, and saves the results in markdown or HTML format.

## Features

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

Run the script:

```bash
python summarize.py
```

### Command-line options:

- `--input-dir`: Directory containing PDF files (default: "downloaded_papers")
- `--output`: Output file path (default: "paper_summaries.md")
- `--html`: Generate HTML output instead of markdown
- `--hf-token`: HuggingFace API token (optional, limited requests possible without it)

Example with all options:

```bash
python summarize.py --input-dir="my_papers" --output="summaries.md" --html --hf-token="your-huggingface-token"
```

## How it works

1. The script scans the input directory for PDF files
2. For each PDF, it extracts the title from the filename and the abstract from the content
3. The abstract is summarized using:
   - First attempt: HuggingFace's free inference API
   - Second attempt: SMMRY web service
   - Final fallback: Local NLTK-based extractive summarization
4. All summaries are compiled into a single markdown or HTML file with proper formatting

## Output

The output is a markdown or HTML file with sections for each paper, including:
- The paper title
- The original abstract
- A concise AI-generated summary

The HTML output includes styling for better readability.
