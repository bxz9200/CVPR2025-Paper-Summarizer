# Paper Abstract Summarizer and Downloader

This repository contains advanced tools for downloading and summarizing academic papers from CVPR 2025 conference.

## Components

### 1. Paper Downloader (`download.py`)

Automatically downloads CVPR papers related to Vision Language Models from arXiv.

- Scrapes the CVPR 2025 accepted papers list
- Filters papers using keywords (currently "VLM", "Vision Language Model", "Multimodal", etc.)
- Searches for matching papers on arXiv
- Downloads the PDFs to the `downloaded_papers` folder
- Verifies downloaded papers match expected titles

### 2. Abstract Summarizer (`summarize.py`)

This script extracts abstracts from PDF papers, validates them, extracts proposed methods, generates keywords, and creates well-structured summaries using multiple approaches.

### 3. Web Server (`summary_server.py`)

A Flask-based web server that provides a user interface for uploading papers and viewing summaries.

- Asynchronous processing of paper summarization requests
- Background worker threads for handling multiple papers
- Real-time status updates on summarization progress
- Interactive web interface for viewing results

## Features

- **Paper Download Pipeline**:
  - Automatic scraping of CVPR conference papers
  - Keyword-based filtering to find relevant papers
  - arXiv API integration for retrieving paper PDFs
  - Organized downloading with detailed logging
  - Title verification to ensure correct papers are downloaded

- **Abstract Extraction and Validation**:
  - Robust multi-stage abstract extraction from PDFs
  - Sophisticated validation to confirm extracted text is an actual abstract
  - Support for various paper formats and structures
  - Detailed error handling and logging for troubleshooting

- **Method Description Extraction**:
  - Identifies and extracts the proposed method description from papers
  - Highlights the technical contributions and approaches
  - Enhances understanding of papers' key innovations

- **Abstract Summarization**:
  - Extracts abstracts from PDF papers in a specified folder
  - Summarizes abstracts using multiple methods with automatic fallbacks:
    1. Enhanced prompt engineering with Hugging Face's API
    2. Chunk-based summarization for long abstracts
    3. SMMRY web service as a fallback
    4. Advanced extractive summarization based on academic importance
  - Post-processing to improve summary coherence and readability
  - Comprehensive academic keyword extraction

- **Output Formats**:
  - Beautiful HTML output with modern styling and responsive design
  - Clean Markdown format for integration with documentation
  - Structured sections for title, keywords, proposed method, abstract, and summary

## Demo Output

Below is an example of how the generated HTML summary looks:

<div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
  <h2>VLs I: Verbalized Layers-to-Interactions 2505.12345v1</h2>
  
  <h3>Keywords</h3>
  <div style="margin-bottom: 15px;">
    <span style="display: inline-block; background-color: #e1e8ed; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; margin-right: 8px;">vlms</span>
    <span style="display: inline-block; background-color: #e1e8ed; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; margin-right: 8px;">model</span>
    <span style="display: inline-block; background-color: #e1e8ed; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; margin-right: 8px;">vision-language</span>
    <span style="display: inline-block; background-color: #e1e8ed; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; margin-right: 8px;">efficiency</span>
    <span style="display: inline-block; background-color: #e1e8ed; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; margin-right: 8px;">distillation</span>
  </div>
  
  <h3>Proposed Method</h3>
  <div style="background-color: #ebf5eb; padding: 15px; border-radius: 8px; border-left: 3px solid #28a745; margin-bottom: 20px;">
    <p>The paper proposes VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones.</p>
  </div>
  
  <h3>Original Abstract</h3>
  <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic;">
    <p>The recent surge in high-quality visual instruction tuning samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger models brings significant computational challenges, especially for deployment on resource-constrained devices like mobile platforms and robots. To address this, we propose VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes efficiency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones. We validate VLsI across ten challenging vision-language benchmarks, achieving notable performance gains (11.0% for 2B and 17.4% for 7B) over GPT-4V without the need for model scaling, merging, or architectural changes.</p>
  </div>
  
  <h3>Summary</h3>
  <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 3px solid #0078a0;">
    <p>This paper introduces VLsI (Verbalized Layers-to-Interactions), a new family of efficient vision-language models (VLMs) in 2B and 7B sizes designed for resource-constrained devices. The key innovation is a layer-wise distillation process that uses intermediate "verbalizers" to map features from each layer to natural language space, allowing smaller models to align with larger models' reasoning processes. This approach goes beyond typical final-layer tuning by aligning the progression of information through all model layers. Testing across ten vision-language benchmarks showed significant performance gains (11.0% for 2B and 17.4% for 7B models) over GPT-4V without requiring additional model scaling or architectural changes.</p>
  </div>
</div>

## Prerequisites

- Python 3.6+
- Required packages (see `requirements.txt`)
- Optional: HuggingFace API key for improved summarization quality

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
- `--max-papers`: Maximum number of papers to process (default: all)
- `--hf-token`: HuggingFace API token (optional, limited requests possible without it)

Example with all options:

```bash
python summarize.py --input-dir="my_papers" --output="summaries.md" --html --max-papers=10
```

### Step 3: Using the API key (optional but recommended)

For better summarization quality, you can use the provided shell script with your HuggingFace API key:

```bash
# First, edit the script to add your API key
nano run_with_api_key.sh

# Then run the script
./run_with_api_key.sh
```

### Step 4: Running the web server (optional)

For an interactive experience, you can run the web server:

```bash
# Set your API key (optional)
export HUGGINGFACE_API_KEY="your_api_key_here"

# Start the server
python summary_server.py

# Access the web interface at http://localhost:5000
```

## How it works

### Abstract extraction and validation process

1. The system reads the first few pages of each PDF
2. It uses multiple strategies to locate and extract the abstract:
   - Pattern-based detection for "Abstract" sections
   - Structural analysis of academic paper layouts
   - Paragraph analysis for abstracts without explicit labels
3. The extracted text undergoes validation to ensure it's an actual abstract
4. Validation includes checks for:
   - Common abstract starting phrases
   - Appropriate length and structure
   - Absence of section headers and other non-abstract content

### Method description extraction

1. The system analyzes the abstract to identify sentences describing the proposed method
2. It looks for key phrases like "we propose", "our approach", etc.
3. The extracted method description highlights the technical contribution of the paper

### Summarization process

1. The abstract is preprocessed to fix formatting issues and improve readability
2. For shorter abstracts, the entire text is summarized using enhanced prompts
3. For longer abstracts, the text is split into semantic chunks that are:
   - Individually summarized with contextual markers
   - Combined with transition phrases for coherence
4. If the primary method fails, the system falls back to:
   - SMMRY web service
   - Advanced extractive summarization that prioritizes sentences based on:
     - Position in the abstract
     - Content relevance to academic importance
     - Information density
     - Logical flow
5. The summary undergoes post-processing to:
   - Remove repetitive content
   - Fix awkward transitions
   - Ensure proper formatting
   - Improve overall readability

## Output

The summarization output includes:
- The paper title
- Extracted keywords
- Proposed method description
- Original abstract
- Concise AI-generated summary

The HTML output features modern styling with responsive design for better readability.

## Recent Improvements

For a detailed list of recent enhancements, please see the [IMPROVEMENTS.md](IMPROVEMENTS.md) file.
