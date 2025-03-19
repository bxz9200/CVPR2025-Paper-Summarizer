#!/usr/bin/env python3
import os
import sys
import json
import logging
import threading
import queue
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from summarize import extract_and_validate_abstract, summarize_abstract_with_huggingface, post_process_summary, keywords_from_abstract, preprocess_abstract, extract_method_description

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='summary_server.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

app = Flask(__name__, static_folder='static')

# Queue for paper summarization requests
summary_queue = queue.Queue()
# Results storage
results = {}
# Lock for thread-safe operations
results_lock = threading.Lock()

# Get HuggingFace API key from environment
api_key = os.environ.get("HUGGINGFACE_API_KEY")

def worker():
    """Background worker that processes papers from the queue."""
    while True:
        try:
            # Get the next paper from the queue
            pdf_path, request_id = summary_queue.get()
            
            if pdf_path == "SHUTDOWN":
                summary_queue.task_done()
                break
                
            logging.info(f"Processing [{request_id}]: {os.path.basename(pdf_path)}")
            
            # Extract abstract with validation
            abstract = extract_and_validate_abstract(pdf_path)
            
            if not abstract:
                with results_lock:
                    results[request_id] = {
                        "status": "error",
                        "message": "Failed to extract abstract",
                        "pdf_path": pdf_path
                    }
                logging.warning(f"Could not extract abstract from {pdf_path}")
                summary_queue.task_done()
                continue
                
            # Extract title from filename
            title = os.path.splitext(os.path.basename(pdf_path))[0]
            title = title.replace('_', ' ')
            
            # Get keywords
            keywords = keywords_from_abstract(abstract)
            
            # Extract method description
            method_description = extract_method_description(abstract)
            if method_description:
                logging.info(f"Extracted method description for [{request_id}]")
            
            # Generate summary
            summary = summarize_abstract_with_huggingface(abstract, api_key, title)
            
            # Apply post-processing
            if summary:
                summary = post_process_summary(summary)
                
            with results_lock:
                results[request_id] = {
                    "status": "completed",
                    "title": title,
                    "abstract": abstract,
                    "summary": summary,
                    "keywords": keywords,
                    "method_description": method_description,
                    "pdf_path": pdf_path
                }
                
            logging.info(f"Completed [{request_id}]: {title}")
            
        except Exception as e:
            logging.error(f"Error processing paper: {str(e)}")
            with results_lock:
                if request_id in results:
                    results[request_id]["status"] = "error"
                    results[request_id]["message"] = str(e)
                else:
                    results[request_id] = {
                        "status": "error",
                        "message": str(e),
                        "pdf_path": pdf_path
                    }
        finally:
            summary_queue.task_done()

# Start worker threads
num_worker_threads = 3
worker_threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    worker_threads.append(t)
    
@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_paper():
    """Submit a paper for summarization."""
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
        
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Uploaded file must be a PDF"}), 400
    
    # Save the uploaded file
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create unique filename to avoid conflicts
    filename = pdf_file.filename
    file_path = os.path.join(upload_dir, filename)
    pdf_file.save(file_path)
    
    # Generate a unique request ID
    request_id = f"req_{int(time.time())}_{os.path.basename(filename)}"
    
    # Initialize result entry
    with results_lock:
        results[request_id] = {
            "status": "pending",
            "pdf_path": file_path
        }
    
    # Queue the paper for processing
    summary_queue.put((file_path, request_id))
    
    return jsonify({
        "request_id": request_id,
        "status": "pending",
        "message": "Paper submitted for summarization"
    })

@app.route('/status/<request_id>')
def get_status(request_id):
    """Get the status of a summarization request."""
    with results_lock:
        if request_id not in results:
            return jsonify({"error": "Request ID not found"}), 404
        return jsonify(results[request_id])

@app.route('/results')
def get_all_results():
    """Get all summarization results."""
    with results_lock:
        return jsonify(results)

@app.route('/templates/<path:path>')
def send_template(path):
    """Serve template files."""
    return send_from_directory('templates', path)

if __name__ == "__main__":
    # Create directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Create a basic index.html if it doesn't exist
    index_path = os.path.join('templates', 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Paper Summarizer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-section {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
        }
        .results-section {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
        }
        .summary-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: white;
        }
        .summary-card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .keyword-tag {
            display: inline-block;
            background-color: #e1e8ed;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .loading {
            display: none;
            margin-top: 20px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background-color: #ddd;
            border: none;
            border-radius: 4px 4px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <h1>Academic Paper Summarizer</h1>
    <p>Upload academic papers to generate summaries.</p>
    
    <div class="tabs">
        <button class="tab active" onclick="openTab(event, 'uploadTab')">Upload Paper</button>
        <button class="tab" onclick="openTab(event, 'resultsTab')">View Results</button>
    </div>
    
    <div id="uploadTab" class="tab-content active">
        <div class="upload-section">
            <h2>Upload a Paper</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="pdfFile" name="pdf_file" accept=".pdf" required>
                <button type="submit" class="submit-btn">Submit for Summarization</button>
            </form>
            <div id="loading" class="loading">
                <p>Processing your paper... This may take a few moments.</p>
            </div>
            <div id="uploadResult"></div>
        </div>
    </div>
    
    <div id="resultsTab" class="tab-content">
        <div class="results-section">
            <h2>Summarization Results</h2>
            <button onclick="refreshResults()" class="submit-btn">Refresh Results</button>
            <div id="summaryResults"></div>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
            
            if (tabName === "resultsTab") {
                refreshResults();
            }
        }
        
        document.getElementById("uploadForm").addEventListener("submit", function(e) {
            e.preventDefault();
            
            var fileInput = document.getElementById("pdfFile");
            if (fileInput.files.length === 0) {
                alert("Please select a PDF file to upload.");
                return;
            }
            
            var formData = new FormData();
            formData.append("pdf_file", fileInput.files[0]);
            
            var loading = document.getElementById("loading");
            var uploadResult = document.getElementById("uploadResult");
            
            loading.style.display = "block";
            uploadResult.innerHTML = "";
            
            fetch("/submit", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = "none";
                if (data.error) {
                    uploadResult.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                } else {
                    uploadResult.innerHTML = `
                        <p style="color: green;">Paper submitted successfully!</p>
                        <p>Request ID: ${data.request_id}</p>
                        <p>Status: ${data.status}</p>
                        <p>${data.message}</p>
                    `;
                    // Switch to results tab
                    document.querySelector(".tab[onclick*='resultsTab']").click();
                }
            })
            .catch(error => {
                loading.style.display = "none";
                uploadResult.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            });
        });
        
        function refreshResults() {
            var resultsContainer = document.getElementById("summaryResults");
            resultsContainer.innerHTML = "<p>Loading results...</p>";
            
            fetch("/results")
            .then(response => response.json())
            .then(data => {
                resultsContainer.innerHTML = "";
                
                if (Object.keys(data).length === 0) {
                    resultsContainer.innerHTML = "<p>No summarization results yet.</p>";
                    return;
                }
                
                for (let requestId in data) {
                    let result = data[requestId];
                    let card = document.createElement("div");
                    card.className = "summary-card";
                    
                    if (result.status === "pending") {
                        card.innerHTML = `
                            <h3>${result.pdf_path.split('/').pop()}</h3>
                            <p>Status: <strong>Pending</strong></p>
                            <p>Your paper is in the processing queue.</p>
                        `;
                    } else if (result.status === "error") {
                        card.innerHTML = `
                            <h3>${result.pdf_path.split('/').pop()}</h3>
                            <p>Status: <strong style="color: red;">Error</strong></p>
                            <p>Error message: ${result.message}</p>
                        `;
                    } else if (result.status === "completed") {
                        let keywordsHtml = '';
                        if (result.keywords && result.keywords.length > 0) {
                            keywordsHtml = '<div style="margin-top: 10px; margin-bottom: 15px;">';
                            result.keywords.forEach(keyword => {
                                keywordsHtml += `<span class="keyword-tag">${keyword}</span>`;
                            });
                            keywordsHtml += '</div>';
                        }
                        
                        // Method description HTML
                        let methodHtml = '';
                        if (result.method_description) {
                            methodHtml = `
                                <div style="margin-top: 15px; margin-bottom: 15px;">
                                    <h4>Proposed Method</h4>
                                    <div style="background-color: #ebf5eb; padding: 15px; border-radius: 8px; border-left: 3px solid #28a745;">
                                        ${result.method_description.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #28a745;">$1</strong>')}
                                    </div>
                                </div>
                            `;
                        }
                        
                        card.innerHTML = `
                            <h3>${result.title}</h3>
                            <p>Status: <strong style="color: green;">Completed</strong></p>
                            ${keywordsHtml}
                            ${methodHtml}
                            <h4>Summary</h4>
                            <p>${result.summary}</p>
                            <div style="margin-top: 15px;">
                                <details>
                                    <summary>View Original Abstract</summary>
                                    <p style="font-style: italic; margin-top: 10px;">${result.abstract}</p>
                                </details>
                            </div>
                        `;
                    }
                    
                    resultsContainer.appendChild(card);
                }
            })
            .catch(error => {
                resultsContainer.innerHTML = `<p style="color: red;">Error loading results: ${error.message}</p>`;
            });
        }
        
        // Refresh results every 10 seconds if on results tab
        setInterval(() => {
            if (document.getElementById("resultsTab").classList.contains("active")) {
                refreshResults();
            }
        }, 10000);
    </script>
</body>
</html>
            """)
    
    try:
        # Start the Flask app
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        # Shutdown worker threads
        for _ in range(num_worker_threads):
            summary_queue.put(("SHUTDOWN", None))
        
        # Wait for all worker threads to complete
        for t in worker_threads:
            t.join()
            
        logging.info("Server shutdown complete")
        sys.exit(0) 