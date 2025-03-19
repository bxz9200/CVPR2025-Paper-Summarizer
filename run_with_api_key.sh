#!/bin/bash

# Replace YOUR_API_KEY with your actual HuggingFace API key
export HUGGINGFACE_API_KEY="YOUR_API_KEY"

# Run the summarization script with HTML output
python summarize.py --max-papers 3 --html

# Note: You can get a HuggingFace API key by:
# 1. Creating an account at https://huggingface.co
# 2. Going to your profile -> Settings -> Access Tokens
# 3. Creating a new token with read permissions 