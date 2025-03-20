#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Content Recommendation Engine..."
python3 -m streamlit run app.py