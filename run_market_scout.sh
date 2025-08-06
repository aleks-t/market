#!/bin/bash

# Activate virtual environment if it exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run market-app.py