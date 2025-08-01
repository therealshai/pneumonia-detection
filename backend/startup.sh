#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Change to the application root directory.
cd /home/site/wwwroot

echo "Creating and activating virtual environment..."
# Create a virtual environment named 'antenv'
python3 -m venv antenv
# Activate the virtual environment
source antenv/bin/activate

echo "Installing dependencies from requirements.txt..."
# Install packages from the requirements file, avoiding cache to ensure a fresh install
pip install --no-cache-dir -r requirements.txt

echo "Starting the Uvicorn server..."
# Run the application using the uvicorn from the activated virtual environment
# Use the absolute path to uvicorn to ensure it's always found
./antenv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
