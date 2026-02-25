#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting DermAI Engine Pipeline..."
# Start un-cached API server and Dashboard
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
