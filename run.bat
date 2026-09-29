@echo off
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
