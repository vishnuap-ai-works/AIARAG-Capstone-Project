#!/bin/bash
# Shell script to start the FastAPI server and Streamlit UI concurrently.
source .venv/bin/activate
uvicorn api.main:app --reload &
streamlit run ui/app.py
