# PowerShell script to start the FastAPI server and Streamlit UI concurrently.
.\.venv\Scripts\Activate.ps1
Start-Process "uvicorn" -ArgumentList "api.main:app --reload"
streamlit run ui\app.py
