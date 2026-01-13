"""
HuggingFace Spaces Entry Point
This file serves as the entry point for HuggingFace Spaces deployment.
It starts the FastAPI application which serves both the API and the React frontend.
"""

import uvicorn
from backend.api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        log_level="info"
    )