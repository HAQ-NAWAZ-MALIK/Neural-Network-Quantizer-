"""
FastAPI Main Application
Neural Network Weight Quantizer API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os

from .routes import quantization, models, analysis, system

# Create FastAPI app
app = FastAPI(
    title="Neural Network Quantizer API",
    description="API for quantizing neural network weights to lower precision formats",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(quantization.router, prefix="/api/quantize", tags=["Quantization"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "quantizer-api"}

# Serve frontend in production
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Serve index.html for SPA routing
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
