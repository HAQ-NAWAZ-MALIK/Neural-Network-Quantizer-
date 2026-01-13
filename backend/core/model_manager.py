"""
Model Manager with Download Progress, Caching, and Auto-Cleanup
Designed to work with HuggingFace Spaces disk storage
"""

import os
import time
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

# HuggingFace cache directory - works on Spaces
HF_CACHE_DIR = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
MODEL_CACHE_DIR = Path(HF_CACHE_DIR) / "hub"

# Sample models that should always be available (tiny models for quick testing)
SAMPLE_MODELS = [
    "prajjwal1/bert-tiny",  # 4.4MB - Perfect for testing
]

# Auto-cleanup interval (4 hours)
CLEANUP_INTERVAL_HOURS = 4

# Track download progress
_download_progress: Dict[str, Dict[str, Any]] = {}
_cleanup_thread: Optional[threading.Thread] = None


@dataclass
class DownloadProgress:
    """Track download progress for a model"""
    model_name: str
    status: str  # "pending", "downloading", "extracting", "complete", "error"
    current_file: str
    files_completed: int
    total_files: int
    bytes_downloaded: int
    total_bytes: int
    speed_mbps: float
    eta_seconds: int
    error: Optional[str] = None


def get_download_progress(model_name: str) -> Optional[Dict[str, Any]]:
    """Get current download progress for a model"""
    return _download_progress.get(model_name)


def set_download_progress(model_name: str, progress: Dict[str, Any]):
    """Update download progress"""
    _download_progress[model_name] = {
        **progress,
        "timestamp": time.time()
    }


def clear_download_progress(model_name: str):
    """Clear download progress after completion"""
    if model_name in _download_progress:
        del _download_progress[model_name]


def get_cached_models() -> list:
    """Get list of models currently in cache"""
    cached = []
    
    if not MODEL_CACHE_DIR.exists():
        return cached
    
    for item in MODEL_CACHE_DIR.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Parse model name from directory name
            parts = item.name.replace("models--", "").split("--")
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"
            else:
                model_name = parts[0]
            
            # Get size
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
            
            # Get last access time
            try:
                last_access = item.stat().st_atime
            except:
                last_access = time.time()
            
            cached.append({
                "name": model_name,
                "path": str(item),
                "size_mb": round(size_mb, 2),
                "last_access": datetime.fromtimestamp(last_access).isoformat(),
                "is_sample": model_name in SAMPLE_MODELS
            })
    
    return cached


def cleanup_old_models(max_age_hours: float = CLEANUP_INTERVAL_HOURS):
    """
    Remove models that haven't been accessed in max_age_hours.
    Sample models are never deleted.
    """
    if not MODEL_CACHE_DIR.exists():
        return {"deleted": [], "kept": []}
    
    deleted = []
    kept = []
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    for item in MODEL_CACHE_DIR.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Parse model name
            parts = item.name.replace("models--", "").split("--")
            if len(parts) >= 2:
                model_name = f"{parts[0]}/{parts[1]}"
            else:
                model_name = parts[0]
            
            # Never delete sample models
            if model_name in SAMPLE_MODELS:
                kept.append(model_name)
                continue
            
            # Check last access time
            try:
                last_access = item.stat().st_atime
                if last_access < cutoff_time:
                    shutil.rmtree(item)
                    deleted.append(model_name)
                else:
                    kept.append(model_name)
            except Exception as e:
                kept.append(f"{model_name} (error: {str(e)})")
    
    return {"deleted": deleted, "kept": kept}


def delete_model_cache(model_name: str) -> bool:
    """Delete a specific model from cache"""
    if model_name in SAMPLE_MODELS:
        return False  # Don't delete sample models
    
    # Convert model name to cache directory name
    cache_name = f"models--{model_name.replace('/', '--')}"
    cache_path = MODEL_CACHE_DIR / cache_name
    
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
            return True
        except:
            return False
    return False


def ensure_sample_models():
    """
    Ensure sample models are downloaded.
    Called on startup to pre-download tiny test models.
    """
    try:
        from transformers import AutoModel, AutoConfig
        
        for model_name in SAMPLE_MODELS:
            try:
                # Just load config first (fast)
                AutoConfig.from_pretrained(model_name)
                print(f"[ModelManager] Sample model '{model_name}' is available")
            except Exception as e:
                print(f"[ModelManager] Sample model '{model_name}' not cached: {e}")
    except ImportError:
        print("[ModelManager] transformers not installed, skipping sample model check")


def start_cleanup_scheduler():
    """Start background thread for periodic cleanup"""
    global _cleanup_thread
    
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    
    def cleanup_loop():
        while True:
            time.sleep(CLEANUP_INTERVAL_HOURS * 3600)  # Wait 4 hours
            try:
                result = cleanup_old_models()
                if result["deleted"]:
                    print(f"[ModelManager] Cleaned up models: {result['deleted']}")
            except Exception as e:
                print(f"[ModelManager] Cleanup error: {e}")
    
    _cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    _cleanup_thread.start()
    print(f"[ModelManager] Cleanup scheduler started (every {CLEANUP_INTERVAL_HOURS} hours)")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    models = get_cached_models()
    total_size = sum(m["size_mb"] for m in models)
    sample_count = sum(1 for m in models if m["is_sample"])
    
    return {
        "cache_dir": str(MODEL_CACHE_DIR),
        "total_models": len(models),
        "sample_models": sample_count,
        "total_size_mb": round(total_size, 2),
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "models": models
    }


# Progress callback for HuggingFace downloads
class DownloadProgressCallback:
    """Callback to track HuggingFace download progress"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start_time = time.time()
        self.last_update = 0
    
    def __call__(self, current: int, total: int, filename: str = ""):
        now = time.time()
        
        # Throttle updates to every 0.5 seconds
        if now - self.last_update < 0.5:
            return
        
        self.last_update = now
        elapsed = now - self.start_time
        speed = current / elapsed if elapsed > 0 else 0
        eta = int((total - current) / speed) if speed > 0 else 0
        
        set_download_progress(self.model_name, {
            "status": "downloading",
            "current_file": filename,
            "bytes_downloaded": current,
            "total_bytes": total,
            "percent": round(100 * current / total, 1) if total > 0 else 0,
            "speed_mbps": round(speed / (1024 * 1024), 2),
            "eta_seconds": eta
        })
