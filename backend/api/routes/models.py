"""
Model Routes with Download Progress Streaming
Supports HuggingFace Spaces with proper cache management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch
import asyncio
import json
import traceback
import time
from backend.core.model_loader import model_loader

from backend.core.model_manager import (
    get_download_progress, set_download_progress, clear_download_progress,
    get_cached_models, cleanup_old_models, delete_model_cache,
    get_cache_stats, ensure_sample_models, start_cleanup_scheduler,
    SAMPLE_MODELS
)

router = APIRouter()


class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_name: str
    dtype: str = "auto"
    device: str = "auto"
    trust_remote_code: bool = True


class DeleteModelRequest(BaseModel):
    """Request to delete a cached model"""
    model_name: str


# In-memory state
_loaded_model = None
_loaded_tokenizer = None
_model_name = None

# Start cleanup scheduler on module load
start_cleanup_scheduler()


def _get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_torch_dtype(dtype_str: str, device: str):
    """Convert dtype string to torch dtype"""
    if dtype_str == "auto":
        if device == "cuda":
            return torch.float16
        return torch.float32
    
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


async def _load_model_with_progress(model_name: str, dtype: str, device: str, trust_remote_code: bool):
    """Load model and yield progress updates"""
    global _loaded_model, _loaded_tokenizer, _model_name
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
    except ImportError:
        yield {"type": "error", "error": "transformers library not installed"}
        return
    
    try:
        # Phase 1: Fetching config
        yield {"type": "progress", "phase": "config", "percent": 5, "message": "Fetching model configuration..."}
        
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except Exception as e:
            yield {"type": "error", "error": f"Model not found: {str(e)}", "suggestion": "Check the model ID is correct"}
            return
        
        # Phase 2: Determine device and dtype
        actual_device = device if device != "auto" else _get_device()
        torch_dtype = _get_torch_dtype(dtype, actual_device)
        
        yield {"type": "progress", "phase": "download", "percent": 10, "message": f"Downloading model to {actual_device}..."}
        
        # Set download progress for polling
        set_download_progress(model_name, {
            "status": "downloading",
            "percent": 10,
            "message": "Downloading model files..."
        })
        
        # Phase 3: Download and load model
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True
            )
            yield {"type": "progress", "phase": "download", "percent": 70, "message": "Model downloaded successfully"}
        except Exception as e:
            # Try without low_cpu_mem_usage
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code
                )
                yield {"type": "progress", "phase": "download", "percent": 70, "message": "Model downloaded (fallback mode)"}
            except Exception as e2:
                yield {"type": "error", "error": f"Failed to load model: {str(e2)}"}
                clear_download_progress(model_name)
                return
        
        # Phase 4: Move to device
        yield {"type": "progress", "phase": "device", "percent": 80, "message": f"Moving model to {actual_device}..."}
        
        if actual_device != "cpu" and not hasattr(model, 'hf_device_map'):
            try:
                model = model.to(actual_device)
            except Exception:
                actual_device = "cpu"
                model = model.to("cpu")
        
        model.eval()
        
        # Phase 5: Load tokenizer
        yield {"type": "progress", "phase": "tokenizer", "percent": 90, "message": "Loading tokenizer..."}
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except Exception:
            tokenizer = None
        
        # Store in memory
        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _model_name = model_name
        
        # Sync with global model loader
        if model_loader:
            model_loader.register_model(model, model_name, tokenizer)
        
        # Compute model info
        num_params = sum(p.numel() for p in model.parameters())
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        quantizable_layers = []
        for name, module in model.named_modules():
            if any(t in module.__class__.__name__ for t in ["Linear", "Conv1d", "Conv2d"]):
                quantizable_layers.append(name)
        
        # Phase 6: Complete
        clear_download_progress(model_name)
        
        yield {
            "type": "complete",
            "percent": 100,
            "model_info": {
                "name": model_name,
                "architecture": model.config.architectures[0] if hasattr(model.config, 'architectures') and model.config.architectures else "Unknown",
                "num_params": num_params,
                "num_params_millions": round(num_params / 1e6, 2),
                "memory_mb": round(memory_mb, 2),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "num_quantizable_layers": len(quantizable_layers),
                "has_tokenizer": tokenizer is not None,
                "is_sample": model_name in SAMPLE_MODELS
            }
        }
        
    except Exception as e:
        clear_download_progress(model_name)
        yield {"type": "error", "error": str(e), "traceback": traceback.format_exc()}


@router.post("/load")
async def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """Load a model (non-streaming version for simple requests)"""
    result = None
    async for update in _load_model_with_progress(
        request.model_name, request.dtype, request.device, request.trust_remote_code
    ):
        result = update
    
    if result and result.get("type") == "complete":
        return {"success": True, "model_info": result["model_info"]}
    elif result and result.get("type") == "error":
        return {"success": False, "error": result.get("error"), "suggestion": result.get("suggestion")}
    else:
        return {"success": False, "error": "Unknown error"}


@router.post("/load/stream")
async def load_model_stream(request: LoadModelRequest):
    """Load a model with Server-Sent Events for progress updates"""
    
    async def event_generator():
        async for update in _load_model_with_progress(
            request.model_name, request.dtype, request.device, request.trust_remote_code
        ):
            yield f"data: {json.dumps(update)}\n\n"
            await asyncio.sleep(0.1)  # Small delay between events
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/progress/{model_name}")
async def get_model_progress(model_name: str) -> Dict[str, Any]:
    """Get download progress for a model (polling endpoint)"""
    progress = get_download_progress(model_name)
    if progress:
        return {"downloading": True, **progress}
    return {"downloading": False}


@router.get("/status")
async def get_loading_status() -> Dict[str, Any]:
    """Get current model loading status"""
    return {
        "model_loaded": _loaded_model is not None,
        "model_name": _model_name,
        "has_tokenizer": _loaded_tokenizer is not None
    }


@router.get("/info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the currently loaded model"""
    if _loaded_model is None:
        return {"loaded": False, "message": "No model loaded"}
    
    num_params = sum(p.numel() for p in _loaded_model.parameters())
    memory_mb = sum(p.numel() * p.element_size() for p in _loaded_model.parameters()) / (1024 * 1024)
    
    return {
        "loaded": True,
        "name": _model_name,
        "num_params": num_params,
        "num_params_millions": round(num_params / 1e6, 2),
        "memory_mb": round(memory_mb, 2),
        "device": str(next(_loaded_model.parameters()).device),
        "dtype": str(next(_loaded_model.parameters()).dtype)
    }


@router.get("/layers")
async def get_layers() -> Dict[str, Any]:
    """Get list of layers in the loaded model"""
    if _loaded_model is None:
        return {"error": "No model loaded", "layers": []}
    
    layers = []
    quantizable_names = []
    
    for name, module in _loaded_model.named_modules():
        if not name:
            continue
        
        module_type = module.__class__.__name__
        is_quantizable = any(t in module_type for t in ["Linear", "Conv1d", "Conv2d", "Embedding"])
        
        shape = None
        num_params = 0
        if hasattr(module, 'weight') and module.weight is not None:
            shape = list(module.weight.shape)
            num_params = module.weight.numel()
        
        if num_params > 0:
            layers.append({
                "name": name,
                "type": module_type,
                "shape": shape,
                "params": num_params,
                "quantizable": is_quantizable
            })
            
            if is_quantizable:
                quantizable_names.append(name)
    
    return {
        "total_layers": len(layers),
        "quantizable_count": len(quantizable_names),
        "quantizable_layers": quantizable_names,
        "layers": layers
    }


@router.post("/unload")
async def unload_model() -> Dict[str, Any]:
    """Unload the current model and free memory"""
    global _loaded_model, _loaded_tokenizer, _model_name
    
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
    
    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None
    
    _model_name = None
    
    # Sync with global module loader
    if model_loader:
        model_loader.unload()
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"success": True, "message": "Model unloaded"}


# ============================================
# Cache Management Endpoints
# ============================================

@router.get("/cache")
async def get_cache_info() -> Dict[str, Any]:
    """Get information about cached models"""
    return get_cache_stats()


@router.post("/cache/cleanup")
async def trigger_cleanup(hours: float = 4.0) -> Dict[str, Any]:
    """Manually trigger cache cleanup"""
    result = cleanup_old_models(hours)
    return {
        "success": True,
        "deleted_count": len(result["deleted"]),
        "kept_count": len(result["kept"]),
        **result
    }


@router.delete("/cache/{model_name:path}")
async def delete_cached_model(model_name: str) -> Dict[str, Any]:
    """Delete a specific model from cache"""
    if model_name in SAMPLE_MODELS:
        return {"success": False, "error": "Cannot delete sample models"}
    
    success = delete_model_cache(model_name)
    return {"success": success, "model_name": model_name}


# ============================================
# Example Models
# ============================================

@router.get("/examples")
async def get_example_models() -> Dict[str, Any]:
    """Get list of example models for testing"""
    return {
        "sample_models": [
            {"id": model, "is_default": True, "description": "Pre-cached for quick testing"}
            for model in SAMPLE_MODELS
        ],
        "small_models": [
            {"id": "gpt2", "size": "124M", "description": "GPT-2 base model"},
            {"id": "distilbert-base-uncased", "size": "66M", "description": "DistilBERT for NLP"},
            {"id": "prajjwal1/bert-tiny", "size": "4.4M", "description": "Tiny BERT for testing"},
            {"id": "microsoft/DialoGPT-small", "size": "124M", "description": "Small conversational model"},
        ],
        "medium_models": [
            {"id": "gpt2-medium", "size": "355M", "description": "GPT-2 medium"},
            {"id": "bert-base-uncased", "size": "110M", "description": "BERT base model"},
        ],
        "cleanup_policy": f"Non-sample models are deleted after {4} hours of inactivity",
        "note": "Sample models are always available for quick testing"
    }


# Helper functions for other routes
def get_loaded_model():
    return _loaded_model


def get_layer_weights_tensor(layer_name: str):
    if _loaded_model is None:
        return None
    for name, module in _loaded_model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            return module.weight.data.clone()
    return None
