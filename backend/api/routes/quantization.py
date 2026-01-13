"""
Quantization Routes
Core quantization API endpoints
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch
import asyncio
import json

from backend.core.quantizer import (
    QuantizationConfig, QuantizationMethod, QuantizationMode,
    INT8Quantizer, INT4Quantizer, NF4Quantizer, get_quantizer
)
from backend.core.model_loader import model_loader
from backend.core.visualization import visualizer

router = APIRouter()


class QuantizeWeightsRequest(BaseModel):
    """Request to quantize custom weights"""
    in_features: int = 64
    out_features: int = 128
    bits: int = 8  # 4 or 8
    method: str = "int8"  # int8, int4, nf4
    mode: str = "symmetric"  # symmetric, asymmetric
    group_size: Optional[int] = None
    weight_pattern: str = "random"  # random, eye, ones, alternating, gradient
    dtype: str = "float32"


class QuantizeLayerRequest(BaseModel):
    """Request to quantize a specific layer from loaded model"""
    layer_name: str
    bits: int = 8
    method: str = "int8"
    mode: str = "symmetric"
    group_size: Optional[int] = None


class QuantizeModelRequest(BaseModel):
    """Request to quantize entire model"""
    bits: int = 8
    method: str = "int8"
    mode: str = "symmetric"
    group_size: Optional[int] = None
    layers_to_skip: List[str] = []
    layers_to_include: Optional[List[str]] = None  # None = all quantizable


def _generate_weights(pattern: str, out_features: int, in_features: int, 
                      dtype: torch.dtype) -> torch.Tensor:
    """Generate weights based on pattern"""
    if pattern == "random":
        return torch.randn((out_features, in_features), dtype=dtype)
    elif pattern == "eye":
        weights = torch.zeros((out_features, in_features), dtype=dtype)
        min_dim = min(out_features, in_features)
        weights[:min_dim, :min_dim] = torch.eye(min_dim, dtype=dtype)
        return weights
    elif pattern == "ones":
        return torch.ones((out_features, in_features), dtype=dtype)
    elif pattern == "alternating":
        weights = torch.ones((out_features, in_features), dtype=dtype)
        for i in range(out_features):
            for j in range(in_features):
                if (i + j) % 2 == 1:
                    weights[i, j] = -1.0
        return weights
    elif pattern == "gradient":
        x = torch.linspace(-1, 1, in_features)
        y = torch.linspace(-1, 1, out_features)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return (xx + yy).t().to(dtype)
    else:
        return torch.randn((out_features, in_features), dtype=dtype)


def _get_quantizer_from_config(request) -> tuple:
    """Get quantizer and config from request parameters"""
    method_map = {
        "int8": QuantizationMethod.INT8,
        "int4": QuantizationMethod.INT4,
        "nf4": QuantizationMethod.NF4
    }
    mode_map = {
        "symmetric": QuantizationMode.SYMMETRIC,
        "asymmetric": QuantizationMode.ASYMMETRIC
    }
    
    config = QuantizationConfig(
        bits=request.bits,
        method=method_map.get(request.method, QuantizationMethod.INT8),
        mode=mode_map.get(request.mode, QuantizationMode.SYMMETRIC),
        group_size=request.group_size
    )
    
    quantizer = get_quantizer(config)
    return quantizer, config


@router.post("/weights")
async def quantize_custom_weights(request: QuantizeWeightsRequest) -> Dict[str, Any]:
    """
    Quantize custom generated weights.
    This endpoint works without loading a real model.
    """
    # Map dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(request.dtype, torch.float32)
    
    # Generate weights
    weights = _generate_weights(
        request.weight_pattern, 
        request.out_features, 
        request.in_features, 
        dtype
    )
    
    # Get quantizer
    quantizer, config = _get_quantizer_from_config(request)
    
    # Quantize
    result = quantizer.quantize(weights)
    
    # Dequantize for visualization
    dequantized = quantizer.dequantize(result)
    
    # Generate visualizations
    original_heatmap = visualizer.to_dict(
        visualizer.weight_heatmap(weights, "Original Weights")
    )
    quantized_heatmap = visualizer.to_dict(
        visualizer.weight_heatmap(result.quantized_weights.float(), f"Quantized Weights ({request.bits}-bit)")
    )
    dequantized_heatmap = visualizer.to_dict(
        visualizer.weight_heatmap(dequantized, "Dequantized Weights")
    )
    error_heatmap = visualizer.to_dict(
        visualizer.weight_heatmap((weights - dequantized).abs(), "Quantization Error")
    )
    original_hist = visualizer.to_dict(
        visualizer.weight_histogram(weights, "Original Distribution")
    )
    quantized_hist = visualizer.to_dict(
        visualizer.weight_histogram(result.quantized_weights.float(), "Quantized Distribution")
    )
    scales_hist = visualizer.to_dict(
        visualizer.scales_histogram(result.scales)
    )
    
    return {
        "success": True,
        "config": config.to_dict(),
        "stats": {
            "original_shape": list(weights.shape),
            "quantized_shape": list(result.quantized_weights.shape),
            "scales_shape": list(result.scales.shape),
            "max_error": result.max_error,
            "mean_error": result.mean_error,
            "memory_savings_percent": result.memory_savings_percent,
            "original_dtype": str(weights.dtype),
            "quantized_dtype": str(result.quantized_weights.dtype)
        },
        "visualizations": {
            "original_heatmap": original_heatmap,
            "quantized_heatmap": quantized_heatmap,
            "dequantized_heatmap": dequantized_heatmap,
            "error_heatmap": error_heatmap,
            "original_histogram": original_hist,
            "quantized_histogram": quantized_hist,
            "scales_histogram": scales_hist
        }
    }


@router.post("/layer")
async def quantize_layer(request: QuantizeLayerRequest) -> Dict[str, Any]:
    """
    Quantize a specific layer from the loaded model.
    Requires a model to be loaded first.
    """
    if model_loader is None or model_loader.get_model() is None:
        raise HTTPException(
            status_code=400, 
            detail="No model loaded. Load a model first or use /quantize/weights for custom weights."
        )
    
    # Get layer weights
    weights = model_loader.get_layer_weights(request.layer_name)
    if weights is None:
        raise HTTPException(status_code=404, detail=f"Layer not found: {request.layer_name}")
    
    # Ensure 2D
    original_shape = weights.shape
    if len(weights.shape) == 1:
        weights = weights.unsqueeze(0)
    elif len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)
    
    # Get quantizer
    quantizer, config = _get_quantizer_from_config(request)
    
    # Quantize
    result = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(result)
    
    # Generate Visualizations
    original_hist = visualizer.to_dict(visualizer.weight_histogram(weights, "Original Distribution"))
    quantized_hist = visualizer.to_dict(visualizer.weight_histogram(result.quantized_weights.float(), "Quantized Distribution"))
    scales_hist = visualizer.to_dict(visualizer.scales_histogram(result.scales))

    return {
        "success": True,
        "layer_name": request.layer_name,
        "config": config.to_dict(),
        "stats": {
            "original_shape": list(original_shape),
            "quantized_shape": list(result.quantized_weights.shape),
            "scales_shape": list(result.scales.shape),
            "max_error": result.max_error,
            "mean_error": result.mean_error,
            "memory_savings_percent": result.memory_savings_percent,
            "original_dtype": str(weights.dtype),
            "quantized_dtype": str(result.quantized_weights.dtype)
        },
        "visualizations": {
            "original_heatmap": visualizer.to_dict(
                visualizer.weight_heatmap(weights, f"Original: {request.layer_name}")
            ),
            "quantized_heatmap": visualizer.to_dict(
                visualizer.weight_heatmap(result.quantized_weights.float(), f"Quantized ({request.bits}-bit)")
            ),
            "dequantized_heatmap": visualizer.to_dict(
                visualizer.weight_heatmap(dequantized, "Dequantized Weights")
            ),
            "error_heatmap": visualizer.to_dict(
                visualizer.weight_heatmap((weights - dequantized).abs(), "Error")
            ),
            "original_histogram": original_hist,
            "quantized_histogram": quantized_hist,
            "scales_histogram": scales_hist
        }
    }


@router.post("/model")
async def quantize_model(request: QuantizeModelRequest) -> Dict[str, Any]:
    """
    Quantize all quantizable layers in the loaded model.
    Returns summary statistics for all layers.
    """
    if model_loader is None or model_loader.get_model() is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. This feature requires a loaded model."
        )
    
    model_info = model_loader.get_model_info()
    if model_info is None:
        raise HTTPException(status_code=500, detail="Failed to get model info")
    
    # Determine layers to quantize
    if request.layers_to_include:
        layers_to_quantize = request.layers_to_include
    else:
        layers_to_quantize = model_info.quantizable_layers
    
    # Remove skipped layers
    layers_to_quantize = [l for l in layers_to_quantize if l not in request.layers_to_skip]
    
    # Get quantizer
    quantizer, config = _get_quantizer_from_config(request)
    
    # Quantize each layer
    results = []
    total_memory_saved = 0
    total_original_size = 0
    
    for layer_name in layers_to_quantize:
        weights = model_loader.get_layer_weights(layer_name)
        if weights is None:
            continue
        
        # Handle non-2D weights
        original_shape = weights.shape
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(0)
        elif len(weights.shape) > 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        try:
            result = quantizer.quantize(weights)
            
            original_bytes = weights.numel() * weights.element_size()
            total_original_size += original_bytes
            total_memory_saved += original_bytes * (result.memory_savings_percent / 100)
            
            results.append({
                "layer": layer_name,
                "shape": list(original_shape),
                "max_error": result.max_error,
                "mean_error": result.mean_error,
                "memory_savings_percent": result.memory_savings_percent
            })
        except Exception as e:
            results.append({
                "layer": layer_name,
                "error": str(e)
            })
    
    return {
        "success": True,
        "config": config.to_dict(),
        "summary": {
            "layers_quantized": len([r for r in results if "error" not in r]),
            "layers_failed": len([r for r in results if "error" in r]),
            "total_memory_saved_mb": total_memory_saved / (1024 * 1024),
            "average_memory_savings_percent": (total_memory_saved / total_original_size * 100) if total_original_size > 0 else 0
        },
        "layers": results
    }


# WebSocket for real-time progress
@router.websocket("/stream")
async def quantization_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming quantization progress"""
    await websocket.accept()
    
    try:
        while True:
            # Receive quantization request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Process and send updates
            await websocket.send_json({
                "type": "progress",
                "progress": 0,
                "message": "Starting quantization..."
            })
            
            # Simulate progress (in real implementation, this would be actual quantization)
            for i in range(0, 101, 10):
                await asyncio.sleep(0.1)
                await websocket.send_json({
                    "type": "progress", 
                    "progress": i,
                    "message": f"Processing... {i}%"
                })
            
            await websocket.send_json({
                "type": "complete",
                "message": "Quantization complete"
            })
            
    except WebSocketDisconnect:
        pass
