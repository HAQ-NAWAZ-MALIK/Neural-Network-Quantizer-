"""
Analysis Routes
Weight analysis and visualization endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch

from backend.core.model_loader import model_loader
from backend.core.visualization import visualizer
from backend.core.quantizer import (
    QuantizationConfig, QuantizationMethod, QuantizationMode,
    get_quantizer
)

router = APIRouter()


class AnalyzeLayerRequest(BaseModel):
    """Request to analyze a specific layer"""
    layer_name: str


class CompareQuantizationRequest(BaseModel):
    """Compare different quantization methods on same weights"""
    layer_name: Optional[str] = None
    in_features: int = 64
    out_features: int = 128
    methods: List[str] = ["int8", "int4", "nf4"]


@router.get("/weights/{layer_name}")
async def get_weight_analysis(layer_name: str) -> Dict[str, Any]:
    """
    Get detailed weight analysis for a specific layer.
    """
    if model_loader is None or model_loader.get_model() is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    weights = model_loader.get_layer_weights(layer_name)
    if weights is None:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")
    
    # Flatten for analysis
    flat = weights.flatten()
    
    # Statistics
    stats = {
        "shape": list(weights.shape),
        "dtype": str(weights.dtype),
        "num_params": int(weights.numel()),
        "memory_mb": weights.numel() * weights.element_size() / (1024 * 1024),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "median": float(torch.median(flat)),
        "sparsity": float((weights == 0).sum() / weights.numel()),
        "abs_mean": float(weights.abs().mean()),
        "percentiles": {
            "1%": float(torch.quantile(flat.float(), 0.01)),
            "5%": float(torch.quantile(flat.float(), 0.05)),
            "25%": float(torch.quantile(flat.float(), 0.25)),
            "50%": float(torch.quantile(flat.float(), 0.50)),
            "75%": float(torch.quantile(flat.float(), 0.75)),
            "95%": float(torch.quantile(flat.float(), 0.95)),
            "99%": float(torch.quantile(flat.float(), 0.99))
        }
    }
    
    # Visualizations
    heatmap = visualizer.to_dict(
        visualizer.weight_heatmap(weights, f"Weights: {layer_name}")
    )
    histogram = visualizer.to_dict(
        visualizer.weight_histogram(weights, "Weight Distribution")
    )
    
    return {
        "layer_name": layer_name,
        "stats": stats,
        "visualizations": {
            "heatmap": heatmap,
            "histogram": histogram
        }
    }


@router.post("/compare")
async def compare_quantization_methods(request: CompareQuantizationRequest) -> Dict[str, Any]:
    """
    Compare multiple quantization methods on the same weights.
    """
    # Get or generate weights
    if request.layer_name and model_loader and model_loader.get_model():
        weights = model_loader.get_layer_weights(request.layer_name)
        if weights is None:
            raise HTTPException(status_code=404, detail=f"Layer not found: {request.layer_name}")
        source = f"layer:{request.layer_name}"
    else:
        weights = torch.randn(request.out_features, request.in_features)
        source = "random"
    
    # Ensure 2D
    if len(weights.shape) == 1:
        weights = weights.unsqueeze(0)
    elif len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)
    
    # Compare methods
    method_map = {
        "int8": QuantizationMethod.INT8,
        "int4": QuantizationMethod.INT4,
        "nf4": QuantizationMethod.NF4
    }
    
    comparison = []
    
    for method_name in request.methods:
        if method_name not in method_map:
            continue
        
        config = QuantizationConfig(
            bits=8 if method_name == "int8" else 4,
            method=method_map[method_name],
            group_size=128 if method_name in ["int4", "nf4"] else None
        )
        
        try:
            quantizer = get_quantizer(config)
            result = quantizer.quantize(weights)
            
            comparison.append({
                "method": method_name,
                "bits": config.bits,
                "max_error": result.max_error,
                "mean_error": result.mean_error,
                "memory_savings_percent": result.memory_savings_percent,
                "histogram": visualizer.to_dict(
                    visualizer.weight_histogram(
                        result.quantized_weights.float(), 
                        f"{method_name.upper()} Distribution"
                    )
                )
            })
        except Exception as e:
            comparison.append({
                "method": method_name,
                "error": str(e)
            })
    
    return {
        "source": source,
        "original_shape": list(weights.shape),
        "original_stats": {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std())
        },
        "comparison": comparison
    }


@router.get("/model-summary")
async def get_model_summary() -> Dict[str, Any]:
    """
    Get summary statistics for all layers in loaded model.
    """
    if model_loader is None or model_loader.get_model() is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    model_info = model_loader.get_model_info()
    if model_info is None:
        raise HTTPException(status_code=500, detail="Failed to get model info")
    
    # Analyze each layer
    layer_stats = []
    total_params = 0
    quantizable_params = 0
    
    for layer in model_info.layers:
        total_params += layer.num_params
        if layer.is_quantizable:
            quantizable_params += layer.num_params
        
        layer_stats.append({
            "name": layer.name,
            "type": layer.module_type,
            "params": layer.num_params,
            "params_mb": layer.num_params * 4 / (1024 * 1024),  # Assuming FP32
            "quantizable": layer.is_quantizable
        })
    
    # Sort by parameter count
    layer_stats.sort(key=lambda x: x["params"], reverse=True)
    
    return {
        "model_name": model_info.name,
        "architecture": model_info.architecture,
        "total_params": total_params,
        "total_params_billions": total_params / 1e9,
        "quantizable_params": quantizable_params,
        "quantizable_percent": quantizable_params / total_params * 100 if total_params > 0 else 0,
        "memory_fp32_gb": total_params * 4 / (1024**3),
        "memory_int8_estimate_gb": quantizable_params * 1 / (1024**3) + (total_params - quantizable_params) * 4 / (1024**3),
        "memory_int4_estimate_gb": quantizable_params * 0.5 / (1024**3) + (total_params - quantizable_params) * 4 / (1024**3),
        "top_layers": layer_stats[:20]  # Top 20 largest layers
    }


@router.get("/outliers/{layer_name}")
async def detect_outliers(layer_name: str, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect outlier weights that may cause quantization issues.
    """
    if model_loader is None or model_loader.get_model() is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    weights = model_loader.get_layer_weights(layer_name)
    if weights is None:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")
    
    flat = weights.flatten()
    mean = flat.mean()
    std = flat.std()
    
    # Find outliers (values beyond threshold * std from mean)
    outlier_mask = (flat - mean).abs() > threshold * std
    num_outliers = outlier_mask.sum().item()
    outlier_values = flat[outlier_mask].tolist()[:100]  # Limit to 100
    
    return {
        "layer_name": layer_name,
        "threshold": threshold,
        "total_weights": int(flat.numel()),
        "num_outliers": num_outliers,
        "outlier_percent": num_outliers / flat.numel() * 100,
        "mean": float(mean),
        "std": float(std),
        "outlier_range": {
            "below": float(mean - threshold * std),
            "above": float(mean + threshold * std)
        },
        "sample_outliers": outlier_values,
        "recommendation": "Consider clipping or mixed-precision for this layer" if num_outliers > flat.numel() * 0.01 else "Layer is suitable for quantization"
    }
