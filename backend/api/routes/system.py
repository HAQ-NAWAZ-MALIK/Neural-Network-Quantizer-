"""
System Routes
Hardware detection and system information
"""

from fastapi import APIRouter
from typing import Dict, Any

from backend.core.system_checker import system_checker, check_model_requirements

router = APIRouter()


@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    """
    Get complete system information including GPU, RAM, and capabilities.
    """
    return system_checker.to_dict()


@router.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """
    Get system capabilities for quantization tasks.
    """
    info = system_checker.check()
    return {
        "capability": info.capability.value,
        "recommended_batch_size": info.recommended_batch_size,
        "max_model_size": info.max_model_size,
        "cuda_available": info.cuda_available,
        "mps_available": info.mps_available,
        "gpus": [
            {
                "name": gpu.name,
                "memory_gb": gpu.total_memory_gb
            }
            for gpu in info.gpus
        ]
    }


@router.post("/check-model")
async def check_model_requirements_endpoint(
    model_params_billions: float,
    dtype: str = "fp16"
) -> Dict[str, Any]:
    """
    Check if system can handle a model of specified size.
    
    Args:
        model_params_billions: Model size in billions of parameters
        dtype: Data type (fp32, fp16, int8, int4)
    """
    return check_model_requirements(model_params_billions, dtype)


@router.get("/refresh")
async def refresh_system_info() -> Dict[str, Any]:
    """
    Force refresh system information.
    """
    return system_checker.check(force_refresh=True).__dict__
