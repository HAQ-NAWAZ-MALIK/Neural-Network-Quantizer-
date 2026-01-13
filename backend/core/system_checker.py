"""
System Requirements Checker
Detects GPU availability, memory, and provides hardware recommendations.
"""

import torch
import psutil
import platform
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class HardwareCapability(Enum):
    """Hardware capability levels"""
    FULL_GPU = "full_gpu"           # CUDA GPU with sufficient VRAM
    LIMITED_GPU = "limited_gpu"      # CUDA GPU with limited VRAM
    CPU_ONLY = "cpu_only"           # No GPU available
    APPLE_SILICON = "apple_silicon"  # M1/M2/M3 with MPS


@dataclass
class GPUInfo:
    """Information about a GPU device"""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: Optional[str] = None


@dataclass
class SystemInfo:
    """Complete system information"""
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    mps_available: bool
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: List[GPUInfo]
    capability: HardwareCapability
    recommended_batch_size: int
    max_model_size: str
    warnings: List[str]


class SystemChecker:
    """Check system capabilities for quantization tasks"""
    
    # Model size thresholds (in billions of parameters)
    MODEL_SIZES = {
        "tiny": 0.1,      # ~100M params
        "small": 0.5,     # ~500M params  
        "medium": 1.0,    # ~1B params
        "large": 7.0,     # ~7B params
        "xlarge": 13.0,   # ~13B params
        "xxlarge": 70.0   # ~70B params
    }
    
    # Memory requirements per billion parameters (GB)
    MEMORY_PER_BILLION_PARAMS = {
        "fp32": 4.0,
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5
    }
    
    def __init__(self):
        self._system_info: Optional[SystemInfo] = None
    
    def check(self, force_refresh: bool = False) -> SystemInfo:
        """Perform full system check"""
        if self._system_info is not None and not force_refresh:
            return self._system_info
        
        warnings = []
        gpus = []
        
        # Basic info
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # CUDA version
        cuda_version = None
        if cuda_available:
            cuda_version = torch.version.cuda
        
        # GPU detection
        if cuda_available:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / (1024**3)
                    free_mem = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
                    
                    gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_gb=round(total_mem, 2),
                        free_memory_gb=round(free_mem, 2),
                        compute_capability=f"{props.major}.{props.minor}"
                    ))
            except Exception as e:
                warnings.append(f"Error detecting GPU: {str(e)}")
        
        # RAM info
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        
        # Determine capability
        capability = self._determine_capability(gpus, mps_available, ram_total_gb)
        
        # Recommendations
        recommended_batch_size = self._get_recommended_batch_size(capability, gpus)
        max_model_size = self._get_max_model_size(capability, gpus, ram_total_gb)
        
        # Add warnings
        if not cuda_available and not mps_available:
            warnings.append("No GPU detected. Quantization will run on CPU (slower).")
        
        if ram_available_gb < 8:
            warnings.append(f"Low RAM available ({ram_available_gb:.1f}GB). Large models may fail.")
        
        if gpus and gpus[0].free_memory_gb < 4:
            warnings.append(f"Low GPU memory ({gpus[0].free_memory_gb:.1f}GB free). Consider smaller models.")
        
        self._system_info = SystemInfo(
            platform=platform.system(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            mps_available=mps_available,
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            ram_total_gb=round(ram_total_gb, 2),
            ram_available_gb=round(ram_available_gb, 2),
            gpus=gpus,
            capability=capability,
            recommended_batch_size=recommended_batch_size,
            max_model_size=max_model_size,
            warnings=warnings
        )
        
        return self._system_info
    
    def _determine_capability(self, gpus: List[GPUInfo], mps_available: bool,
                              ram_total_gb: float) -> HardwareCapability:
        """Determine hardware capability level"""
        if mps_available:
            return HardwareCapability.APPLE_SILICON
        
        if not gpus:
            return HardwareCapability.CPU_ONLY
        
        # Check if any GPU has >= 8GB VRAM
        max_vram = max(gpu.total_memory_gb for gpu in gpus)
        
        if max_vram >= 8:
            return HardwareCapability.FULL_GPU
        else:
            return HardwareCapability.LIMITED_GPU
    
    def _get_recommended_batch_size(self, capability: HardwareCapability,
                                     gpus: List[GPUInfo]) -> int:
        """Get recommended batch size based on hardware"""
        if capability == HardwareCapability.CPU_ONLY:
            return 1
        elif capability == HardwareCapability.LIMITED_GPU:
            return 4
        elif capability == HardwareCapability.APPLE_SILICON:
            return 8
        else:
            # Full GPU - scale with VRAM
            if gpus:
                vram = gpus[0].total_memory_gb
                if vram >= 24:
                    return 32
                elif vram >= 16:
                    return 16
                elif vram >= 8:
                    return 8
            return 8
    
    def _get_max_model_size(self, capability: HardwareCapability,
                            gpus: List[GPUInfo], ram_gb: float) -> str:
        """Get maximum recommended model size"""
        if capability == HardwareCapability.CPU_ONLY:
            # CPU-only: limited by RAM, very slow for large models
            if ram_gb >= 32:
                return "medium (1B)"
            elif ram_gb >= 16:
                return "small (500M)"
            else:
                return "tiny (100M)"
        
        elif capability == HardwareCapability.LIMITED_GPU:
            return "small (500M)"
        
        elif capability == HardwareCapability.APPLE_SILICON:
            # Apple Silicon: depends on unified memory
            if ram_gb >= 32:
                return "large (7B)"
            elif ram_gb >= 16:
                return "medium (1B)"
            else:
                return "small (500M)"
        
        else:  # FULL_GPU
            if gpus:
                vram = gpus[0].total_memory_gb
                if vram >= 48:
                    return "xxlarge (70B)"
                elif vram >= 24:
                    return "xlarge (13B)"
                elif vram >= 16:
                    return "large (7B)"
                elif vram >= 8:
                    return "medium (1B)"
            return "medium (1B)"
    
    def can_load_model(self, model_params_billions: float, 
                       dtype: str = "fp16") -> Dict[str, Any]:
        """Check if a specific model can be loaded"""
        info = self.check()
        
        memory_required = model_params_billions * self.MEMORY_PER_BILLION_PARAMS.get(dtype, 2.0)
        memory_required *= 1.3  # 30% overhead for activations, optimizer, etc.
        
        # Check GPU memory
        gpu_ok = False
        gpu_memory = 0
        if info.gpus:
            gpu_memory = info.gpus[0].free_memory_gb
            gpu_ok = gpu_memory >= memory_required
        
        # Check RAM
        ram_ok = info.ram_available_gb >= memory_required
        
        can_load = gpu_ok or (info.capability == HardwareCapability.CPU_ONLY and ram_ok)
        
        return {
            "can_load": can_load,
            "memory_required_gb": round(memory_required, 2),
            "gpu_available_gb": round(gpu_memory, 2) if info.gpus else 0,
            "ram_available_gb": round(info.ram_available_gb, 2),
            "recommended_device": "cuda" if gpu_ok else ("mps" if info.mps_available else "cpu"),
            "warnings": [] if can_load else [
                f"Model requires ~{memory_required:.1f}GB memory. " +
                f"Available: GPU={gpu_memory:.1f}GB, RAM={info.ram_available_gb:.1f}GB"
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system info to dictionary"""
        info = self.check()
        return {
            "platform": info.platform,
            "python_version": info.python_version,
            "torch_version": info.torch_version,
            "cuda_available": info.cuda_available,
            "cuda_version": info.cuda_version,
            "mps_available": info.mps_available,
            "cpu_cores": info.cpu_cores,
            "ram_total_gb": info.ram_total_gb,
            "ram_available_gb": info.ram_available_gb,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_gb": gpu.total_memory_gb,
                    "free_memory_gb": gpu.free_memory_gb,
                    "compute_capability": gpu.compute_capability
                }
                for gpu in info.gpus
            ],
            "capability": info.capability.value,
            "recommended_batch_size": info.recommended_batch_size,
            "max_model_size": info.max_model_size,
            "warnings": info.warnings
        }


# Global instance
system_checker = SystemChecker()


def get_system_info() -> Dict[str, Any]:
    """Get system information as dictionary"""
    return system_checker.to_dict()


def check_model_requirements(model_params_billions: float, dtype: str = "fp16") -> Dict[str, Any]:
    """Check if system can handle a specific model"""
    return system_checker.can_load_model(model_params_billions, dtype)
