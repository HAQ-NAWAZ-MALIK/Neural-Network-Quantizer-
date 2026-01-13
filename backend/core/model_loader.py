"""
HuggingFace Model Loader
Loads models from HuggingFace Hub or local files with memory-efficient options.
"""

import torch
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import (
        AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
        AutoTokenizer, AutoConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

if TYPE_CHECKING:
    from transformers import PreTrainedModel

from .system_checker import system_checker, check_model_requirements


class ModelType(Enum):
    """Supported model types"""
    CAUSAL_LM = "causal_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"  
    GENERIC = "generic"


@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    module_type: str
    shape: Optional[Tuple[int, ...]]
    num_params: int
    dtype: str
    is_quantizable: bool


@dataclass 
class ModelInfo:
    """Complete model information"""
    name: str
    model_type: ModelType
    architecture: str
    num_params: int
    num_params_billions: float
    hidden_size: int
    num_layers: int
    vocab_size: Optional[int]
    dtype: str
    memory_footprint_gb: float
    layers: List[LayerInfo]
    quantizable_layers: List[str]


class ModelLoader:
    """
    Load and inspect HuggingFace models with memory-efficient options.
    Provides layer-by-layer analysis for selective quantization.
    """
    
    # Layer types that can be quantized
    QUANTIZABLE_TYPES = (
        "Linear",
        "Conv1d", 
        "Conv2d",
        "Embedding"
    )
    
    def __init__(self):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
        self._loaded_model = None  # Optional[PreTrainedModel]
        self._model_info: Optional[ModelInfo] = None
        self._tokenizer = None
    
    def check_requirements(self, model_name: str, dtype: str = "fp16") -> Dict[str, Any]:
        """Check if system can load the model before attempting"""
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Estimate parameters
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            else:
                # Estimate from config
                hidden = getattr(config, 'hidden_size', 768)
                layers = getattr(config, 'num_hidden_layers', 12)
                vocab = getattr(config, 'vocab_size', 30000)
                num_params = self._estimate_params(hidden, layers, vocab)
            
            params_billions = num_params / 1e9
            return check_model_requirements(params_billions, dtype)
            
        except Exception as e:
            return {
                "can_load": False,
                "error": str(e),
                "warnings": [f"Failed to fetch model config: {str(e)}"]
            }
    
    def _estimate_params(self, hidden: int, layers: int, vocab: int) -> int:
        """Estimate parameter count from config"""
        # Rough estimate: embeddings + transformer layers
        embedding_params = vocab * hidden
        # Each layer: attention (4 * hidden^2) + FFN (8 * hidden^2)
        layer_params = layers * (12 * hidden * hidden)
        return embedding_params + layer_params
    
    def load(self, model_name: str, 
             model_type: ModelType = ModelType.GENERIC,
             dtype: str = "auto",
             device: str = "auto",
             trust_remote_code: bool = True,
             low_memory: bool = False) -> Tuple[Any, Optional[Any]]:
        """
        Load a model from HuggingFace Hub or local path.
        
        Args:
            model_name: HuggingFace model ID or local path
            model_type: Type of model to load
            dtype: Data type ("auto", "fp32", "fp16", "bf16")
            device: Device to load to ("auto", "cuda", "cpu", "mps")
            trust_remote_code: Allow custom code from model repos
            low_memory: Use memory-efficient loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Clear previous model
        self.unload()
        
        # Determine device
        if device == "auto":
            sys_info = system_checker.check()
            if sys_info.cuda_available:
                device = "cuda"
            elif sys_info.mps_available:
                device = "mps"
            else:
                device = "cpu"
        
        # Determine dtype
        if dtype == "auto":
            if device == "cuda":
                dtype = "fp16"
            elif device == "mps":
                dtype = "fp32"  # MPS has limited bf16 support
            else:
                dtype = "fp32"
        
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }.get(dtype, torch.float32)
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # Select model class
        if model_type == ModelType.CAUSAL_LM:
            model_class = AutoModelForCausalLM
        elif model_type == ModelType.SEQUENCE_CLASSIFICATION:
            model_class = AutoModelForSequenceClassification
        else:
            model_class = AutoModel
        
        # Load model
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        
        if low_memory:
            load_kwargs["low_cpu_mem_usage"] = True
            if device == "cuda":
                load_kwargs["device_map"] = "auto"
        
        model = model_class.from_pretrained(**load_kwargs)
        
        if not low_memory and device != "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        except Exception:
            tokenizer = None
        
        self._loaded_model = model
        self._tokenizer = tokenizer
        self._model_info = self._analyze_model(model, model_name, model_type)
        
        return model, tokenizer
    
    def load_weights_only(self, model_name: str) -> Dict[str, torch.Tensor]:
        """
        Load only the state dict without instantiating the model.
        More memory efficient for inspection.
        """
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        
        try:
            # Try safetensors first
            path = hf_hub_download(model_name, "model.safetensors")
            weights = {}
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            return weights
        except Exception:
            # Fallback to torch
            try:
                path = hf_hub_download(model_name, "pytorch_model.bin")
                return torch.load(path, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load weights: {str(e)}")
    
    def _analyze_model(self, model: Any, name: str, 
                       model_type: ModelType) -> ModelInfo:
        """Analyze model structure and extract layer information"""
        layers = []
        quantizable_layers = []
        total_params = 0
        
        for layer_name, module in model.named_modules():
            if not layer_name:
                continue
            
            # Get module info
            module_type = module.__class__.__name__
            
            # Check if quantizable
            is_quantizable = any(
                qt in module_type for qt in self.QUANTIZABLE_TYPES
            )
            
            # Get shape and params for leaf modules
            shape = None
            num_params = 0
            dtype = "N/A"
            
            if hasattr(module, 'weight') and module.weight is not None:
                shape = tuple(module.weight.shape)
                num_params = module.weight.numel()
                dtype = str(module.weight.dtype)
                if hasattr(module, 'bias') and module.bias is not None:
                    num_params += module.bias.numel()
            
            if num_params > 0:
                total_params += num_params
                layers.append(LayerInfo(
                    name=layer_name,
                    module_type=module_type,
                    shape=shape,
                    num_params=num_params,
                    dtype=dtype,
                    is_quantizable=is_quantizable
                ))
                
                if is_quantizable:
                    quantizable_layers.append(layer_name)
        
        # Get config info
        config = model.config
        hidden_size = getattr(config, 'hidden_size', 768)
        num_layers = getattr(config, 'num_hidden_layers', 12)
        vocab_size = getattr(config, 'vocab_size', None)
        
        # Calculate memory
        memory_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        
        return ModelInfo(
            name=name,
            model_type=model_type,
            architecture=config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "Unknown",
            num_params=total_params,
            num_params_billions=total_params / 1e9,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            dtype=str(next(model.parameters()).dtype),
            memory_footprint_gb=round(memory_gb, 2),
            layers=layers,
            quantizable_layers=quantizable_layers
        )

    def register_model(self, model: Any, name: str, tokenizer: Any = None):
        """Register an externally loaded model"""
        self._loaded_model = model
        self._tokenizer = tokenizer
        self._model_info = self._analyze_model(model, name, ModelType.GENERIC)
    
    def get_layer_weights(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get weights from a specific layer"""
        if self._loaded_model is None:
            raise RuntimeError("No model loaded")
        
        for name, module in self._loaded_model.named_modules():
            if name == layer_name:
                if hasattr(module, 'weight'):
                    return module.weight.data.clone()
        return None
    
    def set_layer_weights(self, layer_name: str, weights: torch.Tensor):
        """Set weights for a specific layer"""
        if self._loaded_model is None:
            raise RuntimeError("No model loaded")
        
        for name, module in self._loaded_model.named_modules():
            if name == layer_name:
                if hasattr(module, 'weight'):
                    module.weight.data = weights.to(module.weight.device)
                    return
        raise ValueError(f"Layer not found: {layer_name}")
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information"""
        return self._model_info
    
    def get_model(self) -> Optional[Any]:
        """Get loaded model"""
        return self._loaded_model
    
    def get_tokenizer(self):
        """Get loaded tokenizer"""
        return self._tokenizer
    
    def unload(self):
        """Unload model and free memory"""
        if self._loaded_model is not None:
            del self._loaded_model
            self._loaded_model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        self._model_info = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert model info to dictionary"""
        if self._model_info is None:
            return None
        
        info = self._model_info
        return {
            "name": info.name,
            "model_type": info.model_type.value,
            "architecture": info.architecture,
            "num_params": info.num_params,
            "num_params_billions": round(info.num_params_billions, 3),
            "hidden_size": info.hidden_size,
            "num_layers": info.num_layers,
            "vocab_size": info.vocab_size,
            "dtype": info.dtype,
            "memory_footprint_gb": info.memory_footprint_gb,
            "num_quantizable_layers": len(info.quantizable_layers),
            "quantizable_layers": info.quantizable_layers,
            "layers": [
                {
                    "name": layer.name,
                    "module_type": layer.module_type,
                    "shape": layer.shape,
                    "num_params": layer.num_params,
                    "dtype": layer.dtype,
                    "is_quantizable": layer.is_quantizable
                }
                for layer in info.layers
            ]
        }


# Global instance
model_loader = ModelLoader() if HAS_TRANSFORMERS else None


def load_model(model_name: str, **kwargs) -> Tuple[Any, Any]:
    """Convenience function to load a model"""
    if model_loader is None:
        raise ImportError("transformers not available")
    return model_loader.load(model_name, **kwargs)


def get_model_info() -> Optional[Dict[str, Any]]:
    """Get current model information"""
    if model_loader is None:
        return None
    return model_loader.to_dict()
