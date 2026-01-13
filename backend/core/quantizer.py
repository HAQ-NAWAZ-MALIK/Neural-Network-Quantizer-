"""
Multi-bit Weight Quantization Engine
Supports INT8, INT4, NF4, and GPTQ-style quantization methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum


class QuantizationMethod(Enum):
    """Supported quantization methods"""
    INT8 = "int8"           # 8-bit integer quantization
    INT4 = "int4"           # 4-bit integer quantization
    NF4 = "nf4"             # Normal Float 4-bit (QLoRA style)
    GPTQ = "gptq"           # GPTQ reconstruction-based


class QuantizationMode(Enum):
    """Quantization modes"""
    SYMMETRIC = "symmetric"     # Range: [-max, max]
    ASYMMETRIC = "asymmetric"   # Range: [min, max]


@dataclass
class QuantizationConfig:
    """Configuration for quantization process"""
    bits: int = 8
    method: QuantizationMethod = QuantizationMethod.INT8
    mode: QuantizationMode = QuantizationMode.SYMMETRIC
    group_size: Optional[int] = None  # None = per-channel, else group quantization
    use_double_quant: bool = False    # Double quantization for scales
    compute_dtype: torch.dtype = torch.float32
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "method": self.method.value,
            "mode": self.mode.value,
            "group_size": self.group_size,
            "use_double_quant": self.use_double_quant,
            "compute_dtype": str(self.compute_dtype)
        }


@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    quantized_weights: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    original_shape: Tuple[int, ...]
    config: QuantizationConfig
    max_error: float
    mean_error: float
    memory_savings_percent: float


class BaseQuantizer:
    """Base class for all quantizers"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
    
    def quantize(self, weights: torch.Tensor) -> QuantizationResult:
        raise NotImplementedError
    
    def dequantize(self, result: QuantizationResult) -> torch.Tensor:
        raise NotImplementedError
    
    def _calculate_error(self, original: torch.Tensor, dequantized: torch.Tensor) -> Tuple[float, float]:
        """Calculate quantization error metrics"""
        abs_error = (original - dequantized).abs()
        return abs_error.max().item(), abs_error.mean().item()
    
    def _calculate_memory_savings(self, original: torch.Tensor, quantized: torch.Tensor, 
                                   scales: torch.Tensor) -> float:
        """Calculate memory savings percentage"""
        original_bytes = original.numel() * original.element_size()
        quantized_bytes = quantized.numel() * quantized.element_size()
        scales_bytes = scales.numel() * scales.element_size()
        
        new_bytes = quantized_bytes + scales_bytes
        savings = 100 * (1 - new_bytes / original_bytes)
        return savings


class INT8Quantizer(BaseQuantizer):
    """8-bit integer quantization (W8A16)"""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        if config is None:
            config = QuantizationConfig(bits=8, method=QuantizationMethod.INT8)
        super().__init__(config)
    
    def quantize(self, weights: torch.Tensor) -> QuantizationResult:
        """
        Quantize weights to INT8 precision.
        
        Args:
            weights: Tensor of shape (out_features, in_features)
            
        Returns:
            QuantizationResult with int8 weights and scales
        """
        original_shape = weights.shape
        w_fp32 = weights.clone().to(torch.float32)
        
        if self.config.group_size is not None:
            # Group quantization
            return self._quantize_grouped(w_fp32, original_shape)
        
        # Per-channel quantization
        if self.config.mode == QuantizationMode.SYMMETRIC:
            # Symmetric: scale = max(|w|) / 127
            scales = w_fp32.abs().max(dim=-1).values / 127
            scales = scales.clamp(min=1e-8)  # Avoid division by zero
            zero_points = None
            
            # Quantize
            int8_weights = torch.round(w_fp32 / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
            
        else:
            # Asymmetric: use full [-128, 127] range
            w_min = w_fp32.min(dim=-1).values
            w_max = w_fp32.max(dim=-1).values
            
            scales = (w_max - w_min) / 255
            scales = scales.clamp(min=1e-8)
            zero_points = torch.round(-w_min / scales).clamp(0, 255).to(torch.int32)
            
            # Quantize
            int8_weights = torch.round(w_fp32 / scales.unsqueeze(1) + zero_points.unsqueeze(1))
            int8_weights = int8_weights.clamp(-128, 127).to(torch.int8)
        
        # Dequantize for error calculation
        dequantized = self.dequantize_weights(int8_weights, scales, zero_points)
        max_error, mean_error = self._calculate_error(weights, dequantized)
        memory_savings = self._calculate_memory_savings(weights, int8_weights, scales)
        
        return QuantizationResult(
            quantized_weights=int8_weights,
            scales=scales,
            zero_points=zero_points,
            original_shape=original_shape,
            config=self.config,
            max_error=max_error,
            mean_error=mean_error,
            memory_savings_percent=memory_savings
        )
    
    def _quantize_grouped(self, weights: torch.Tensor, original_shape: Tuple[int, ...]) -> QuantizationResult:
        """Quantize with group-wise scaling"""
        out_features, in_features = weights.shape
        group_size = self.config.group_size
        
        # Pad if necessary
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weights = F.pad(weights, (0, pad_size))
            in_features = weights.shape[1]
        
        # Reshape for group quantization
        num_groups = in_features // group_size
        weights_grouped = weights.reshape(out_features, num_groups, group_size)
        
        # Calculate scales per group
        scales = weights_grouped.abs().max(dim=-1).values / 127
        scales = scales.clamp(min=1e-8)
        
        # Quantize
        int8_weights = torch.round(weights_grouped / scales.unsqueeze(-1))
        int8_weights = int8_weights.clamp(-128, 127).to(torch.int8)
        int8_weights = int8_weights.reshape(out_features, in_features)
        
        # Trim padding
        int8_weights = int8_weights[:, :original_shape[1]]
        scales = scales.reshape(out_features, num_groups)
        
        # Dequantize for error calculation  
        dequantized = self.dequantize_weights(int8_weights, scales, None, group_size)
        max_error, mean_error = self._calculate_error(
            weights[:, :original_shape[1]], dequantized
        )
        memory_savings = self._calculate_memory_savings(
            weights[:, :original_shape[1]], int8_weights, scales
        )
        
        return QuantizationResult(
            quantized_weights=int8_weights,
            scales=scales,
            zero_points=None,
            original_shape=original_shape,
            config=self.config,
            max_error=max_error,
            mean_error=mean_error,
            memory_savings_percent=memory_savings
        )
    
    def dequantize_weights(self, int8_weights: torch.Tensor, scales: torch.Tensor,
                           zero_points: Optional[torch.Tensor] = None,
                           group_size: Optional[int] = None) -> torch.Tensor:
        """Dequantize INT8 weights back to floating point"""
        if group_size is not None:
            # Group dequantization
            out_features, in_features = int8_weights.shape
            num_groups = scales.shape[1]
            
            # Expand scales to match weight shape
            scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size)
            scales_expanded = scales_expanded.reshape(out_features, -1)[:, :in_features]
            
            return int8_weights.float() * scales_expanded
        
        if zero_points is not None:
            # Asymmetric dequantization
            return (int8_weights.float() - zero_points.unsqueeze(1).float()) * scales.unsqueeze(1)
        
        # Symmetric dequantization
        return int8_weights.float() * scales.unsqueeze(1)
    
    def dequantize(self, result: QuantizationResult) -> torch.Tensor:
        """Dequantize from QuantizationResult"""
        return self.dequantize_weights(
            result.quantized_weights, 
            result.scales, 
            result.zero_points,
            result.config.group_size
        )


class INT4Quantizer(BaseQuantizer):
    """4-bit integer quantization (W4A16)"""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        if config is None:
            config = QuantizationConfig(bits=4, method=QuantizationMethod.INT4, group_size=128)
        super().__init__(config)
    
    def quantize(self, weights: torch.Tensor) -> QuantizationResult:
        """
        Quantize weights to INT4 precision.
        Uses group quantization for better accuracy.
        
        Args:
            weights: Tensor of shape (out_features, in_features)
            
        Returns:
            QuantizationResult with packed int4 weights and scales
        """
        original_shape = weights.shape
        w_fp32 = weights.clone().to(torch.float32)
        out_features, in_features = w_fp32.shape
        
        group_size = self.config.group_size or 128
        
        # Pad if necessary
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            w_fp32 = F.pad(w_fp32, (0, pad_size))
            in_features = w_fp32.shape[1]
        
        # Reshape for group quantization
        num_groups = in_features // group_size
        weights_grouped = w_fp32.reshape(out_features, num_groups, group_size)
        
        if self.config.mode == QuantizationMode.SYMMETRIC:
            # Symmetric: range [-8, 7] for signed int4
            scales = weights_grouped.abs().max(dim=-1).values / 7
            scales = scales.clamp(min=1e-8)
            zero_points = None
            
            # Quantize to int4 range
            int4_weights = torch.round(weights_grouped / scales.unsqueeze(-1))
            int4_weights = int4_weights.clamp(-8, 7).to(torch.int8)  # Store as int8
        else:
            # Asymmetric: range [0, 15] for unsigned int4
            w_min = weights_grouped.min(dim=-1).values
            w_max = weights_grouped.max(dim=-1).values
            
            scales = (w_max - w_min) / 15
            scales = scales.clamp(min=1e-8)
            zero_points = torch.round(-w_min / scales).clamp(0, 15).to(torch.int8)
            
            int4_weights = torch.round(weights_grouped / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
            int4_weights = int4_weights.clamp(0, 15).to(torch.int8)
        
        # Reshape back
        int4_weights = int4_weights.reshape(out_features, in_features)
        int4_weights = int4_weights[:, :original_shape[1]]
        
        # Pack two int4 values into one int8 (for memory efficiency)
        packed_weights = self._pack_int4(int4_weights)
        
        # Dequantize for error calculation
        dequantized = self.dequantize_weights(int4_weights, scales, zero_points, group_size)
        dequantized = dequantized[:, :original_shape[1]]
        
        max_error, mean_error = self._calculate_error(weights, dequantized)
        
        # Memory savings: int4 is half of int8
        original_bytes = weights.numel() * weights.element_size()
        packed_bytes = packed_weights.numel() * packed_weights.element_size()
        scales_bytes = scales.numel() * scales.element_size()
        memory_savings = 100 * (1 - (packed_bytes + scales_bytes) / original_bytes)
        
        return QuantizationResult(
            quantized_weights=packed_weights,
            scales=scales.reshape(out_features, num_groups),
            zero_points=zero_points.reshape(out_features, num_groups) if zero_points is not None else None,
            original_shape=original_shape,
            config=self.config,
            max_error=max_error,
            mean_error=mean_error,
            memory_savings_percent=memory_savings
        )
    
    def _pack_int4(self, int4_weights: torch.Tensor) -> torch.Tensor:
        """Pack two int4 values into one int8"""
        out_features, in_features = int4_weights.shape
        
        # Ensure even number of features
        if in_features % 2 != 0:
            int4_weights = F.pad(int4_weights, (0, 1))
            in_features += 1
        
        # Reshape and pack
        reshaped = int4_weights.reshape(out_features, in_features // 2, 2)
        # Pack: low 4 bits + high 4 bits
        packed = (reshaped[:, :, 0] & 0x0F) | ((reshaped[:, :, 1] & 0x0F) << 4)
        return packed.to(torch.int8)
    
    def _unpack_int4(self, packed_weights: torch.Tensor, original_in_features: int) -> torch.Tensor:
        """Unpack int8 to two int4 values"""
        out_features = packed_weights.shape[0]
        
        # Unpack
        low = packed_weights & 0x0F
        high = (packed_weights >> 4) & 0x0F
        
        # Handle signed values
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        
        # Interleave
        unpacked = torch.stack([low, high], dim=-1).reshape(out_features, -1)
        return unpacked[:, :original_in_features]
    
    def dequantize_weights(self, int4_weights: torch.Tensor, scales: torch.Tensor,
                           zero_points: Optional[torch.Tensor] = None,
                           group_size: Optional[int] = None) -> torch.Tensor:
        """Dequantize INT4 weights back to floating point"""
        out_features, in_features = int4_weights.shape
        group_size = group_size or self.config.group_size or 128
        num_groups = scales.shape[1] if scales.dim() > 1 else 1
        
        # Expand scales
        scales_flat = scales.reshape(out_features, num_groups)
        scales_expanded = scales_flat.unsqueeze(-1).expand(-1, -1, group_size)
        scales_expanded = scales_expanded.reshape(out_features, -1)[:, :in_features]
        
        if zero_points is not None:
            zp_flat = zero_points.reshape(out_features, num_groups)
            zp_expanded = zp_flat.unsqueeze(-1).expand(-1, -1, group_size)
            zp_expanded = zp_expanded.reshape(out_features, -1)[:, :in_features]
            return (int4_weights.float() - zp_expanded.float()) * scales_expanded
        
        return int4_weights.float() * scales_expanded
    
    def dequantize(self, result: QuantizationResult) -> torch.Tensor:
        """Dequantize from QuantizationResult (handles packed weights)"""
        unpacked = self._unpack_int4(result.quantized_weights, result.original_shape[1])
        return self.dequantize_weights(
            unpacked,
            result.scales,
            result.zero_points,
            result.config.group_size
        )


class NF4Quantizer(BaseQuantizer):
    """
    Normal Float 4-bit quantization (NF4).
    Uses a fixed codebook optimized for normally distributed weights.
    """
    
    # NF4 codebook: values optimized for normal distribution
    NF4_CODEBOOK = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611008348274231,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ])
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        if config is None:
            config = QuantizationConfig(bits=4, method=QuantizationMethod.NF4, group_size=64)
        super().__init__(config)
        
        # Fix the codebook
        self.codebook = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
        ])
    
    def quantize(self, weights: torch.Tensor) -> QuantizationResult:
        """Quantize weights using NF4 codebook"""
        original_shape = weights.shape
        w_fp32 = weights.clone().to(torch.float32)
        out_features, in_features = w_fp32.shape
        
        group_size = self.config.group_size or 64
        
        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            w_fp32 = F.pad(w_fp32, (0, pad_size))
            in_features = w_fp32.shape[1]
        
        # Reshape for group quantization
        num_groups = in_features // group_size
        weights_grouped = w_fp32.reshape(out_features, num_groups, group_size)
        
        # Calculate absmax scales per group
        scales = weights_grouped.abs().max(dim=-1).values
        scales = scales.clamp(min=1e-8)
        
        # Normalize weights
        normalized = weights_grouped / scales.unsqueeze(-1)
        
        # Find nearest codebook entry for each weight
        codebook = self.codebook.to(weights.device)
        distances = torch.abs(normalized.unsqueeze(-1) - codebook)
        indices = distances.argmin(dim=-1).to(torch.int8)
        
        # Reshape back
        indices = indices.reshape(out_features, in_features)[:, :original_shape[1]]
        
        # Pack indices
        packed = self._pack_int4(indices)
        
        # Dequantize for error calculation
        dequantized = self.dequantize_weights(indices, scales.reshape(out_features, num_groups), group_size)
        dequantized = dequantized[:, :original_shape[1]]
        
        max_error, mean_error = self._calculate_error(weights, dequantized)
        
        # Memory savings
        original_bytes = weights.numel() * weights.element_size()
        packed_bytes = packed.numel() * packed.element_size()
        scales_bytes = scales.numel() * scales.element_size()
        memory_savings = 100 * (1 - (packed_bytes + scales_bytes) / original_bytes)
        
        return QuantizationResult(
            quantized_weights=packed,
            scales=scales.reshape(out_features, num_groups),
            zero_points=None,
            original_shape=original_shape,
            config=self.config,
            max_error=max_error,
            mean_error=mean_error,
            memory_savings_percent=memory_savings
        )
    
    def _pack_int4(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack two indices into one int8"""
        out_features, in_features = indices.shape
        if in_features % 2 != 0:
            indices = F.pad(indices, (0, 1))
            in_features += 1
        
        reshaped = indices.reshape(out_features, in_features // 2, 2)
        packed = (reshaped[:, :, 0] & 0x0F) | ((reshaped[:, :, 1] & 0x0F) << 4)
        return packed.to(torch.int8)
    
    def _unpack_int4(self, packed: torch.Tensor, original_in_features: int) -> torch.Tensor:
        """Unpack int8 to two indices"""
        out_features = packed.shape[0]
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).reshape(out_features, -1)
        return unpacked[:, :original_in_features]
    
    def dequantize_weights(self, indices: torch.Tensor, scales: torch.Tensor,
                           group_size: Optional[int] = None) -> torch.Tensor:
        """Dequantize NF4 indices back to floating point"""
        codebook = self.codebook.to(indices.device)
        
        # Look up codebook values
        dequantized = codebook[indices.long()]
        
        # Apply scales
        out_features, in_features = indices.shape
        group_size = group_size or self.config.group_size or 64
        num_groups = scales.shape[1]
        
        scales_expanded = scales.unsqueeze(-1).expand(-1, -1, group_size)
        scales_expanded = scales_expanded.reshape(out_features, -1)[:, :in_features]
        
        return dequantized * scales_expanded
    
    def dequantize(self, result: QuantizationResult) -> torch.Tensor:
        """Dequantize from QuantizationResult"""
        unpacked = self._unpack_int4(result.quantized_weights, result.original_shape[1])
        return self.dequantize_weights(
            unpacked,
            result.scales,
            result.config.group_size
        )


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer supporting multiple quantization methods.
    Compatible with W8A16, W4A16, NF4, and GPTQ quantization.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 config: Optional[QuantizationConfig] = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Initialize quantizer based on config
        self.quantizer = self._get_quantizer()
        
        # Buffers for quantized weights
        self.register_buffer("quantized_weights", None)
        self.register_buffer("scales", None)
        self.register_buffer("zero_points", None)
        
        # Bias (kept in full precision)
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None
        
        self._quantized = False
    
    def _get_quantizer(self) -> BaseQuantizer:
        """Get appropriate quantizer based on config"""
        if self.config.method == QuantizationMethod.INT8:
            return INT8Quantizer(self.config)
        elif self.config.method == QuantizationMethod.INT4:
            return INT4Quantizer(self.config)
        elif self.config.method == QuantizationMethod.NF4:
            return NF4Quantizer(self.config)
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.method}")
    
    def quantize_weights(self, weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> QuantizationResult:
        """Quantize weights and store in layer"""
        result = self.quantizer.quantize(weights)
        
        self.quantized_weights = result.quantized_weights
        self.scales = result.scales
        self.zero_points = result.zero_points
        
        if bias is not None:
            self.bias = bias.clone()
        
        self._quantized = True
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization on-the-fly"""
        if not self._quantized:
            raise RuntimeError("Layer has not been quantized. Call quantize_weights first.")
        
        # Dequantize weights
        weights = self.quantizer.dequantize(QuantizationResult(
            quantized_weights=self.quantized_weights,
            scales=self.scales,
            zero_points=self.zero_points,
            original_shape=(self.out_features, self.in_features),
            config=self.config,
            max_error=0, mean_error=0, memory_savings_percent=0
        ))
        
        # Linear operation
        output = F.linear(x, weights.to(x.dtype))
        
        if self.bias is not None:
            output = output + self.bias.to(x.dtype)
        
        return output


def get_quantizer(config: QuantizationConfig) -> BaseQuantizer:
    """Factory function to get appropriate quantizer"""
    if config.method == QuantizationMethod.INT8:
        return INT8Quantizer(config)
    elif config.method == QuantizationMethod.INT4:
        return INT4Quantizer(config)
    elif config.method == QuantizationMethod.NF4:
        return NF4Quantizer(config)
    else:
        raise ValueError(f"Unsupported method: {config.method}")
