"""
Visualization utilities for weight matrices and quantization analysis.
Generates chart data for frontend consumption.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import base64
import io

# Import matplotlib with non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


@dataclass
class ChartData:
    """Data structure for chart rendering"""
    chart_type: str
    data: Dict[str, Any]
    layout: Dict[str, Any]


class Visualizer:
    """Generate visualization data for weight matrices and quantization analysis"""
    
    def __init__(self, max_display_size: int = 128):
        """
        Args:
            max_display_size: Maximum dimension for heatmap display (downsampled if larger)
        """
        self.max_display_size = max_display_size
    
    def weight_heatmap(self, weights: torch.Tensor, title: str = "Weight Matrix",
                       downsample: bool = True) -> ChartData:
        """
        Generate heatmap data for weight matrix visualization.
        Returns Plotly-compatible data structure.
        """
        w = weights.detach().cpu().float().numpy()
        
        # Downsample if too large
        if downsample and (w.shape[0] > self.max_display_size or w.shape[1] > self.max_display_size):
            w = self._downsample_2d(w, self.max_display_size)
        
        # Calculate symmetric colorscale bounds - convert to Python float for JSON
        vmax = float(max(abs(w.min()), abs(w.max())))
        
        return ChartData(
            chart_type="heatmap",
            data={
                "z": w.tolist(),
                "colorscale": "RdBu_r",
                "zmin": -vmax,
                "zmax": vmax,
                "zmid": 0
            },
            layout={
                "title": title,
                "xaxis": {"title": "Input Features"},
                "yaxis": {"title": "Output Features"}
            }
        )
    
    def weight_histogram(self, weights: torch.Tensor, title: str = "Weight Distribution",
                         bins: int = 50) -> ChartData:
        """Generate histogram data for weight distribution"""
        w = weights.detach().cpu().float().numpy().flatten()
        
        hist, bin_edges = np.histogram(w, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return ChartData(
            chart_type="bar",
            data={
                "x": bin_centers.tolist(),
                "y": hist.tolist(),
                "type": "bar"
            },
            layout={
                "title": title,
                "xaxis": {"title": "Weight Value"},
                "yaxis": {"title": "Frequency"},
                "bargap": 0.05
            }
        )
    
    def error_heatmap(self, original: torch.Tensor, quantized: torch.Tensor,
                      scales: torch.Tensor, title: str = "Quantization Error") -> ChartData:
        """Generate error heatmap between original and dequantized weights"""
        orig = original.detach().cpu().float()
        quant = quantized.detach().cpu().float()
        sc = scales.detach().cpu().float()
        
        # Dequantize
        if sc.dim() == 1:
            dequant = quant * sc.unsqueeze(1)
        else:
            # Group quantization - expand scales
            dequant = quant * sc.unsqueeze(-1)
            dequant = dequant.reshape(orig.shape)
        
        error = (orig - dequant).abs().numpy()
        
        # Downsample if needed
        if error.shape[0] > self.max_display_size or error.shape[1] > self.max_display_size:
            error = self._downsample_2d(error, self.max_display_size)
        
        return ChartData(
            chart_type="heatmap",
            data={
                "z": error.tolist(),
                "colorscale": "Reds",
                "zmin": 0
            },
            layout={
                "title": title,
                "xaxis": {"title": "Input Features"},
                "yaxis": {"title": "Output Features"}
            }
        )
    
    def comparison_overlay(self, original: torch.Tensor, dequantized: torch.Tensor,
                           sample_size: int = 1000) -> ChartData:
        """Generate scatter plot comparing original vs dequantized values"""
        orig = original.detach().cpu().float().numpy().flatten()
        deq = dequantized.detach().cpu().float().numpy().flatten()
        
        # Sample if too large
        if len(orig) > sample_size:
            indices = np.random.choice(len(orig), sample_size, replace=False)
            orig = orig[indices]
            deq = deq[indices]
        
        return ChartData(
            chart_type="scatter",
            data={
                "x": orig.tolist(),
                "y": deq.tolist(),
                "mode": "markers",
                "marker": {"size": 3, "opacity": 0.5}
            },
            layout={
                "title": "Original vs Dequantized Weights",
                "xaxis": {"title": "Original Value"},
                "yaxis": {"title": "Dequantized Value"},
                "shapes": [{
                    "type": "line",
                    "x0": float(orig.min()),
                    "x1": float(orig.max()),
                    "y0": float(orig.min()),
                    "y1": float(orig.max()),
                    "line": {"color": "red", "dash": "dash"}
                }]
            }
        )
    
    def scales_histogram(self, scales: torch.Tensor, 
                         title: str = "Quantization Scales Distribution") -> ChartData:
        """Generate histogram of quantization scales"""
        s = scales.detach().cpu().float().numpy().flatten()
        
        hist, bin_edges = np.histogram(s, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return ChartData(
            chart_type="bar",
            data={
                "x": bin_centers.tolist(),
                "y": hist.tolist(),
                "marker": {"color": "green"}
            },
            layout={
                "title": title,
                "xaxis": {"title": "Scale Value"},
                "yaxis": {"title": "Frequency"}
            }
        )
    
    def layer_error_bar(self, layer_errors: Dict[str, float],
                        title: str = "Quantization Error by Layer") -> ChartData:
        """Generate bar chart of errors per layer"""
        layers = list(layer_errors.keys())
        errors = list(layer_errors.values())
        
        return ChartData(
            chart_type="bar",
            data={
                "x": layers,
                "y": errors,
                "marker": {"color": "coral"}
            },
            layout={
                "title": title,
                "xaxis": {"title": "Layer", "tickangle": 45},
                "yaxis": {"title": "Mean Absolute Error"}
            }
        )
    
    def memory_comparison(self, original_mb: float, quantized_mb: float,
                          overhead_mb: float = 0) -> ChartData:
        """Generate memory comparison chart"""
        return ChartData(
            chart_type="bar",
            data={
                "x": ["Original (FP32)", "Quantized + Scales", "Savings"],
                "y": [original_mb, quantized_mb + overhead_mb, original_mb - quantized_mb - overhead_mb],
                "marker": {"color": ["#3498db", "#2ecc71", "#e74c3c"]}
            },
            layout={
                "title": "Memory Usage Comparison",
                "yaxis": {"title": "Memory (MB)"}
            }
        )
    
    def _downsample_2d(self, arr: np.ndarray, max_size: int) -> np.ndarray:
        """Downsample 2D array to max_size x max_size"""
        h, w = arr.shape
        
        if h > max_size:
            step_h = h // max_size
            arr = arr[::step_h, :][:max_size, :]
        
        if w > max_size:
            step_w = w // max_size
            arr = arr[:, ::step_w][:, :max_size]
        
        return arr
    
    def generate_png(self, weights: torch.Tensor, title: str = "Weights") -> bytes:
        """Generate PNG image bytes (for backward compatibility)"""
        w = weights.detach().cpu().float().numpy()
        
        if w.shape[0] > self.max_display_size or w.shape[1] > self.max_display_size:
            w = self._downsample_2d(w, self.max_display_size)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        vmax = max(abs(w.min()), abs(w.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im = ax.imshow(w, cmap='RdBu_r', norm=norm)
        plt.colorbar(im, label='Weight Value')
        ax.set_title(title)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
    
    def to_dict(self, chart: ChartData) -> Dict[str, Any]:
        """Convert ChartData to dictionary"""
        return {
            "type": chart.chart_type,
            "data": chart.data,
            "layout": chart.layout
        }


# Global instance
visualizer = Visualizer()


def get_weight_heatmap(weights: torch.Tensor, title: str = "Weights") -> Dict[str, Any]:
    """Generate weight heatmap data"""
    return visualizer.to_dict(visualizer.weight_heatmap(weights, title))


def get_weight_histogram(weights: torch.Tensor, title: str = "Distribution") -> Dict[str, Any]:
    """Generate weight histogram data"""
    return visualizer.to_dict(visualizer.weight_histogram(weights, title))
