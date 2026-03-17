"""
tools/building — Static building-calculation tools for the design_building branch.

Usage
-----
    from tools.building import calculate_aspect_ratio, calculate_total_height
"""

from .tools import (
    CalculateAspectRatioTool,
    CalculateTotalHeightTool,
    CalculateWindowAreaTool,
    ComputeOtherDimensionTool,
    calculate_aspect_ratio,
    calculate_total_height,
    calculate_window_area,
    compute_other_dimension,
)

__all__ = [
    "ComputeOtherDimensionTool",
    "CalculateAspectRatioTool",
    "CalculateTotalHeightTool",
    "CalculateWindowAreaTool",
    "compute_other_dimension",
    "calculate_aspect_ratio",
    "calculate_total_height",
    "calculate_window_area",
]
