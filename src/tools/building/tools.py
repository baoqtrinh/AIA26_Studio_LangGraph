from tools.base import BaseAgentTool


class ComputeOtherDimensionTool(BaseAgentTool):
    """Compute the other dimension (depth or width) given area and one dimension."""
    name: str = "compute_other_dimension"
    description: str = "Compute the other dimension (depth or width) given area and one dimension."
    categories: list = ["building_design"]

    def _run(self, area: float, known_dimension: float) -> float:  # type: ignore[override]
        return area / known_dimension


class CalculateAspectRatioTool(BaseAgentTool):
    """Calculate the aspect ratio (width:depth)."""
    name: str = "calculate_aspect_ratio"
    description: str = "Calculate the aspect ratio (width:depth)."
    categories: list = ["building_design"]

    def _run(self, width: float, depth: float) -> float:  # type: ignore[override]
        return width / depth


class CalculateTotalHeightTool(BaseAgentTool):
    """Calculate the total height of the building."""
    name: str = "calculate_total_height"
    description: str = "Calculate the total height of the building."
    categories: list = ["building_design"]

    def _run(self, n_floors: int, floor_height: float) -> float:  # type: ignore[override]
        return n_floors * floor_height


class CalculateWindowAreaTool(BaseAgentTool):
    """Calculate required window area based on floor area."""
    name: str = "calculate_window_area"
    description: str = "Calculate required window area based on floor area."
    categories: list = ["building_design"]

    def _run(self, floor_area: float, window_ratio: float) -> float:  # type: ignore[override]
        return floor_area * window_ratio


# Module-level instances
compute_other_dimension = ComputeOtherDimensionTool()
calculate_aspect_ratio  = CalculateAspectRatioTool()
calculate_total_height  = CalculateTotalHeightTool()
calculate_window_area   = CalculateWindowAreaTool()
