# Expanded design guidelines with more constraints
DESIGN_GUIDE = [
    {"rule": "Maximum area is 3000 sqm", "type": "area", "max": 3000},
    {"rule": "Maximum building height is 18m", "type": "height", "max": 18},
    {"rule": "Maximum building width is 20m", "type": "width", "max": 20},
    {"rule": "Maximum building depth is 50m", "type": "depth", "max": 50},
    {"rule": "Maximum number of floors is 4", "type": "n_floors", "max": 4},
    {"rule": "Minimum floor height is 2.7m", "type": "floor_height", "min": 2.7},
    {"rule": "Maximum floor height is 5m", "type": "floor_height", "max": 5},
    {"rule": "Width to depth ratio should be between 1:1 and 1:3", "type": "ratio", "min": 1/3, "max": 1},
    {"rule": "Minimum window area is 10% of floor area", "type": "window_area_ratio", "min": 0.1},
    {"rule": "Building must have emergency exits if floor area > 500 sqm", "type": "emergency_exits", "condition": "area > 500"},
]