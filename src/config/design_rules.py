# Expanded design guidelines with more constraints
DESIGN_GUIDE = [
    {"rule": "Maximum gross floor area is 3000 sqm (GFA = site_area × n_floors)", "type": "gfa", "max": 3000},
    {"rule": "Maximum building width is 20m", "type": "width", "max": 20},
    {"rule": "Maximum building depth is 50m", "type": "depth", "max": 50},
    {"rule": "Minimum floor height is 2.7m", "type": "floor_height", "min": 2.7},
    {"rule": "Maximum floor height is 5m", "type": "floor_height", "max": 5},
    {"rule": "Width to depth ratio should be between 1:1 and 1:3", "type": "ratio", "min": 1/3, "max": 3},
    {"rule": "Building must have emergency exits if GFA > 500 sqm", "type": "emergency_exits", "condition": "gfa > 500"},
]