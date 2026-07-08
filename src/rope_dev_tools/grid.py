"""ROPE's fixed 3D forecast grid dimensions (rope-framework docs/grid.md).

Not configurable per model — every ensemble_fusion_decoder model targets
this same physical grid.
"""

GRID_LST = 72   # Local Solar Time samples, hours [0, 24)
GRID_LAT = 36   # Geodetic latitude samples, degrees [-87.5, 87.5]
GRID_ALT = 45   # Altitude samples, km [100, 980]

LAT_MIN, LAT_MAX = -87.5, 87.5
ALT_MIN_KM, ALT_MAX_KM = 100.0, 980.0
