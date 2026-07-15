"""Reference/default 3D forecast grid dimensions (rope-framework docs/grid.md).
"""

GRID_LST = 72   # Local Solar Time samples, hours [0, 24)
GRID_LAT = 36   # Geodetic latitude samples, degrees [-87.5, 87.5]
GRID_ALT = 45   # Altitude samples, km [100, 980]

LAT_MIN, LAT_MAX = -87.5, 87.5
ALT_MIN_KM, ALT_MAX_KM = 100.0, 980.0

DEFAULT_GRID = {
    "n_lst": GRID_LST, "n_lat": GRID_LAT, "n_alt": GRID_ALT,
    "lat_min_deg": LAT_MIN, "lat_max_deg": LAT_MAX,
    "alt_min_km": ALT_MIN_KM, "alt_max_km": ALT_MAX_KM,
}
