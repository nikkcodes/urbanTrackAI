"""Configuration constants and units for UrbanTrackAI mobility engine."""

from typing import Final

# Units documentation:
# - Distance: Kilometers (km)
# - Speed: Kilometers per hour (km/h)
# - Time: Minutes (min)
# - Capacity / Flow: Vehicles per hour (vph)

# Default routing parameters
DEFAULT_MAX_CANDIDATE_ROUTES: Final[int] = 5
DEFAULT_ROUTE_CUTOFF_HOPS: Final[int] = 20

# Physical validation thresholds
MIN_DISTANCE_KM: Final[float] = 0.001      # 1 meter
MAX_DISTANCE_KM: Final[float] = 500.0      # 500 km (within metropolitan bounds)

MIN_SPEED_KMPH: Final[float] = 1.0         # 1 km/h (avoids division by zero)
MAX_SPEED_KMPH: Final[float] = 250.0       # 250 km/h (realistic road speed ceiling)

MIN_CAPACITY_VPH: Final[float] = 1.0       # 1 vehicle per hour minimum
MAX_CAPACITY_VPH: Final[float] = 20000.0   # Multilane highway ceiling
