"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

# ==============================================================================
# CHART DISPLAY SETTINGS
# ==============================================================================

CHART_WIDTH_DEFAULT = 860
CHART_WIDTH_DASHBOARD = 1100
SMOOTHING_WINDOW_SECONDS = 10

# ==============================================================================
# ACTIVITY CATEGORIES
# ==============================================================================

CATEGORY_ORDER = ["RUN", "TRAIL_RUN", "HIKE", "RIDE", "BACKCOUNTRY_SKI"]
PRIMARY_CATEGORIES = {"RUN", "TRAIL_RUN", "HIKE", "RIDE", "BACKCOUNTRY_SKI"}
TRAINING_LOAD_CATEGORIES = {"RUN", "TRAIL_RUN", "HIKE", "BACKCOUNTRY_SKI"}

CATEGORY_LABELS_FR = {
    "RUN": "Course",
    "TRAIL_RUN": "Trail",
    "HIKE": "Randonnée",
    "RIDE": "Cyclisme",
    "BACKCOUNTRY_SKI": "Ski de rando",
}

# ==============================================================================
# METRIC DISPLAY CONFIG (Analytics)
# ==============================================================================

METRIC_CONFIG = {
    "Time": {
        "planned_col": "plannedTimeSec",
        "category_suffix": "TimeSec",
        "transform": "seconds_to_hours",
        "unit": "heures",
    },
    "Distance": {
        "planned_col": "plannedDistanceKm",
        "category_suffix": "DistanceKm",
        "transform": None,
        "unit": "km",
    },
    "Trimp": {
        "planned_col": "plannedTrimp",
        "category_suffix": "Trimp",
        "transform": None,
        "unit": "TRIMP",
    },
    "DistEq": {
        "planned_col": "plannedDistanceEqKm",
        "category_suffix": "DistanceEqKm",
        "transform": None,
        "unit": "km équivalent",
    },
}

# ==============================================================================
# SESSION TYPES & INTERVAL TARGETS
# ==============================================================================

SESSION_TYPE_LABELS_FR = {
    "FUNDAMENTAL_ENDURANCE": "Endurance fondamentale",
    "LONG_RUN": "Sortie longue",
    "INTERVAL_SIMPLE": "Séance d'intervalles",
    "RACE": "Course",
}

INTERVAL_TARGET_TYPES = [
    "pace",
    "hr",
    "threshold",
    "speed",
    "distance",
    "denivele",
    "sensation",
    "none",
]

# ==============================================================================
# COLOR MAPPINGS
# ==============================================================================

GRADE_COLOR_MAPPING = {
    "grade_lt_neg_0_5": "#001f3f",
    "grade_lt_neg_0_25": "#004d26",
    "grade_lt_neg_0_05": "#22c55e",
    "grade_neutral": "#d1d5db",
    "grade_lt_0_1": "#eab308",
    "grade_lt_0_25": "#f97316",
    "grade_lt_0_5": "#dc2626",
    "grade_ge_0_5": "#000000",
    "unknown": "#808080",
}

PACER_SEGMENT_COLORS = {
    "steep_up": "#dc2626",
    "run_up": "#f97316",
    "flat": "#d1d5db",
    "down": "#22c55e",
    "steep_down": "#004d26",
}

# ==============================================================================
# MAP STYLES
# ==============================================================================

BASE_MAP_STYLES = [
    {
        "label": "Carto clair (défaut)",
        "provider": "carto",
        "style": "light",
    },
    {
        "label": "Carto sombre",
        "provider": "carto",
        "style": "dark",
    },
]

MAPBOX_STYLES = [
    {
        "label": "Mapbox Light",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/light-v11",
    },
    {
        "label": "Mapbox Dark",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/dark-v11",
    },
    {
        "label": "Mapbox Outdoors",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/outdoors-v12",
    },
    {
        "label": "Mapbox Satellite",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/satellite-streets-v12",
    },
]

# ==============================================================================
# ATHLETE DEFAULTS & LOCALE
# ==============================================================================

DEFAULT_HR_REST = 60.0
DEFAULT_HR_MAX = 190.0
DISPLAY_LOCALE = "fr_FR"

# ==============================================================================
# SPEED PROFILE SETTINGS
# ==============================================================================

METRICS = ["Time", "Distance", "Trimp", "DistEq"]

PROFILE_WINDOW_SIZES = [
    5,
    10,
    15,
    20,
    30,
    60,
    120,
    180,
    300,
    600,
    900,
    1200,
    1500,
    1800,
    2100,
    2400,
    2700,
    3000,
    3300,
    3600,
    4500,
    5400,
    7200,
    9000,
    12600,
    14400,
    18000,
    21600,
    23400,
    27000,
    30600,
    34200,
    36000,
    45000,
    54000,
    72000,
    90000,
]

# ==============================================================================
# STRAVA API SETTINGS
# ==============================================================================

STRAVA_API_BASE = "https://www.strava.com/api/v3"
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = f"{STRAVA_API_BASE}/oauth/token"
STRAVA_PROVIDER = "strava"
STRAVA_DEFAULT_SCOPE = "activity:read,activity:read_all"
STRAVA_STREAM_KEYS = "time,distance,altitude,heartrate,cadence,latlng,velocity_smooth"
STRAVA_RATE_LIMIT_15MIN = 100
STRAVA_DAILY_LIMIT = 1000

# ==============================================================================
# SESSION STATE KEYS
# ==============================================================================

STATE_CURRENT_ATHLETE_ID = "current_athlete_id"
STATE_COACH_SETTINGS = "coach_settings"
STATE_OAUTH_STATE = "oauth_state"
STATE_STRAVA_TOKENS_META = "strava_tokens_meta"

# ==============================================================================
# Compatibility aliases
# ==============================================================================

CATEGORY_OPTIONS = CATEGORY_LABELS_FR
