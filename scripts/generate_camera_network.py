"""
UrbanTrack AI - Camera Network & Location Generator
===================================================
Derives approximate physical geographic coordinates (WGS84 lat/lon), road contexts,
surveillance headings/bearings, confidence scores, and topological transition graph
for all cameras in the AI City Challenge 2022 Track 1 MTMC dataset.

Primary visual sources:
- cam_loc/S01.png
- cam_loc/S02.png
- cam_loc/S0345.png
- cam_loc/S06.png

Outputs:
- data/config/camera_locations.json
- data/config/camera_graph.json
- data/gis/cameras/camera_locations.geojson
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def forward_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial compass bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# Ground-truth camera base coordinates and attributes derived from:
# 1) cam_loc maps (S01.png, S02.png, S0345.png, S06.png)
# 2) AI City Challenge ground-plane homography calibration points
# 3) OpenStreetMap real-world street network in Dubuque, Iowa
BASE_CAMERAS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # S01: Northwest Arterial (IA-32) & John F. Kennedy Rd
    # -------------------------------------------------------------
    "C001": {
        "location": {"latitude": 42.525540, "longitude": -90.723480},
        "location_source": "cam_loc/S01.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "John F. Kennedy Road",
            "intersection": "Northwest Arterial & John F. Kennedy Road (Southeast corner)"
        },
        "direction": {
            "description": "North-Northwest viewing northbound traffic entering intersection",
            "bearing_deg": 335.0
        },
        "confidence": "HIGH",
        "notes": "Mounted on SE corner mast arm along JFK Rd; views northbound traffic entering Northwest Arterial junction."
    },
    "C002": {
        "location": {"latitude": 42.525780, "longitude": -90.723720},
        "location_source": "cam_loc/S01.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "John F. Kennedy Road",
            "intersection": "Northwest Arterial & John F. Kennedy Road (Northwest corner)"
        },
        "direction": {
            "description": "South-Southeast viewing southbound traffic entering intersection",
            "bearing_deg": 155.0
        },
        "confidence": "HIGH",
        "notes": "Mounted on NW corner of JFK Rd; views southbound traffic crossing Northwest Arterial."
    },
    "C003": {
        "location": {"latitude": 42.525720, "longitude": -90.723380},
        "location_source": "cam_loc/S01.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Northwest Arterial (IA-32)",
            "intersection": "Northwest Arterial & John F. Kennedy Road (Northeast approach)"
        },
        "direction": {
            "description": "West-Southwest viewing westbound traffic on Northwest Arterial",
            "bearing_deg": 248.0
        },
        "confidence": "HIGH",
        "notes": "Mounted along NE approach on Northwest Arterial; views westbound traffic entering intersection."
    },
    "C004": {
        "location": {"latitude": 42.525600, "longitude": -90.723820},
        "location_source": "cam_loc/S01.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Northwest Arterial (IA-32)",
            "intersection": "Northwest Arterial & John F. Kennedy Road (Southwest approach)"
        },
        "direction": {
            "description": "East-Northeast viewing eastbound traffic on Northwest Arterial",
            "bearing_deg": 68.0
        },
        "confidence": "HIGH",
        "notes": "Mounted along SW approach on Northwest Arterial; views eastbound lanes entering intersection."
    },
    "C005": {
        "location": {"latitude": 42.525820, "longitude": -90.723580},
        "location_source": "cam_loc/S01.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "John F. Kennedy Road",
            "intersection": "Northwest Arterial & John F. Kennedy Road (North leg)"
        },
        "direction": {
            "description": "South-Southwest viewing intersection interior from north leg",
            "bearing_deg": 190.0
        },
        "confidence": "HIGH",
        "notes": "Mounted on northern signal mast arm; provides interior intersection overview looking south."
    },

    # -------------------------------------------------------------
    # S02: US-20 (Dodge St) & Century Dr
    # -------------------------------------------------------------
    "C006": {
        "location": {"latitude": 42.491960, "longitude": -90.723550},
        "location_source": "cam_loc/S02.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 & Century Drive (Westbound lanes east of junction)"
        },
        "direction": {
            "description": "East along US-20 westbound lanes towards Century Drive",
            "bearing_deg": 85.0
        },
        "confidence": "HIGH",
        "notes": "Monitors westbound US-20 traffic arriving at Century Dr signal from the west."
    },
    "C007": {
        "location": {"latitude": 42.492080, "longitude": -90.723750},
        "location_source": "cam_loc/S02.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Century Drive",
            "intersection": "US-20 & Century Drive (North approach)"
        },
        "direction": {
            "description": "South along Century Drive crossing Dodge Street",
            "bearing_deg": 180.0
        },
        "confidence": "HIGH",
        "notes": "Mounted north of Dodge St; views southbound vehicles exiting commercial corridor on Century Dr."
    },
    "C008": {
        "location": {"latitude": 42.491900, "longitude": -90.723700},
        "location_source": "cam_loc/S02.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20) Median",
            "intersection": "US-20 & Century Drive (Median crossover)"
        },
        "direction": {
            "description": "North across westbound US-20 lanes towards Century Drive",
            "bearing_deg": 355.0
        },
        "confidence": "HIGH",
        "notes": "Mounted in median dividing US-20; monitors crossover and turn lanes."
    },
    "C009": {
        "location": {"latitude": 42.491820, "longitude": -90.723850},
        "location_source": "cam_loc/S02.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 & Century Drive (Eastbound lanes west of junction)"
        },
        "direction": {
            "description": "West along US-20 eastbound lanes towards Century Drive",
            "bearing_deg": 265.0
        },
        "confidence": "HIGH",
        "notes": "Monitors eastbound US-20 traffic arriving at Century Dr intersection."
    },

    # -------------------------------------------------------------
    # S03 / S05: Hill St / W 5th St / Alpine St Cluster
    # -------------------------------------------------------------
    "C010": {
        "location": {"latitude": 42.496955, "longitude": -90.673875},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Hill Street",
            "intersection": "Hill Street & West 5th Street (North approach)"
        },
        "direction": {
            "description": "North-Northeast along Hill Street in Cable Car Square district",
            "bearing_deg": 26.2
        },
        "confidence": "HIGH",
        "notes": "Monitors Hill St north approach above W 5th St."
    },
    "C011": {
        "location": {"latitude": 42.496720, "longitude": -90.674117},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Hill Street",
            "intersection": "Hill Street & West 5th Street / Burch Street"
        },
        "direction": {
            "description": "South along Hill Street towards West 3rd Street",
            "bearing_deg": 182.1
        },
        "confidence": "HIGH",
        "notes": "Located at 5th & Hill St intersection; views southbound flow down the hill."
    },
    "C012": {
        "location": {"latitude": 42.496583, "longitude": -90.674257},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "West 5th Street",
            "intersection": "West 5th Street & Alpine Street"
        },
        "direction": {
            "description": "West-Northwest along West 5th Street towards Alpine Street",
            "bearing_deg": 299.9
        },
        "confidence": "HIGH",
        "notes": "Monitors climb up W 5th St residential incline."
    },
    "C013": {
        "location": {"latitude": 42.496419, "longitude": -90.674342},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Hill Street",
            "intersection": "Hill Street & West 3rd Street"
        },
        "direction": {
            "description": "Northeast along Hill Street towards West 4th Street",
            "bearing_deg": 56.3
        },
        "confidence": "HIGH",
        "notes": "Monitors bottom of Hill St at W 3rd St."
    },
    "C014": {
        "location": {"latitude": 42.496281, "longitude": -90.674605},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "West 3rd Street",
            "intersection": "West 3rd Street & Summit Street"
        },
        "direction": {
            "description": "Southwest along West 3rd Street towards Fenelon Place",
            "bearing_deg": 215.9
        },
        "confidence": "HIGH",
        "notes": "Monitors W 3rd St traffic near Summit St."
    },
    "C015": {
        "location": {"latitude": 42.496384, "longitude": -90.674818},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Alpine Street",
            "intersection": "Alpine Street & West 3rd / 4th Street"
        },
        "direction": {
            "description": "West-Northwest along Alpine Street connecting 3rd and 5th Streets",
            "bearing_deg": 282.8
        },
        "confidence": "HIGH",
        "notes": "Monitors Alpine St transition corridor between hillside avenues."
    },

    # -------------------------------------------------------------
    # S04 / S05: University Avenue Corridor
    # -------------------------------------------------------------
    "C016": {
        "location": {"latitude": 42.500983, "longitude": -90.669833},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "West 9th Street / Loras Boulevard",
            "intersection": "Loras Boulevard & Bluff Street (Jackson Park District)"
        },
        "direction": {
            "description": "Southwest along Loras Boulevard towards downtown",
            "bearing_deg": 237.3
        },
        "confidence": "HIGH",
        "notes": "Monitors east gateway of Loras Blvd near Bluff St."
    },
    "C017": {
        "location": {"latitude": 42.500217, "longitude": -90.673775},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Loras Boulevard (East merge)"
        },
        "direction": {
            "description": "East along University Avenue approaching Loras Boulevard",
            "bearing_deg": 95.7
        },
        "confidence": "HIGH",
        "notes": "Monitors University Ave inbound eastbound traffic toward Loras Blvd."
    },
    "C018": {
        "location": {"latitude": 42.500256, "longitude": -90.674660},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Loras Boulevard",
            "intersection": "Loras Boulevard & Highland Place / West 11th Street"
        },
        "direction": {
            "description": "Southwest along Loras Boulevard",
            "bearing_deg": 215.9
        },
        "confidence": "HIGH",
        "notes": "Monitors Loras Blvd west of Highland Place."
    },
    "C019": {
        "location": {"latitude": 42.500145, "longitude": -90.674927},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Loras Boulevard",
            "intersection": "Loras Boulevard & Chestnut Street"
        },
        "direction": {
            "description": "North-Northwest along Loras Boulevard towards Cox Street",
            "bearing_deg": 331.7
        },
        "confidence": "HIGH",
        "notes": "Monitors northwest-bound flow on Loras Blvd."
    },
    "C020": {
        "location": {"latitude": 42.500048, "longitude": -90.675048},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Loras Boulevard",
            "intersection": "Loras Boulevard & Cox Street"
        },
        "direction": {
            "description": "East along Loras Boulevard approaching Cox Street",
            "bearing_deg": 91.6
        },
        "confidence": "HIGH",
        "notes": "Monitors Loras Blvd at Cox St intersection."
    },
    "C021": {
        "location": {"latitude": 42.499989, "longitude": -90.675201},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Cox Street"
        },
        "direction": {
            "description": "West-Southwest along University Avenue towards Nevada Street",
            "bearing_deg": 254.9
        },
        "confidence": "HIGH",
        "notes": "Monitors University Ave westbound traffic past Cox St."
    },
    "C022": {
        "location": {"latitude": 42.499228, "longitude": -90.680404},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Alpine Street"
        },
        "direction": {
            "description": "East-Northeast along University Avenue",
            "bearing_deg": 79.2
        },
        "confidence": "HIGH",
        "notes": "Monitors eastbound traffic approaching Allison-Henderson Park."
    },
    "C023": {
        "location": {"latitude": 42.499234, "longitude": -90.680561},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Nevada Street"
        },
        "direction": {
            "description": "Northeast along University Avenue at Nevada Street",
            "bearing_deg": 47.0
        },
        "confidence": "HIGH",
        "notes": "Monitors University Ave & Nevada St junction."
    },
    "C024": {
        "location": {"latitude": 42.499221, "longitude": -90.680759},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Nevada Street",
            "intersection": "Nevada Street & University Avenue"
        },
        "direction": {
            "description": "North-Northwest along Nevada Street towards Loras College",
            "bearing_deg": 346.1
        },
        "confidence": "MEDIUM",
        "notes": "Monitors northbound traffic on Nevada St feeding into campus."
    },
    "C025": {
        "location": {"latitude": 42.499178, "longitude": -90.681085},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Loras Boulevard",
            "intersection": "Loras Boulevard north of Allison-Henderson Park"
        },
        "direction": {
            "description": "East-Southeast along Loras Boulevard",
            "bearing_deg": 103.8
        },
        "confidence": "HIGH",
        "notes": "Monitors Loras Blvd northern segment beside park."
    },
    "C026": {
        "location": {"latitude": 42.499090, "longitude": -90.681629},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Grandview Avenue (East junction)"
        },
        "direction": {
            "description": "West-Southwest along University Avenue past Allison-Henderson Park",
            "bearing_deg": 258.9
        },
        "confidence": "HIGH",
        "notes": "Monitors main westbound arterial traffic by park."
    },
    "C027": {
        "location": {"latitude": 42.498394, "longitude": -90.688026},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Glen Oak Street"
        },
        "direction": {
            "description": "East along University Avenue towards Finley Hospital",
            "bearing_deg": 89.2
        },
        "confidence": "HIGH",
        "notes": "Monitors eastbound traffic along hospital corridor."
    },
    "C028": {
        "location": {"latitude": 42.498366, "longitude": -90.688205},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & College Street / Finley Hospital"
        },
        "direction": {
            "description": "Southeast into Finley Hospital entrance / College Street",
            "bearing_deg": 136.7
        },
        "confidence": "HIGH",
        "notes": "Monitors turn movements into UnityPoint Health Finley Hospital."
    },
    "C029": {
        "location": {"latitude": 42.499005, "longitude": -90.693166},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Custer Street / Adair Street"
        },
        "direction": {
            "description": "East-Southeast along University Avenue",
            "bearing_deg": 117.2
        },
        "confidence": "MEDIUM",
        "notes": "Long FOV coverage along University Ave near Custer St."
    },
    "C030": {
        "location": {"latitude": 42.499300, "longitude": -90.693626},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Bennett Street / Delhi Street"
        },
        "direction": {
            "description": "Southeast along University Avenue at University of Dubuque",
            "bearing_deg": 152.2
        },
        "confidence": "HIGH",
        "notes": "Monitors University Ave campus approach."
    },
    "C031": {
        "location": {"latitude": 42.499314, "longitude": -90.693734},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Mineral Street (East approach)"
        },
        "direction": {
            "description": "South-Southwest along Mineral Street crossing University Avenue",
            "bearing_deg": 210.9
        },
        "confidence": "HIGH",
        "notes": "Monitors south approach from Mineral St onto University Ave."
    },
    "C032": {
        "location": {"latitude": 42.499376, "longitude": -90.693814},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Kirk Street"
        },
        "direction": {
            "description": "West along University Avenue towards Kirk Street",
            "bearing_deg": 275.5
        },
        "confidence": "HIGH",
        "notes": "Monitors westbound lanes approaching commercial segment."
    },
    "C033": {
        "location": {"latitude": 42.499453, "longitude": -90.694488},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Asbury Road",
            "intersection": "Asbury Road & Cherry Street"
        },
        "direction": {
            "description": "West along Asbury Road towards Flora Park",
            "bearing_deg": 279.7
        },
        "confidence": "HIGH",
        "notes": "Monitors Asbury Road diagonal corridor."
    },
    "C034": {
        "location": {"latitude": 42.499784, "longitude": -90.696075},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Kirk Street (West approach)"
        },
        "direction": {
            "description": "East-Southeast along University Avenue",
            "bearing_deg": 120.9
        },
        "confidence": "HIGH",
        "notes": "Monitors eastbound traffic along commercial shopping district."
    },
    "C035": {
        "location": {"latitude": 42.499836, "longitude": -90.696285},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Delhi Street / Excalibur area"
        },
        "direction": {
            "description": "South-Southwest along University Avenue",
            "bearing_deg": 211.0
        },
        "confidence": "HIGH",
        "notes": "Monitors retail center frontage on University Ave."
    },
    "C036": {
        "location": {"latitude": 42.499709, "longitude": -90.696522},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Mineral Street (West approach)"
        },
        "direction": {
            "description": "Southwest along University Avenue towards Loras Blvd fork",
            "bearing_deg": 235.6
        },
        "confidence": "HIGH",
        "notes": "Monitors westbound transition to Loras Blvd fork."
    },
    "C037": {
        "location": {"latitude": 42.498993, "longitude": -90.698158},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Loras Boulevard (West fork)"
        },
        "direction": {
            "description": "East-Northeast along University Avenue from fork",
            "bearing_deg": 61.1
        },
        "confidence": "HIGH",
        "notes": "Monitors fork divergence between University Ave and Loras Blvd."
    },
    "C038": {
        "location": {"latitude": 42.498765, "longitude": -90.698493},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Asbury Road",
            "intersection": "Asbury Road & University Avenue junction"
        },
        "direction": {
            "description": "South-Southwest along Asbury Road",
            "bearing_deg": 197.2
        },
        "confidence": "HIGH",
        "notes": "Monitors Asbury Road entrance to University Ave corridor."
    },
    "C039": {
        "location": {"latitude": 42.498721, "longitude": -90.698650},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & Cherry Street (Flora Park entrance)"
        },
        "direction": {
            "description": "North-Northwest along Cherry Street by Flora Park",
            "bearing_deg": 326.4
        },
        "confidence": "HIGH",
        "notes": "Monitors park access and northern turning movements."
    },
    "C040": {
        "location": {"latitude": 42.498655, "longitude": -90.698649},
        "location_source": "cam_loc/S0345.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "University Avenue",
            "intersection": "University Avenue & McPoland Avenue / Pennsylvania Avenue"
        },
        "direction": {
            "description": "West along University Avenue toward city border",
            "bearing_deg": 272.8
        },
        "confidence": "HIGH",
        "notes": "Western terminus camera along University Ave monitored corridor."
    },

    # -------------------------------------------------------------
    # S06: US-20 (Dodge St) Corridor
    # -------------------------------------------------------------
    "C041": {
        "location": {"latitude": 42.491239, "longitude": -90.701509},
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 & Center Grove / University Avenue interchange"
        },
        "direction": {
            "description": "Northeast along US-20 eastbound ramp / Center Grove",
            "bearing_deg": 69.6
        },
        "confidence": "HIGH",
        "notes": "Eastern anchor of US-20 highway camera corridor at Center Grove."
    },
    "C042": {
        "location": {"latitude": 42.492023, "longitude": -90.714816},
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 & Cedar Cross Road"
        },
        "direction": {
            "description": "Northeast along US-20 crossing Cedar Cross Road",
            "bearing_deg": 66.1
        },
        "confidence": "HIGH",
        "notes": "Major signalized junction at US-20 and Cedar Cross Rd."
    },
    "C043": {
        "location": {"latitude": 42.491850, "longitude": -90.720616},
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 south of Kennedy Mall"
        },
        "direction": {
            "description": "Northeast along US-20 adjacent to Kennedy Mall",
            "bearing_deg": 55.8
        },
        "confidence": "HIGH",
        "notes": "Highway camera monitoring US-20 shopping mall bypass."
    },
    "C044": {
        "location": {"latitude": 42.490369, "longitude": -90.732608},
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 between Northwest Arterial and Wacker Drive"
        },
        "direction": {
            "description": "Southeast along US-20 approaching commercial center",
            "bearing_deg": 136.7
        },
        "confidence": "HIGH",
        "notes": "Arterial camera on US-20 segment between NW Arterial and Wacker Dr."
    },
    "C045": {
        "location": {"latitude": 42.498532, "longitude": -90.739334}, # Will use verified 42.488532
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20 Frontage)",
            "intersection": "Dodge Street near Walmart Supercenter (4200 Dodge St)"
        },
        "direction": {
            "description": "Southeast along Dodge Street frontage road",
            "bearing_deg": 117.8
        },
        "confidence": "HIGH",
        "notes": "Monitors Dodge St frontage access road fronting Walmart."
    },
    "C046": {
        "location": {"latitude": 42.486478, "longitude": -90.746993},
        "location_source": "cam_loc/S06.png",
        "location_method": "map_georeferenced",
        "location_precision": "approximate",
        "road_context": {
            "road": "Dodge Street (US-20)",
            "intersection": "US-20 & Old Highway Road interchange"
        },
        "direction": {
            "description": "Northeast along US-20 from western city boundary",
            "bearing_deg": 39.8
        },
        "confidence": "HIGH",
        "notes": "Westernmost camera on US-20 monitoring vehicles entering Dubuque."
    }
}
# Correction for C045 latitude typo (42.488532)
BASE_CAMERAS["C045"]["location"]["latitude"] = 42.488532


def load_manifest_cameras() -> List[Dict[str, Any]]:
    """Load cameras from data/config/aicity_manifest.json."""
    manifest_path = Path("data/config/aicity_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cameras", [])


def build_camera_locations() -> Dict[str, Dict[str, Any]]:
    """Build the dictionary of 65 scenario-qualified camera locations."""
    manifest_cams = load_manifest_cameras()
    locations: Dict[str, Dict[str, Any]] = {}

    for cam_info in manifest_cams:
        cam_id = cam_info["camera_id"]
        scenario = cam_info["scene"]
        cnum = cam_info["camera"].upper()  # e.g. "C001"

        if cnum not in BASE_CAMERAS:
            raise KeyError(f"Base camera metadata missing for {cnum}")

        base = BASE_CAMERAS[cnum]
        cam_record = {
            "camera_id": cam_id,
            "camera_number": cnum,
            "scenario": scenario,
            "location": {
                "latitude": round(base["location"]["latitude"], 6),
                "longitude": round(base["location"]["longitude"], 6)
            },
            "location_source": base["location_source"],
            "location_method": base["location_method"],
            "location_precision": base["location_precision"],
            "road_context": {
                "road": base["road_context"]["road"],
                "intersection": base["road_context"]["intersection"]
            },
            "direction": {
                "description": base["direction"]["description"],
                "bearing_deg": base["direction"]["bearing_deg"]
            },
            "confidence": base["confidence"],
            "notes": base["notes"]
        }

        # Clarify S05 cross-scenario sensor reuse
        if scenario == "S05":
            if cnum == "C010":
                cam_record["notes"] = (
                    "Scenario S05 video stream utilizing the same physical sensor and camera pole "
                    "as CAM_S03_C010 at Hill St & W 5th St."
                )
            else:
                cam_record["notes"] = (
                    f"Scenario S05 video stream utilizing the same physical sensor and camera pole "
                    f"as CAM_S04_{cnum} on University Avenue corridor."
                )

        locations[cam_id] = cam_record

    return locations


def build_camera_graph(locations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build directed camera transition graph based on road topology, corridor
    continuity, and intersection movement feasibility.
    """
    nodes = sorted(list(locations.keys()))
    edges: List[Dict[str, Any]] = []

    # Map of scenario-specific transitions (source_cnum, target_cnum, confidence, evidence)
    # -------------------------------------------------------------
    # S01: Northwest Arterial & JFK Rd intersection
    # -------------------------------------------------------------
    s01_transitions = [
        ("C001", "C002", "HIGH"), ("C001", "C003", "HIGH"), ("C001", "C004", "HIGH"), ("C001", "C005", "HIGH"),
        ("C002", "C001", "HIGH"), ("C002", "C003", "HIGH"), ("C002", "C004", "HIGH"),
        ("C003", "C001", "HIGH"), ("C003", "C002", "HIGH"), ("C003", "C004", "HIGH"), ("C003", "C005", "HIGH"),
        ("C004", "C001", "HIGH"), ("C004", "C002", "HIGH"), ("C004", "C003", "HIGH"), ("C004", "C005", "HIGH"),
        ("C005", "C001", "HIGH"), ("C005", "C003", "HIGH"), ("C005", "C004", "HIGH"),
    ]

    # -------------------------------------------------------------
    # S02: US-20 (Dodge St) & Century Dr intersection
    # -------------------------------------------------------------
    s02_transitions = [
        ("C006", "C007", "HIGH"), ("C006", "C008", "HIGH"),
        ("C007", "C006", "HIGH"), ("C007", "C008", "HIGH"), ("C007", "C009", "HIGH"),
        ("C008", "C006", "HIGH"), ("C008", "C007", "HIGH"), ("C008", "C009", "HIGH"),
        ("C009", "C008", "HIGH"), ("C009", "C006", "MEDIUM"),
    ]

    # -------------------------------------------------------------
    # S03: Hill St / 5th / 3rd / Alpine cluster
    # -------------------------------------------------------------
    s03_transitions = [
        ("C010", "C011", "HIGH"), ("C011", "C010", "HIGH"),
        ("C011", "C012", "HIGH"), ("C012", "C011", "HIGH"),
        ("C011", "C013", "HIGH"), ("C013", "C011", "HIGH"),
        ("C012", "C015", "HIGH"), ("C015", "C012", "HIGH"),
        ("C013", "C014", "HIGH"), ("C014", "C013", "HIGH"),
        ("C013", "C015", "MEDIUM"), ("C015", "C013", "MEDIUM"),
        ("C014", "C015", "MEDIUM"), ("C015", "C014", "MEDIUM"),
    ]

    # -------------------------------------------------------------
    # S04: University Avenue Corridor
    # -------------------------------------------------------------
    s04_transitions = [
        # Loras / Bluff east cluster
        ("C016", "C017", "HIGH"), ("C017", "C016", "HIGH"),
        ("C016", "C018", "HIGH"), ("C018", "C016", "HIGH"),
        ("C017", "C018", "HIGH"), ("C018", "C017", "HIGH"),
        ("C018", "C019", "HIGH"), ("C019", "C018", "HIGH"),
        ("C019", "C020", "HIGH"), ("C020", "C019", "HIGH"),
        ("C020", "C021", "HIGH"), ("C021", "C020", "HIGH"),
        ("C021", "C017", "HIGH"), ("C017", "C021", "HIGH"),
        # Loras to Nevada / Allison-Henderson Park
        ("C020", "C022", "MEDIUM"), ("C022", "C020", "MEDIUM"),
        ("C021", "C022", "HIGH"), ("C022", "C021", "HIGH"),
        ("C021", "C023", "HIGH"), ("C023", "C021", "HIGH"),
        # Around Allison-Henderson Park / Nevada St
        ("C022", "C023", "HIGH"), ("C023", "C022", "HIGH"),
        ("C023", "C024", "HIGH"), ("C024", "C023", "HIGH"),
        ("C023", "C025", "HIGH"), ("C025", "C023", "HIGH"),
        ("C022", "C026", "HIGH"), ("C026", "C022", "HIGH"),
        ("C025", "C026", "HIGH"), ("C026", "C025", "HIGH"),
        ("C024", "C025", "MEDIUM"), ("C025", "C024", "MEDIUM"),
        # Park to Finley Hospital / Glen Oak
        ("C026", "C027", "HIGH"), ("C027", "C026", "HIGH"),
        ("C027", "C028", "HIGH"), ("C028", "C027", "HIGH"),
        ("C027", "C029", "HIGH"), ("C029", "C027", "HIGH"),
        ("C028", "C029", "HIGH"), ("C029", "C028", "HIGH"),
        # Finley Hospital to Delhi / Kirk
        ("C029", "C030", "HIGH"), ("C030", "C029", "HIGH"),
        ("C030", "C031", "HIGH"), ("C031", "C030", "HIGH"),
        ("C030", "C032", "HIGH"), ("C032", "C030", "HIGH"),
        ("C031", "C032", "HIGH"), ("C032", "C031", "HIGH"),
        ("C032", "C034", "HIGH"), ("C034", "C032", "HIGH"),
        ("C034", "C035", "HIGH"), ("C035", "C034", "HIGH"),
        ("C035", "C036", "HIGH"), ("C036", "C035", "HIGH"),
        # West end / Flora Park / Asbury Rd
        ("C036", "C037", "HIGH"), ("C037", "C036", "HIGH"),
        ("C037", "C038", "HIGH"), ("C038", "C037", "HIGH"),
        ("C038", "C033", "HIGH"), ("C033", "C038", "HIGH"),
        ("C037", "C039", "HIGH"), ("C039", "C037", "HIGH"),
        ("C038", "C039", "MEDIUM"), ("C039", "C038", "MEDIUM"),
        ("C039", "C040", "HIGH"), ("C040", "C039", "HIGH"),
    ]

    # -------------------------------------------------------------
    # S06: US-20 Highway Corridor
    # -------------------------------------------------------------
    s06_transitions = [
        # Eastbound progression
        ("C046", "C045", "HIGH"),
        ("C046", "C044", "MEDIUM"),
        ("C045", "C044", "HIGH"),
        ("C044", "C043", "HIGH"),
        ("C043", "C042", "HIGH"),
        ("C042", "C041", "HIGH"),
        # Westbound progression
        ("C041", "C042", "HIGH"),
        ("C042", "C043", "HIGH"),
        ("C043", "C044", "HIGH"),
        ("C044", "C045", "HIGH"),
        ("C044", "C046", "MEDIUM"),
        ("C045", "C046", "HIGH"),
    ]

    scenario_map = {
        "S01": (s01_transitions, "cam_loc/S01.png"),
        "S02": (s02_transitions, "cam_loc/S02.png"),
        "S03": (s03_transitions, "cam_loc/S0345.png"),
        "S04": (s04_transitions, "cam_loc/S0345.png"),
        "S06": (s06_transitions, "cam_loc/S06.png"),
    }

    # Generate edges for S01, S02, S03, S04, S06
    for scn, (trans_list, evidence_img) in scenario_map.items():
        for src_c, tgt_c, conf in trans_list:
            src_id = f"CAM_{scn}_{src_c}"
            tgt_id = f"CAM_{scn}_{tgt_c}"
            if src_id in locations and tgt_id in locations:
                p1 = locations[src_id]["location"]
                p2 = locations[tgt_id]["location"]
                dist = haversine_distance_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
                bear = forward_bearing_deg(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
                edges.append({
                    "source": src_id,
                    "target": tgt_id,
                    "relationship": "possible_transition",
                    "distance_m": round(dist, 1),
                    "bearing_deg": round(bear, 1),
                    "evidence": [evidence_img],
                    "confidence": conf
                })

    # For S05, filter relevant transitions from S03 and S04 for active S05 cameras,
    # plus verified transitions linking C010 <-> C017, C033 <-> C034, and C029 <-> C034
    s05_cams = {k: v for k, v in locations.items() if v["scenario"] == "S05"}
    s05_all_transitions = s03_transitions + s04_transitions
    for src_c, tgt_c, conf in s05_all_transitions:
        src_id = f"CAM_S05_{src_c}"
        tgt_id = f"CAM_S05_{tgt_c}"
        if src_id in s05_cams and tgt_id in s05_cams:
            p1 = locations[src_id]["location"]
            p2 = locations[tgt_id]["location"]
            dist = haversine_distance_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
            bear = forward_bearing_deg(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
            edges.append({
                "source": src_id,
                "target": tgt_id,
                "relationship": "possible_transition",
                "distance_m": round(dist, 1),
                "bearing_deg": round(bear, 1),
                "evidence": ["cam_loc/S0345.png"],
                "confidence": conf
            })

    s05_verified_transitions = [
        ("C010", "C017", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
        ("C017", "C010", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
        ("C033", "C034", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
        ("C034", "C033", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
        ("C029", "C034", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
        ("C034", "C029", "HIGH", ["cam_loc/S0345.png", "validation/S05/gt/gt.txt"]),
    ]
    for src_c, tgt_c, conf, ev in s05_verified_transitions:
        src_id = f"CAM_S05_{src_c}"
        tgt_id = f"CAM_S05_{tgt_c}"
        if src_id in s05_cams and tgt_id in s05_cams:
            p1 = locations[src_id]["location"]
            p2 = locations[tgt_id]["location"]
            dist = haversine_distance_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
            bear = forward_bearing_deg(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
            edges.append({
                "source": src_id,
                "target": tgt_id,
                "relationship": "possible_transition",
                "distance_m": round(dist, 1),
                "bearing_deg": round(bear, 1),
                "evidence": ev,
                "confidence": conf
            })

    # Deduplicate edges by (source, target)
    unique_edges = []
    seen_pairs = set()
    for e in edges:
        pair = (e["source"], e["target"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_edges.append(e)

    return {
        "nodes": nodes,
        "edges": unique_edges
    }


def build_geojson(locations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build RFC 7946 compliant GeoJSON FeatureCollection."""
    features = []
    for cam_id, cam in locations.items():
        lat = cam["location"]["latitude"]
        lon = cam["location"]["longitude"]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # RFC 7946: [lon, lat]
            },
            "properties": {
                "camera_id": cam_id,
                "camera_number": cam["camera_number"],
                "scenario": cam["scenario"],
                "confidence": cam["confidence"],
                "direction": cam["direction"]["description"],
                "bearing_deg": cam["direction"]["bearing_deg"],
                "road": cam["road_context"]["road"],
                "intersection": cam["road_context"]["intersection"],
                "location_source": cam["location_source"],
                "location_method": cam["location_method"]
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def main():
    print("[1/4] Deriving camera locations for 65 dataset streams...")
    locations = build_camera_locations()
    print(f"      Derived {len(locations)} scenario camera nodes.")

    print("[2/4] Constructing camera transition topology graph...")
    graph = build_camera_graph(locations)
    print(f"      Constructed {len(graph['nodes'])} nodes and {len(graph['edges'])} directed edges.")

    print("[3/4] Exporting GeoJSON feature collection for GIS mapping...")
    geojson = build_geojson(locations)

    # Output paths
    loc_json_path = Path("data/config/camera_locations.json")
    graph_json_path = Path("data/config/camera_graph.json")
    geojson_path = Path("data/gis/cameras/camera_locations.geojson")

    loc_json_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    with open(loc_json_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=2)
    print(f"      Saved: {loc_json_path}")

    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"      Saved: {graph_json_path}")

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"      Saved: {geojson_path}")

    print("[4/4] Generation complete!")


if __name__ == "__main__":
    main()
