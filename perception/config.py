"""Perception layer configuration."""

CONFIDENCE_THRESHOLD = 0.55
PLATE_MODEL_PATH = "models/license_plate_yolo11s.pt"
PLATE_BOX_COLOR = (255, 255, 0)
PLATE_LABEL_COLOR = (255, 255, 0)
PLATE_CONF_THRESHOLD = 0.40
OCR_CONF_THRESHOLD = 0.45
OCR_CACHE_FRAMES = 15
OCR_RETRY_FRAMES = 10
TRACKER_CONFIG = {
    "tracker": "bytetrack.yaml",
    "track_high_thresh": 0.45,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.35,
    "track_buffer": 60,
    "match_thresh": 0.80,
    "fuse_score": True,
}
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
ANNOTATION_COLOR = (0, 255, 0)
LABEL_TEXT_COLOR = (0, 0, 0)
LABEL_BACKGROUND_COLOR = (0, 255, 0)
HUD_BACKGROUND_COLOR = (18, 25, 22)
HUD_TEXT_COLOR = (255, 255, 255)
HUD_ACCENT_COLOR = (0, 200, 120)
HUD_OPACITY = 0.78
HUD_PADDING = 12
HUD_RADIUS = 14
CAMERA_ID = None
CAMERA_METADATA = {}
SYSTEM_TITLE = "UrbanTrack AI v1.0"
REFERENCE_FRAME_WIDTH = 640
REFERENCE_FRAME_HEIGHT = 480
MIN_LINE_THICKNESS = 1
HUD_FONT_SCALE_LIMITS = (0.5, 1.2)
LABEL_FONT_SCALE_LIMITS = (0.4, 0.8)
TRAIL_LENGTH = 120
TRAIL_MAX_THICKNESS = 6
TRAIL_MIN_THICKNESS = 2
TRAIL_GAP_PIXELS = 8
INACTIVE_TRACK_MEMORY = 45
VEHICLE_COLORS = {
    "car": (255, 0, 0),
    "auto": (0, 255, 255),
    "motorcycle": (0, 165, 255),
    "bus": (0, 255, 255),
    "truck": (255, 0, 255),
}
