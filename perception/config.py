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
CAMERA_ID = "CAM_001"
CAMERA_METADATA = {}
CAMERA_METADATA_PATH = "data/config/camera_metadata.json"
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
REID_MODEL_NAME = "osnet_x0_25"
REID_MODEL_WEIGHTS = "msmt17"
REID_EMBEDDING_DIM = 512
MIN_REID_CROP_SIZE = 32
EXPORT_TRACK_EMBEDDINGS = True

# Provenance model identifiers (factual names of the loaded artefacts).
DETECTOR_MODEL = "yolov8s.pt"
TRACKER_MODEL = "bytetrack"
OCR_MODEL = "easyocr"
FPS_SOURCE = "video_metadata"

# Synthetic degradation benchmark configuration.
# Enabled by default; rates are conservative (0.02-0.10) so generation is
# lightweight. Synthetic outputs never affect the real perception outputs
# in data/output/.
SYNTHETIC_ENABLE = True

SYNTHETIC_INPUT_DIR = "data/output"
SYNTHETIC_OUTPUT_DIR = "data/synthetic_output"

# Conservative benchmark settings
SYNTHETIC_DROP_FRAME_RATE = 0.05
SYNTHETIC_MISSING_PLATE_RATE = 0.10
SYNTHETIC_OCR_FAILURE_RATE = 0.00
SYNTHETIC_LOW_OCR_RATE = 0.08
SYNTHETIC_MISSING_REID_RATE = 0.08
SYNTHETIC_OCCLUSION_RATE = 0.08
SYNTHETIC_CONFIDENCE_DEGRADATION_RATE = 0.00
SYNTHETIC_TRACK_FRAGMENT_RATE = 0.05
SYNTHETIC_CAMERA_OUTAGE_RATE = 0.02
SYNTHETIC_OCCLUSION_CONFIDENCE_MIN = 0.10
SYNTHETIC_OCCLUSION_CONFIDENCE_MAX = 0.40
SYNTHETIC_CONFIDENCE_DEGRADATION_MIN = 0.30
