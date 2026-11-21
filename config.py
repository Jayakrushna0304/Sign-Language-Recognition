"""
Central configuration file for the Real-Time Sign Language Recognition project.
Defines directory paths, camera parameters, model hyperparameters, and label mappings.
"""

from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

PREPROCESSING_DIR = BASE_DIR / "preprocessing"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODEL_DIR = BASE_DIR / "saved_model"
REALTIME_DIR = BASE_DIR / "realtime"
APP_DIR = BASE_DIR / "app"
UTILS_DIR = BASE_DIR / "utils"
RESULTS_DIR = BASE_DIR / "results"

# Webcam & Real-time Inference Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CONFIDENCE_THRESHOLD = 0.70

# MediaPipe Hand Landmarks Configuration
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Model & Training Configurations
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
MODEL_SAVE_PATH = SAVED_MODEL_DIR / "sign_language_model.pkl"
LABEL_ENCODER_PATH = SAVED_MODEL_DIR / "label_encoder.pkl"

# Supported Sign Labels (Placeholder list - easily customizable)
ACTIONS = [
    "Hello",
    "Thank You",
    "Yes",
    "No",
    "I Love You",
    "Please",
    "Help",
    "Sorry"
]
