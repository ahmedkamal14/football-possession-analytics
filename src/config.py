import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Model weights
MODEL_BEST_PATH = ROOT_DIR / "Model" / "football_detector_best.pt"
MODEL_FALLBACK_PATH = ROOT_DIR / "Model" / "best.pt"
MODEL_PATH = MODEL_BEST_PATH if MODEL_BEST_PATH.exists() else MODEL_FALLBACK_PATH

# Test Data Directory
TEST_DATA_DIR = ROOT_DIR / "Test Data"
INPUT_VIDEO_PATH = TEST_DATA_DIR / "test_video.mp4"
OUTPUT_VIDEO_PATH = TEST_DATA_DIR / "possession_output.mp4"

# Sampling settings for team clustering
SAMPLE_MAX_FRAMES = 180
SAMPLE_FRAME_STEP = 3

# Ball possession settings
POSSESSION_DIST_THRESHOLD = 80.0  # Max pixel distance between ball and player to count as possession

# Class mapping (YOLO output)
CLASS_BALL = 0
CLASS_PLAYER = 2
CLASS_REFEREE = 3
CLASS_NAMES = {0: "ball", 2: "player", 3: "referee"}
