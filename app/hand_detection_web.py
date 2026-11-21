"""
Streamlit Web Application for Real-Time Hand Landmark Detection.
Part of the 'Real-Time Sign Language Recognition' Capstone Project.

Displays live webcam frames and MediaPipe hand landmarks directly in the web browser
using st.image(), eliminating any dependencies on cv2.imshow() or desktop GUI windows.
"""

import sys
import time
from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Configuration
CAMERA_INDEX = 0
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Standard 21-point hand skeleton connections
HAND_CONNECTIONS = [
    # Thumb connections
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger connections
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger connections
    (9, 10), (10, 11), (11, 12),
    # Ring finger connections
    (13, 14), (14, 15), (15, 16),
    # Pinky finger connections
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]


def ensure_model_file() -> Path:
    """
    Verifies that the MediaPipe Tasks hand_landmarker.task model file exists locally.
    Downloads it from the official Google repository if missing.
    """
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with st.spinner("Downloading MediaPipe hand landmarker model asset..."):
                urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        except Exception as e:
            st.error(f"Failed to download model asset: {e}")
            raise
    return MODEL_PATH


@st.cache_resource(show_spinner=False)
def get_hand_landmarker():
    """
    Initializes and caches the MediaPipe Tasks HandLandmarker detector.
    """
    model_file = ensure_model_file()
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_file))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_skeleton_rgb(frame_rgb, hand_landmarks_list):
    """
    Draws hand connections and 21 landmark keypoints directly onto an RGB frame.
    Uses RGB color values suitable for st.image() display.
    """
    h, w, _ = frame_rgb.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert normalized coordinates (0.0 to 1.0) into pixel coordinates
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # 1. Draw connection skeleton lines (RGB: vibrant light-green)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(
                    frame_rgb,
                    points[start_idx],
                    points[end_idx],
                    (0, 230, 115),
                    2,
                    cv2.LINE_AA,
                )

        # 2. Draw all 21 landmark keypoints
        for idx, (px, py) in enumerate(points):
            if idx in (0, 4, 8, 12, 16, 20):  # Wrist and 5 fingertips
                cv2.circle(frame_rgb, (px, py), 6, (255, 50, 50), -1, cv2.LINE_AA)       # Bright red in RGB
                cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)     # White outline
            else:  # Intermediate knuckles
                cv2.circle(frame_rgb, (px, py), 4, (30, 144, 255), -1, cv2.LINE_AA)     # Dodger blue in RGB
                cv2.circle(frame_rgb, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)     # White outline


def draw_overlay_hud(frame_rgb, hand_count: int):
    """
    Draws a clean info card overlay directly on the video frame.
    """
    # Background card
    cv2.rectangle(frame_rgb, (15, 15), (290, 75), (25, 25, 25), -1)
    border_color = (0, 230, 115) if hand_count > 0 else (255, 165, 0)
    cv2.rectangle(frame_rgb, (15, 15), (290, 75), border_color, 2)

    # Hand count text
    cv2.putText(
        frame_rgb,
        f"Hands Detected: {hand_count}",
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    st.set_page_config(
        page_title="Real-Time Hand Detection - Web View",
        page_icon="🖐️",
        layout="wide",
    )

    st.title("🖐️ Real-Time Hand Detection (Web Interface)")
    st.markdown(
        "**Real-Time Sign Language Recognition Capstone Project**  \n"
        "Webcam video stream powered by **OpenCV** and **MediaPipe Tasks API**, rendered directly in your browser without `cv2.imshow()`."
    )
    st.divider()

    # Session state management for camera controls
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    # Control buttons
    col_ctrl1, col_ctrl2, col_info = st.columns([1, 1, 2])
    with col_ctrl1:
        start_clicked = st.button(
            "▶️ Start Camera",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.camera_running,
        )
    with col_ctrl2:
        stop_clicked = st.button(
            "⏹️ Stop Camera",
            use_container_width=True,
            disabled=not st.session_state.camera_running,
        )

    if start_clicked:
        st.session_state.camera_running = True
        st.rerun()

    if stop_clicked:
        st.session_state.camera_running = False
        st.rerun()

    # Layout: Status columns and main video viewport
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_status = metric_col1.empty()
    metric_hands = metric_col2.empty()
    metric_fps = metric_col3.empty()

    frame_placeholder = st.empty()

    # If camera is stopped
    if not st.session_state.camera_running:
        metric_status.info("Status: Camera Stopped")
        metric_hands.metric(label="Hands Detected", value=0)
        metric_fps.metric(label="Inference Latency", value="-- ms")
        frame_placeholder.info("👉 Click **'Start Camera'** above to begin live hand detection.")
        return

    # Initialize MediaPipe Tasks HandLandmarker
    try:
        landmarker = get_hand_landmarker()
    except Exception as e:
        st.error(f"Failed to initialize MediaPipe Tasks detector: {e}")
        st.session_state.camera_running = False
        return

    # Initialize Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(
            f"❌ Unable to access webcam at index {CAMERA_INDEX}.\n\n"
            "**Troubleshooting Checklist:**\n"
            "1. Make sure your webcam is plugged in and recognized.\n"
            "2. Check Windows Settings -> Privacy & Security -> Camera (Allow apps access).\n"
            "3. Ensure no other application (Zoom, Teams, Discord, another browser tab) is locking the webcam."
        )
        st.session_state.camera_running = False
        return

    metric_status.success("Status: Camera Streaming Live")

    # Camera streaming loop
    try:
        while st.session_state.camera_running:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                st.warning("⚠️ Frame capture dropped. Retrying...")
                time.sleep(0.05)
                continue

            # 1. Mirror the video horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)

            # 2. Convert BGR to RGB for processing & web display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Detect hands using MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            hand_count = 0
            if detection_result.hand_landmarks:
                hand_count = len(detection_result.hand_landmarks)
                # 4. Draw all 21 landmarks and skeletal connections
                draw_hand_skeleton_rgb(rgb_frame, detection_result.hand_landmarks)

            # 5. Draw overlay banner
            draw_overlay_hud(rgb_frame, hand_count)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # 6. Update Streamlit metrics
            metric_hands.metric(label="Hands Detected", value=f"{hand_count} / {MAX_NUM_HANDS}")
            metric_fps.metric(label="Frame Latency", value=f"{latency_ms} ms")

            # 7. Render frame directly in browser using st.image()
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Brief sleep to yield execution thread smoothly
            time.sleep(0.01)

    except Exception as e:
        # Allow Streamlit internal flow exceptions (like rerun/stop) to propagate
        if "Rerun" in type(e).__name__ or "Stop" in type(e).__name__:
            raise e
        st.error(f"An unexpected error occurred during streaming: {e}")

    finally:
        # Proper webcam release
        if cap is not None and cap.isOpened():
            cap.release()
        st.session_state.camera_running = False


if __name__ == "__main__":
    main()
