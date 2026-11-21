"""
Streamlit Data Collection Application for Real-Time Sign Language Recognition.
Collects and structures 21-point hand landmark datasets (X, Y, Z coordinates for up to 2 hands)
for 20 basic sign language vocabulary words.

Features:
- Live mirrored webcam stream via OpenCV.
- MediaPipe Tasks API hand landmark extraction.
- Single-sample, auto-interval, and 30-frame sequence collection modes.
- Saves data to dataset/landmarks_data.csv without overwriting existing data.
- Unique sequence_id progress tracking: counts each recording (single or 30-frame sequence) as 1.
- Real-time progress table and statistics for all 20 vocabulary words against a 50-recording target.
- Modern Streamlit parameters (width="stretch") eliminating deprecation warnings.
"""

import sys
import time
import uuid
from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
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
SEQUENCE_LENGTH = 30  # Frame count for dynamic gesture sequence recording
TARGET_RECORDINGS_PER_WORD = 50  # Target unique recordings per word

# File paths
DATASET_DIR = PROJECT_ROOT / "dataset"
CSV_FILE = DATASET_DIR / "landmarks_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# 20 Target Sign Language Words
WORDS = [
    "Hello",
    "Thank You",
    "Please",
    "Sorry",
    "Yes",
    "No",
    "Help",
    "Stop",
    "Good",
    "Bad",
    "Water",
    "Food",
    "Eat",
    "Drink",
    "Love",
    "Friend",
    "Family",
    "Name",
    "Welcome",
    "Goodbye"
]

# Standard 21-point hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base
]

# Generate standard feature column names for up to 2 hands (21 points x 3 coords x 2 hands = 126 features)
FEATURE_COLUMNS = []
for h in range(2):
    for pt in range(21):
        FEATURE_COLUMNS.extend([f"h{h}_x{pt}", f"h{h}_y{pt}", f"h{h}_z{pt}"])

METADATA_COLUMNS = ["label", "sequence_id", "frame_id", "timestamp", "num_hands"]
ALL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def ensure_model_file() -> Path:
    """
    Verifies that the MediaPipe Tasks hand_landmarker.task model file is available.
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


def extract_landmark_features(hand_landmarks_list):
    """
    Extracts 126 coordinate values (X, Y, Z for 21 points x 2 hands).
    Sorts hands left-to-right across the frame for consistent feature alignment.
    Fills undetected hands with 0.0.
    """
    features = [0.0] * 126
    if not hand_landmarks_list:
        return features

    # Sort hands by wrist (landmark 0) x-coordinate for deterministic slotting
    sorted_hands = sorted(hand_landmarks_list, key=lambda hand: hand[0].x)

    for hand_idx, hand_landmarks in enumerate(sorted_hands[:2]):
        offset = hand_idx * 63
        for pt_idx, lm in enumerate(hand_landmarks):
            features[offset + pt_idx * 3 + 0] = float(lm.x)
            features[offset + pt_idx * 3 + 1] = float(lm.y)
            features[offset + pt_idx * 3 + 2] = float(lm.z)

    return features


def draw_hand_skeleton_rgb(frame_rgb, hand_landmarks_list):
    """
    Draws hand connections and 21 landmark keypoints directly onto an RGB frame.
    """
    h, w, _ = frame_rgb.shape

    for hand_landmarks in hand_landmarks_list:
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
            if idx in (0, 4, 8, 12, 16, 20):  # Wrist and fingertips
                cv2.circle(frame_rgb, (px, py), 6, (255, 50, 50), -1, cv2.LINE_AA)
                cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:  # Intermediate knuckles
                cv2.circle(frame_rgb, (px, py), 4, (30, 144, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_rgb, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay_card(frame_rgb, selected_word: str, hand_count: int, recording_count: int, mode_text: str = ""):
    """
    Renders an informative HUD overlay banner directly on the camera frame.
    """
    # Background card
    cv2.rectangle(frame_rgb, (15, 15), (440, 115), (25, 25, 25), -1)
    border_color = (0, 230, 115) if hand_count > 0 else (255, 165, 0)
    cv2.rectangle(frame_rgb, (15, 15), (440, 115), border_color, 2)

    # Word label
    cv2.putText(
        frame_rgb,
        f"Sign: {selected_word}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Hand count and current unique recording total
    cv2.putText(
        frame_rgb,
        f"Hands: {hand_count} | Recordings: {recording_count}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        border_color,
        2,
        cv2.LINE_AA,
    )

    # Mode or status
    status_str = mode_text if mode_text else ("Ready" if hand_count > 0 else "Show Hand(s) to Record")
    cv2.putText(
        frame_rgb,
        f"Status: {status_str}",
        (30, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def save_samples_to_csv(records: list[dict]) -> bool:
    """
    Appends landmark records to dataset/landmarks_data.csv.
    Creates the CSV file with headers if it does not already exist.
    Guarantees no existing data is deleted or overwritten.
    """
    if not records:
        return False

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(records)

    # Ensure all expected columns exist in order
    for col in ALL_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = 0.0
    df_new = df_new[ALL_COLUMNS]

    if not CSV_FILE.exists():
        df_new.to_csv(CSV_FILE, mode="w", index=False)
    else:
        df_new.to_csv(CSV_FILE, mode="a", header=False, index=False)

    return True


def get_dataset_stats(target_per_word: int = TARGET_RECORDINGS_PER_WORD):
    """
    Reads the CSV and calculates UNIQUE RECORDING counts (using sequence_id)
    and progress metrics for all 20 words.
    A 30-frame sequence counts as 1 recording.
    A single-frame capture counts as 1 recording.
    """
    counts = {word: 0 for word in WORDS}
    total_recordings = 0
    total_frames = 0

    if CSV_FILE.exists():
        try:
            df = pd.read_csv(CSV_FILE)
            total_frames = len(df)
            if "label" in df.columns:
                if "sequence_id" in df.columns:
                    # Count unique sequence_id values for each sign label
                    unique_counts = (
                        df.groupby("label")["sequence_id"]
                        .nunique()
                        .to_dict()
                    )
                    for word in WORDS:
                        counts[word] = int(unique_counts.get(word, 0))
                    total_recordings = df["sequence_id"].nunique()
                else:
                    # Fallback if sequence_id column is missing
                    existing_counts = df["label"].value_counts().to_dict()
                    for word in WORDS:
                        counts[word] = int(existing_counts.get(word, 0))
                    total_recordings = total_frames
        except Exception:
            pass

    summary_rows = []
    for idx, word in enumerate(WORDS, 1):
        c = counts[word]
        pct = min(100.0, (c / target_per_word) * 100) if target_per_word > 0 else 0
        if c >= target_per_word:
            status = "✅ Target Reached"
        elif c > 0:
            status = "🟡 In Progress"
        else:
            status = "⚪ Pending"

        summary_rows.append({
            "#": idx,
            "Word Label": word,
            "Unique Recordings": c,
            "Target": target_per_word,
            "Completion": f"{pct:.0f}%",
            "Status": status
        })

    summary_df = pd.DataFrame(summary_rows)
    return counts, total_recordings, total_frames, summary_df


def main():
    st.set_page_config(
        page_title="Sign Language Data Collection",
        page_icon="✍️",
        layout="wide",
    )

    st.title("✍️ Sign Language Data Collection Studio")
    st.markdown(
        "**Real-Time Sign Language Recognition Capstone Project**  \n"
        "Collect 21-point hand landmark coordinates across 20 basic vocabulary words. "
        "Supports both **single static frames** and **30-frame temporal sequences** for machine learning training."
    )
    st.divider()

    # Session state initialization
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "capture_trigger" not in st.session_state:
        st.session_state.capture_trigger = False
    if "sequence_trigger" not in st.session_state:
        st.session_state.sequence_trigger = False
    if "last_saved_msg" not in st.session_state:
        st.session_state.last_saved_msg = ""

    target_recordings = TARGET_RECORDINGS_PER_WORD

    # Read current dataset counts (unique sequence_id recordings)
    word_recordings, total_recordings, total_frames, summary_df = get_dataset_stats(target_recordings)

    # ------------------ SIDEBAR: CONTROLS & SELECTION ------------------
    with st.sidebar:
        st.header("🎯 Sign Selection")
        selected_word = st.selectbox(
            "Select Word to Collect:",
            options=WORDS,
            index=0,
            help="Choose the sign language gesture you want to record."
        )

        current_word_recordings = word_recordings[selected_word]
        st.metric(
            label=f"Recordings for '{selected_word}'",
            value=f"{current_word_recordings} / {target_recordings}",
            delta=f"{current_word_recordings - target_recordings} of target ({target_recordings})"
        )

        st.divider()
        st.header("📸 Collection Actions")

        btn_single_side = st.button(
            "📸 Collect Sample",
            type="primary",
            width="stretch",
            disabled=not st.session_state.camera_running,
            key="side_collect",
            help="Saves the current frame with a unique sequence ID to the dataset."
        )

        btn_sequence_side = st.button(
            "🎬 Record Sequence (30 Frames)",
            width="stretch",
            disabled=not st.session_state.camera_running,
            key="side_sequence",
            help="Records a continuous 30-frame sequence under a single unique sequence ID (counts as 1 recording)."
        )

        auto_mode = st.toggle(
            "⚡ Auto-Collect Interval",
            value=False,
            disabled=not st.session_state.camera_running,
            help="Automatically records 1 sample with a unique sequence ID every 0.6 seconds while hands are detected."
        )

        if btn_single_side:
            st.session_state.capture_trigger = True
        if btn_sequence_side:
            st.session_state.sequence_trigger = True

        st.divider()
        st.markdown(f"**Dataset Location:**  \n`{CSV_FILE}`")
        if CSV_FILE.exists():
            st.caption(f"Unique Recordings: {total_recordings} | Total Frames: {total_frames}")

    # ------------------ MAIN PAGE: CAMERA & FEED ------------------
    col_cam1, col_cam2, col_act1, col_act2 = st.columns([1, 1, 1.2, 1.5])
    with col_cam1:
        if st.button("▶️ Start Camera", type="primary", width="stretch", disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.rerun()
    with col_cam2:
        if st.button("⏹️ Stop Camera", width="stretch", disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
            st.rerun()
    with col_act1:
        if st.button("📸 Collect Sample", width="stretch", disabled=not st.session_state.camera_running, key="main_collect"):
            st.session_state.capture_trigger = True
    with col_act2:
        if st.button("🎬 Record Sequence (30f)", width="stretch", disabled=not st.session_state.camera_running, key="main_sequence"):
            st.session_state.sequence_trigger = True

    # Metric banners
    met1, met2, met3 = st.columns(3)
    metric_status = met1.empty()
    metric_hands = met2.empty()
    metric_total = met3.empty()

    frame_placeholder = st.empty()
    feedback_placeholder = st.empty()

    if st.session_state.last_saved_msg:
        feedback_placeholder.success(st.session_state.last_saved_msg)

    # Dataset Progress Table (Visible in both stopped & active states)
    st.subheader("📋 Dataset Progress Table (All 20 Words)")
    progress_table_placeholder = st.empty()
    progress_table_placeholder.dataframe(summary_df, width="stretch", hide_index=True)

    # If camera is stopped
    if not st.session_state.camera_running:
        metric_status.info("Status: Camera Stopped")
        metric_hands.metric("Hands Detected", "0")
        metric_total.metric("Total Unique Recordings", f"{total_recordings} (Frames: {total_frames})")
        frame_placeholder.info("👉 Click **'Start Camera'** to open the webcam and begin collecting sign language recordings.")
        return

    # Initialize MediaPipe detector
    try:
        landmarker = get_hand_landmarker()
    except Exception as e:
        st.error(f"Failed to load MediaPipe Tasks Landmarker: {e}")
        st.session_state.camera_running = False
        return

    # Initialize Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(
            f"❌ Unable to open webcam at index {CAMERA_INDEX}.\n\n"
            "Please check camera connections and Windows privacy permissions."
        )
        st.session_state.camera_running = False
        return

    metric_status.success("Status: Camera Streaming Live")
    metric_total.metric("Total Unique Recordings", f"{total_recordings} (Frames: {total_frames})")

    last_auto_time = time.time()
    auto_interval = 0.6  # seconds between auto-captures

    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ Dropped frame from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # 1. Mirror horizontally
            frame = cv2.flip(frame, 1)

            # 2. Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Detect hands using MediaPipe Tasks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            hand_count = 0
            hand_landmarks_list = []
            if detection_result.hand_landmarks:
                hand_landmarks_list = detection_result.hand_landmarks
                hand_count = len(hand_landmarks_list)
                draw_hand_skeleton_rgb(rgb_frame, hand_landmarks_list)

            # Metrics
            metric_hands.metric("Hands Detected", f"{hand_count} / {MAX_NUM_HANDS}")

            # ------------------ DATA COLLECTION HANDLING ------------------
            mode_text = ""
            data_updated = False

            # Case A: Record 30-Frame Sequence (Counts as 1 Unique Recording)
            if st.session_state.sequence_trigger:
                if hand_count == 0:
                    feedback_placeholder.warning("⚠️ No hands detected! Please position your hand(s) in view before recording sequence.")
                    st.session_state.sequence_trigger = False
                else:
                    seq_id = f"seq_{uuid.uuid4().hex[:8]}"
                    seq_records = []
                    progress_bar = st.progress(0, text=f"Recording 30-frame sequence for '{selected_word}'...")

                    for frame_idx in range(SEQUENCE_LENGTH):
                        ret_seq, seq_frame = cap.read()
                        if not ret_seq or seq_frame is None:
                            continue

                        seq_frame = cv2.flip(seq_frame, 1)
                        seq_rgb = cv2.cvtColor(seq_frame, cv2.COLOR_BGR2RGB)
                        seq_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=seq_rgb)
                        seq_res = landmarker.detect(seq_mp_image)

                        seq_hands = seq_res.hand_landmarks if seq_res.hand_landmarks else []
                        features = extract_landmark_features(seq_hands)

                        row = {
                            "label": selected_word,
                            "sequence_id": seq_id,
                            "frame_id": frame_idx,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "num_hands": len(seq_hands),
                        }
                        for col_name, val in zip(FEATURE_COLUMNS, features):
                            row[col_name] = round(val, 6)
                        seq_records.append(row)

                        # Live preview during sequence recording
                        if seq_hands:
                            draw_hand_skeleton_rgb(seq_rgb, seq_hands)
                        draw_overlay_card(seq_rgb, selected_word, len(seq_hands), current_word_recordings + 1, f"Recording {frame_idx + 1}/{SEQUENCE_LENGTH}")
                        frame_placeholder.image(seq_rgb, channels="RGB", width="stretch")
                        progress_bar.progress((frame_idx + 1) / SEQUENCE_LENGTH, text=f"Recording '{selected_word}': {frame_idx + 1}/{SEQUENCE_LENGTH}")
                        time.sleep(0.03)

                    save_samples_to_csv(seq_records)
                    progress_bar.empty()
                    st.session_state.sequence_trigger = False
                    current_word_recordings += 1  # 30 frames count as 1 unique recording
                    total_recordings += 1
                    total_frames += len(seq_records)
                    data_updated = True
                    st.session_state.last_saved_msg = f"✅ Saved 30-frame sequence for '{selected_word}'! (Recording #{current_word_recordings}, Sequence ID: {seq_id})"
                    feedback_placeholder.success(st.session_state.last_saved_msg)

            # Case B: Collect Single Sample (Counts as 1 Unique Recording)
            elif st.session_state.capture_trigger:
                if hand_count == 0:
                    feedback_placeholder.warning("⚠️ No hands detected! Show at least 1 hand to collect landmark sample.")
                else:
                    seq_id = f"single_{uuid.uuid4().hex[:8]}"
                    features = extract_landmark_features(hand_landmarks_list)
                    row = {
                        "label": selected_word,
                        "sequence_id": seq_id,
                        "frame_id": 0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_hands": hand_count,
                    }
                    for col_name, val in zip(FEATURE_COLUMNS, features):
                        row[col_name] = round(val, 6)

                    save_samples_to_csv([row])
                    current_word_recordings += 1
                    total_recordings += 1
                    total_frames += 1
                    mode_text = "Recording Saved!"
                    data_updated = True
                    st.session_state.last_saved_msg = f"✅ Collected recording #{current_word_recordings} for '{selected_word}'"
                    feedback_placeholder.success(st.session_state.last_saved_msg)

                st.session_state.capture_trigger = False

            # Case C: Auto-Interval Mode (Each interval gets a unique sequence_id)
            elif auto_mode and hand_count > 0:
                now = time.time()
                if now - last_auto_time >= auto_interval:
                    last_auto_time = now
                    seq_id = f"auto_{uuid.uuid4().hex[:8]}"
                    features = extract_landmark_features(hand_landmarks_list)
                    row = {
                        "label": selected_word,
                        "sequence_id": seq_id,
                        "frame_id": 0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_hands": hand_count,
                    }
                    for col_name, val in zip(FEATURE_COLUMNS, features):
                        row[col_name] = round(val, 6)

                    save_samples_to_csv([row])
                    current_word_recordings += 1
                    total_recordings += 1
                    total_frames += 1
                    mode_text = f"Auto Saved #{current_word_recordings}"
                    data_updated = True
                    st.session_state.last_saved_msg = f"⚡ Auto-collected recording #{current_word_recordings} for '{selected_word}'"
                    feedback_placeholder.info(st.session_state.last_saved_msg)

            # Update progress table live when new data is added
            if data_updated:
                _, _, _, updated_summary_df = get_dataset_stats(target_recordings)
                progress_table_placeholder.dataframe(updated_summary_df, width="stretch", hide_index=True)

            # Render overlay HUD card
            draw_overlay_card(rgb_frame, selected_word, hand_count, current_word_recordings, mode_text)

            # Display video frame
            frame_placeholder.image(rgb_frame, channels="RGB", width="stretch")
            metric_total.metric("Total Unique Recordings", f"{total_recordings} (Frames: {total_frames})")

            # Smooth loop pacing
            time.sleep(0.01)

    except Exception as e:
        if "Rerun" in type(e).__name__ or "Stop" in type(e).__name__:
            raise e
        st.error(f"Error during video capture: {e}")

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        st.session_state.camera_running = False


if __name__ == "__main__":
    main()
