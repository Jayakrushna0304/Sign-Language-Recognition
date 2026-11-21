"""
app/predict.py
==============
Real-Time Sign Language Prediction — Streamlit Web Application
Capstone Project: Real-Time Sign Language Recognition Using Machine Learning

Loads the trained classifier and label encoder from saved_model/, captures live
webcam frames, extracts exactly 126 hand-landmark features (identical format to
training), and displays the predicted sign word with optional confidence score.

Run:
    .\.venv\Scripts\streamlit.exe run app/predict.py
"""

import sys
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX            = 0
MAX_NUM_HANDS           = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── File paths ────────────────────────────────────────────────────────────────
MEDIAPIPE_TASK_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH   = PROJECT_ROOT / "saved_model" / "sign_language_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "saved_model" / "label_encoder.pkl"

# ── 126 feature column names (must match training order exactly) ──────────────
# Order: h0_x0, h0_y0, h0_z0, h0_x1, h0_y1, h0_z1 … h1_x20, h1_y20, h1_z20
FEATURE_COLUMNS = []
for _h in range(2):
    for _pt in range(21):
        FEATURE_COLUMNS.extend([f"h{_h}_x{_pt}", f"h{_h}_y{_pt}", f"h{_h}_z{_pt}"])

# ── Standard 21-point hand skeleton connections ───────────────────────────────
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),   # Thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),   # Index
    (9, 10), (10, 11),(11, 12),           # Middle
    (13, 14),(14, 15),(15, 16),           # Ring
    (0, 17), (17, 18),(18, 19),(19, 20),  # Pinky
    (5, 9),  (9, 13), (13, 17),          # Palm base
]


# ════════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ════════════════════════════════════════════════════════════════════════════════

def ensure_mediapipe_task() -> Path:
    """
    Checks that the MediaPipe Tasks hand_landmarker.task file exists.
    Downloads it from the official Google repository if missing.
    """
    if not MEDIAPIPE_TASK_PATH.exists():
        MEDIAPIPE_TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        with st.spinner("Downloading MediaPipe hand landmarker model (one-time) …"):
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(MEDIAPIPE_TASK_PATH))
    return MEDIAPIPE_TASK_PATH


@st.cache_resource(show_spinner=False)
def load_hand_landmarker():
    """
    Initialises and caches the MediaPipe Tasks HandLandmarker detector.
    Uses IMAGE running mode so it is safe to call detect() per frame.
    """
    task_path   = ensure_mediapipe_task()
    base_opts   = mp_tasks.BaseOptions(model_asset_path=str(task_path))
    options     = vision.HandLandmarkerOptions(
        base_options             = base_opts,
        running_mode             = vision.RunningMode.IMAGE,
        num_hands                = MAX_NUM_HANDS,
        min_hand_detection_confidence = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence  = MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Loads the trained sklearn classifier and label encoder from disk.
    Returns (model, label_encoder, supports_proba: bool).
    Raises FileNotFoundError with a helpful message if files are missing.
    """
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not ENCODER_PATH.exists():
        missing.append(str(ENCODER_PATH))
    if missing:
        raise FileNotFoundError(
            "The following trained-model files were not found:\n"
            + "\n".join(f"  • {p}" for p in missing)
            + "\n\nPlease run:\n  .venv\\Scripts\\python.exe models/train_model.py"
        )

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Detect whether the model exposes predict_proba
    # (Random Forest / KNN: yes; plain SVC without probability=True: no)
    supports_proba = hasattr(model, "predict_proba")

    return model, encoder, supports_proba


# ════════════════════════════════════════════════════════════════════════════════
# Feature extraction  — MUST match data_collection.py exactly
# ════════════════════════════════════════════════════════════════════════════════

def extract_landmark_features(hand_landmarks_list) -> list:
    """
    Converts raw MediaPipe hand landmarks into a flat list of 126 floats.

    Layout (identical to training-time data_collection.py):
        Slot 0 (h0) → leftmost hand by wrist x-coordinate
        Slot 1 (h1) → rightmost hand (or zeros if only 1 hand present)

        Each slot: x0, y0, z0, x1, y1, z1 … x20, y20, z20  (63 values)

    Total: 2 hands × 21 landmarks × 3 coords = 126 features.
    Absent hands are filled with 0.0 (same as during training).
    """
    features = [0.0] * 126

    if not hand_landmarks_list:
        return features

    # Sort detected hands left-to-right by wrist (landmark index 0) x position
    sorted_hands = sorted(hand_landmarks_list, key=lambda hand: hand[0].x)

    for hand_idx, hand_landmarks in enumerate(sorted_hands[:2]):
        offset = hand_idx * 63  # 21 points × 3 coords
        for pt_idx, lm in enumerate(hand_landmarks):
            features[offset + pt_idx * 3 + 0] = float(lm.x)
            features[offset + pt_idx * 3 + 1] = float(lm.y)
            features[offset + pt_idx * 3 + 2] = float(lm.z)

    return features


def predict_sign(model, encoder, supports_proba: bool, features: list):
    """
    Runs inference on the 126-feature vector.

    Returns:
        predicted_label (str)  — the sign word string
        confidence (float | None) — probability 0–100 if available, else None
    """
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    encoded_pred = model.predict(X)[0]
    predicted_label = encoder.inverse_transform([encoded_pred])[0]

    confidence = None
    if supports_proba:
        proba = model.predict_proba(X)[0]
        confidence = float(proba.max()) * 100.0

    return predicted_label, confidence


# ════════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ════════════════════════════════════════════════════════════════════════════════

def draw_hand_skeleton_rgb(frame_rgb, hand_landmarks_list) -> None:
    """
    Draws 21 landmark keypoints and hand connections on the RGB frame in-place.
    Reused verbatim from hand_detection_web.py and data_collection.py.
    """
    h, w, _ = frame_rgb.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert normalised coords → pixel coords
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        # Draw connection lines (vibrant green)
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

        # Draw 21 keypoints
        for idx, (px, py) in enumerate(points):
            if idx in (0, 4, 8, 12, 16, 20):   # Wrist + fingertips → red
                cv2.circle(frame_rgb, (px, py), 6, (255, 50, 50),  -1, cv2.LINE_AA)
                cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:                                # Knuckles → blue
                cv2.circle(frame_rgb, (px, py), 4, (30, 144, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_rgb, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_prediction_overlay(
    frame_rgb,
    hand_count: int,
    predicted_label: str | None,
    confidence: float | None,
) -> None:
    """
    Draws a compact HUD card in the top-left corner of the frame showing:
      • Number of hands detected
      • Predicted sign word (or 'No hand detected')
      • Confidence percentage (if available)
    """
    # ── Background card ──────────────────────────────────────────────────────
    card_w, card_h = 400, 95
    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame_rgb, 0.35, 0, frame_rgb)

    # Border colour: green if detecting, orange if no hands
    border_color = (0, 230, 115) if hand_count > 0 else (255, 165, 0)
    cv2.rectangle(frame_rgb, (12, 12), (12 + card_w, 12 + card_h), border_color, 2)

    # ── Hand count ───────────────────────────────────────────────────────────
    cv2.putText(
        frame_rgb,
        f"Hands: {hand_count}",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # ── Prediction text ───────────────────────────────────────────────────────
    if predicted_label:
        display_word = predicted_label
        cv2.putText(
            frame_rgb,
            f"Sign: {display_word}",
            (22, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame_rgb,
            "No hand detected",
            (22, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 165, 0),
            2,
            cv2.LINE_AA,
        )

    # ── Confidence ────────────────────────────────────────────────────────────
    if confidence is not None and hand_count > 0:
        conf_text  = f"Confidence: {confidence:.1f}%"
        conf_color = (
            (80, 220, 100)  if confidence >= 70   # green
            else (255, 200, 0) if confidence >= 50 # yellow
            else (255, 80, 80)                      # red — low confidence
        )
        cv2.putText(
            frame_rgb,
            conf_text,
            (22, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            conf_color,
            1,
            cv2.LINE_AA,
        )


# ════════════════════════════════════════════════════════════════════════════════
# Main Streamlit application
# ════════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Sign Language Predictor",
        page_icon="🤟",
        layout="wide",
    )

    # ── Page header ──────────────────────────────────────────────────────────
    st.title("🤟 Real-Time Sign Language Recognition")
    st.markdown(
        "**Capstone Project — B.Tech Final Year**  \n"
        "Live sign-language prediction powered by **MediaPipe Tasks API** + "
        "**scikit-learn**. Show your hand to the webcam and the trained model "
        "will recognise the sign in real time."
    )
    st.divider()

    # ── Session state ─────────────────────────────────────────────────────────
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    # ── Load model (cached) — do this before camera controls ─────────────────
    model_load_error = None
    model = encoder = supports_proba = None
    try:
        model, encoder, supports_proba = load_classifier()
    except FileNotFoundError as exc:
        model_load_error = str(exc)
    except Exception as exc:
        model_load_error = f"Unexpected error loading model: {exc}"

    if model_load_error:
        st.error(f"❌ **Model Not Found**\n\n```\n{model_load_error}\n```")
        st.info(
            "**Steps to fix:**\n"
            "1. Collect training data with `app/data_collection.py`\n"
            "2. Prepare the dataset with `preprocessing/prepare_dataset.py`\n"
            "3. Train the model with `models/train_model.py`\n"
            "4. Then reopen this page."
        )
        return   # Cannot proceed without a trained model

    # ── Show model info ───────────────────────────────────────────────────────
    with st.expander("ℹ️ Loaded Model Information", expanded=False):
        st.markdown(f"- **Model file:** `{MODEL_PATH}`")
        st.markdown(f"- **Encoder file:** `{ENCODER_PATH}`")
        st.markdown(f"- **Model type:** `{type(model).__name__}`")
        st.markdown(f"- **Classes ({len(encoder.classes_)}):** {', '.join(encoder.classes_)}")
        st.markdown(
            f"- **Confidence score:** {'✅ Available (predict_proba)' if supports_proba else '⚠️ Not available for this model type'}"
        )

    # ── Camera controls ───────────────────────────────────────────────────────
    col_start, col_stop, col_spacer = st.columns([1, 1, 4])
    with col_start:
        if st.button(
            "▶️ Start Camera",
            type="primary",
            width="stretch",
            disabled=st.session_state.camera_running,
        ):
            st.session_state.camera_running = True
            st.rerun()

    with col_stop:
        if st.button(
            "⏹️ Stop Camera",
            width="stretch",
            disabled=not st.session_state.camera_running,
        ):
            st.session_state.camera_running = False
            st.rerun()

    # ── Live metrics row ──────────────────────────────────────────────────────
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    status_ph    = met_col1.empty()
    hands_ph     = met_col2.empty()
    pred_ph      = met_col3.empty()
    conf_ph      = met_col4.empty()

    # ── Video frame placeholder ───────────────────────────────────────────────
    frame_ph = st.empty()

    # ── Large prediction display (below the video) ────────────────────────────
    st.markdown("---")
    pred_banner_ph = st.empty()

    # ── Stopped state ─────────────────────────────────────────────────────────
    if not st.session_state.camera_running:
        status_ph.info("📷 Camera: Stopped")
        hands_ph.metric("Hands Detected", "—")
        pred_ph.metric("Prediction", "—")
        conf_ph.metric("Confidence", "—")
        frame_ph.info(
            "👉 Click **Start Camera** above to begin live sign language prediction."
        )
        pred_banner_ph.markdown(
            "<div style='text-align:center; padding:20px; "
            "background:#1e1e1e; border-radius:12px; border:2px solid #444;'>"
            "<p style='color:#888; font-size:1.1rem; margin:0;'>"
            "Start the camera to see predictions here</p></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Initialise MediaPipe (cached) ─────────────────────────────────────────
    try:
        landmarker = load_hand_landmarker()
    except Exception as exc:
        st.error(f"❌ Failed to initialise MediaPipe Tasks detector: {exc}")
        st.session_state.camera_running = False
        return

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(
            f"❌ **Cannot access webcam** (index {CAMERA_INDEX}).\n\n"
            "**Troubleshooting steps:**\n"
            "1. Make sure your webcam is connected and recognised by Windows.\n"
            "2. Go to **Settings → Privacy & Security → Camera** and allow app access.\n"
            "3. Close any other app that may be using the webcam (Zoom, Teams, OBS, etc.).\n"
            "4. Try a different webcam index by changing `CAMERA_INDEX` at the top of this file."
        )
        st.session_state.camera_running = False
        return

    status_ph.success("📷 Camera: Live")

    # ── Prediction streaming loop ──────────────────────────────────────────────
    try:
        while st.session_state.camera_running:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ Dropped frame — retrying …")
                time.sleep(0.05)
                continue

            # 1. Mirror horizontally (intuitive mirror-selfie view)
            frame = cv2.flip(frame, 1)

            # 2. BGR → RGB for MediaPipe and st.image()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Detect hands with MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            hand_count      = 0
            predicted_label = None
            confidence      = None

            if detection_result.hand_landmarks:
                hand_count = len(detection_result.hand_landmarks)

                # 4. Draw skeleton on frame
                draw_hand_skeleton_rgb(rgb_frame, detection_result.hand_landmarks)

                # 5. Extract exactly 126 features (same method as data_collection.py)
                features = extract_landmark_features(detection_result.hand_landmarks)

                # 6. Run the trained classifier
                predicted_label, confidence = predict_sign(
                    model, encoder, supports_proba, features
                )

            # 7. Draw HUD overlay
            draw_prediction_overlay(rgb_frame, hand_count, predicted_label, confidence)

            # 8. Update metric widgets
            hands_ph.metric("Hands Detected", f"{hand_count} / {MAX_NUM_HANDS}")

            if predicted_label:
                pred_ph.metric("Prediction", predicted_label)
                if confidence is not None:
                    conf_ph.metric("Confidence", f"{confidence:.1f}%")
                else:
                    conf_ph.metric("Confidence", "N/A")
            else:
                pred_ph.metric("Prediction", "No hand")
                conf_ph.metric("Confidence", "—")

            # 9. Render annotated frame in the browser
            frame_ph.image(rgb_frame, channels="RGB", width="stretch")

            # 10. Large prediction banner below the video
            if predicted_label:
                # Colour the banner based on confidence level
                if confidence is None or confidence >= 70:
                    banner_color = "#1a472a"   # Dark green — high confidence
                    text_color   = "#69ff9d"
                elif confidence >= 50:
                    banner_color = "#3d3200"   # Dark yellow — moderate
                    text_color   = "#ffd966"
                else:
                    banner_color = "#3d0000"   # Dark red — low confidence
                    text_color   = "#ff6b6b"

                conf_line = (
                    f"<p style='color:#aaa; font-size:1rem; margin:4px 0 0;'>"
                    f"Confidence: {confidence:.1f}%</p>"
                    if confidence is not None
                    else ""
                )
                pred_banner_ph.markdown(
                    f"<div style='text-align:center; padding:24px; "
                    f"background:{banner_color}; border-radius:12px; "
                    f"border:2px solid {text_color};'>"
                    f"<p style='color:#ccc; font-size:0.9rem; margin:0 0 6px;'>Detected Sign</p>"
                    f"<p style='color:{text_color}; font-size:2.8rem; "
                    f"font-weight:700; margin:0; letter-spacing:2px;'>"
                    f"{predicted_label}</p>"
                    f"{conf_line}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                pred_banner_ph.markdown(
                    "<div style='text-align:center; padding:24px; "
                    "background:#1e1e1e; border-radius:12px; border:2px solid #555;'>"
                    "<p style='color:#ff9800; font-size:1.6rem; "
                    "font-weight:600; margin:0;'>✋ No hand detected</p>"
                    "<p style='color:#888; font-size:0.95rem; margin:6px 0 0;'>"
                    "Show your hand to the webcam to begin recognition.</p></div>",
                    unsafe_allow_html=True,
                )

            # Smooth loop pacing (~30 fps cap)
            elapsed = time.time() - t0
            sleep_t = max(0.0, 0.033 - elapsed)
            time.sleep(sleep_t)

    except Exception as exc:
        # Let Streamlit's internal Rerun / Stop signals propagate normally
        if "Rerun" in type(exc).__name__ or "Stop" in type(exc).__name__:
            raise exc
        st.error(f"❌ Unexpected error during prediction loop: {exc}")

    finally:
        # Always release the webcam when done
        if cap is not None and cap.isOpened():
            cap.release()
        st.session_state.camera_running = False


if __name__ == "__main__":
    main()
