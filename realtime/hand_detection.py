"""
Real-Time Hand Detection & Landmark Visualization.
Part of the 'Real-Time Sign Language Recognition' Capstone Project.

Features:
- Accesses webcam using OpenCV (index 0).
- Automatically adapts to the installed MediaPipe API (Solutions API or Tasks API).
- Detects up to 2 hands simultaneously.
- Draws all 21 keypoint landmarks and skeleton connections.
- Displays live hand count and instructions on the mirrored video feed.
- Clean exit and proper camera release on pressing 'Q'.
"""

import sys
from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp

# Ensure project root is accessible for project configurations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Configuration constants
CAMERA_INDEX = 0
WINDOW_TITLE = "Hand Detection"
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Official MediaPipe Hand Landmarker model URL (used by Tasks API in MediaPipe 1.x / >=0.10)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Standard 21-point hand landmark skeleton connections
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
    # Palm base connections
    (5, 9), (9, 13), (13, 17)
]


def ensure_model_file() -> Path:
    """
    Ensures the official MediaPipe hand_landmarker.task model file is available locally.
    Downloads it automatically on first run if missing.
    """
    if not MODEL_PATH.exists():
        print(f"[INFO] Model asset not found locally. Downloading to {MODEL_PATH}...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("[INFO] Hand Landmarker model downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download model asset: {e}")
            print(f"[TIP] You can manually download it from:\n  {MODEL_URL}\nand save it to:\n  {MODEL_PATH}")
            raise
    return MODEL_PATH


def create_hand_detector():
    """
    Factory function that detects the installed MediaPipe version and returns
    a compatible hand detector function.
    Supports both legacy Solutions API (mediapipe < 1.0) and modern Tasks API (mediapipe >= 1.0).
    """
    # Option A: Legacy MediaPipe Solutions API (mp.solutions.hands)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("[INFO] Using MediaPipe Solutions API (mp.solutions.hands).")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        def detect_fn(rgb_frame):
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks
            return []

        def close_fn():
            hands.close()

        return detect_fn, close_fn

    # Option B: Modern MediaPipe Tasks Vision API (mp.tasks.python.vision)
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        print(f"[INFO] Using MediaPipe Tasks API (mediapipe v{getattr(mp, '__version__', 'unknown')}).")
        model_file = ensure_model_file()

        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

        def detect_fn(rgb_frame):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)
            if detection_result.hand_landmarks:
                return detection_result.hand_landmarks
            return []

        def close_fn():
            if hasattr(landmarker, "close"):
                landmarker.close()

        return detect_fn, close_fn

    except Exception as e:
        print(f"[ERROR] Could not initialize MediaPipe hand detector: {e}")
        raise


def draw_hand_skeleton(frame, hand_landmarks_list):
    """
    Draws hand skeleton connections and all 21 landmark points on the frame.
    Works seamlessly across all MediaPipe landmark representations.
    """
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        # Extract pixel coordinates for all 21 landmarks
        landmark_points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmark_points.append((px, py))

        # 1. Draw connection lines between landmarks
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                cv2.line(
                    frame,
                    landmark_points[start_idx],
                    landmark_points[end_idx],
                    (0, 255, 128),  # Vibrant light-green connection line
                    2,
                    cv2.LINE_AA,
                )

        # 2. Draw circles for all 21 landmark keypoints
        for idx, (px, py) in enumerate(landmark_points):
            # Highlight key fingertips and wrist with distinct coloring
            if idx in (0, 4, 8, 12, 16, 20):  # Wrist and 5 fingertips
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)       # Red filled core
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)   # White border
            else:  # Intermediate finger joints
                cv2.circle(frame, (px, py), 4, (255, 128, 0), -1, cv2.LINE_AA)     # Cyan/blue dot
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)   # White border


def draw_hud(frame, hand_count: int):
    """
    Draws a clean Heads-Up Display (HUD) showing detected hand count and controls.
    """
    # Background panel
    cv2.rectangle(frame, (15, 15), (320, 95), (20, 20, 20), -1)
    status_color = (0, 255, 0) if hand_count > 0 else (0, 165, 255)
    cv2.rectangle(frame, (15, 15), (320, 95), status_color, 2)

    # Hand count display
    cv2.putText(
        frame,
        f"Hands Detected: {hand_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Exit instruction
    cv2.putText(
        frame,
        "Press 'Q' to Exit",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def run_hand_detection():
    """
    Main loop: captures webcam frames, detects hand landmarks, and renders output.
    """
    print("=" * 65)
    print("      Real-Time Sign Language Recognition - Hand Detection      ")
    print("=" * 65)

    # 1. Initialize MediaPipe detector
    try:
        detect_hands, close_detector = create_hand_detector()
    except Exception as e:
        print(f"[FATAL] Failed to configure MediaPipe: {e}")
        return

    # 2. Initialize webcam
    print(f"[INFO] Opening webcam (Index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Windows DirectShow fallback if default backend fails
    if not cap.isOpened() and sys.platform.startswith("win"):
        print("[INFO] Retrying camera connection using DirectShow backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    # Error handling if webcam cannot be opened
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open webcam at index {CAMERA_INDEX}.")
        print("[ERROR] Please check that:")
        print("  1. Your webcam is connected and enabled.")
        print("  2. App camera permissions are allowed in Windows Settings.")
        print("  3. No other application is using the webcam.")
        print("=" * 65)
        close_detector()
        return

    # Window setup
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)

    print("[INFO] Camera stream started.")
    print("[INFO] Press 'Q' or 'q' on the video window to quit.")
    print("-" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received. Skipping...")
                continue

            # 3. Mirror the video horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)

            # 4. Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 5. Detect hands
            detected_hands = detect_hands(rgb_frame)
            hand_count = len(detected_hands)

            # 6. Draw landmarks and skeleton connections
            if hand_count > 0:
                draw_hand_skeleton(frame, detected_hands)

            # 7. Render on-screen HUD (hands detected and controls)
            draw_hud(frame, hand_count)

            # 8. Show output frame
            cv2.imshow(WINDOW_TITLE, frame)

            # 9. Handle exit on 'Q'
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("[INFO] 'Q' pressed. Exiting...")
                break

            # Handle window close 'X' button
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user. Exiting...")
                break

    except Exception as e:
        print(f"\n[ERROR] Runtime error during hand detection: {e}")

    finally:
        # Proper cleanup of camera and windows
        cap.release()
        cv2.destroyAllWindows()
        close_detector()
        print("[INFO] Camera released and resources cleared successfully.")
        print("=" * 65)


if __name__ == "__main__":
    run_hand_detection()
