"""
Camera Test Script for Real-Time Sign Language Recognition.
Simple and Windows-compatible webcam test using OpenCV with default backend.
Includes diagnostics for camera initialization, backend, frame capture, dimensions, and window status.
"""

import cv2


def test_camera():
    window_name = "Camera Test"

    # 1. Create a simple OpenCV window using cv2.namedWindow()
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Diagnostic 5: Whether the OpenCV display window is created successfully
    try:
        window_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
        window_created = window_prop >= 0
    except Exception as e:
        window_created = False
        print(f"[ERROR] Error checking OpenCV window status: {e}")
    print(f"[DIAGNOSTIC] OpenCV display window created successfully: {window_created}")

    # 2. Open the webcam using cv2.VideoCapture(0)
    print("[INFO] Opening webcam using cv2.VideoCapture(0)...")
    cap = cv2.VideoCapture(0)

    # Diagnostic 1: Whether the webcam opens successfully
    webcam_opened = cap.isOpened()
    print(f"[DIAGNOSTIC] Webcam opened successfully: {webcam_opened}")

    if not webcam_opened:
        print("[ERROR] Could not open webcam using cv2.VideoCapture(0).")
        print("[ERROR] Please check webcam connection, permissions, or if another application is using it.")
        return

    # Diagnostic 2: Which camera backend is being used
    try:
        backend_name = cap.getBackendName()
    except Exception as e:
        backend_name = f"Default (fallback: {e})"
    print(f"[DIAGNOSTIC] Camera backend in use: {backend_name}")

    print("[INFO] Webcam initialized successfully. Streaming live video...")
    print("[INFO] Press 'Q' or 'q' to exit.")

    first_frame_received = False
    imshow_called = False

    # 4. Capture frames continuously
    try:
        while True:
            ret, frame = cap.read()

            # Diagnostic 3: Whether cap.read() successfully receives frames
            if not first_frame_received:
                print(f"[DIAGNOSTIC] cap.read() successfully receives frames: {ret} (frame is valid: {frame is not None and frame.size > 0})")

            # Error handling for frame capture
            if not ret or frame is None:
                print("[ERROR] Failed to capture frame from webcam with cap.read().")
                break

            # Print a message when the first video frame is received
            # Diagnostic 4: The frame width and height
            if not first_frame_received:
                first_frame_received = True
                height, width = frame.shape[:2]
                print("[INFO] First video frame received successfully!")
                print(f"[DIAGNOSTIC] Frame width: {width}, height: {height}")

            # 5. Display frames using cv2.imshow() - executes for every valid frame
            cv2.imshow(window_name, frame)

            # Diagnostic print statement confirming that cv2.imshow is being called
            if not imshow_called:
                imshow_called = True
                print(f"[DIAGNOSTIC] cv2.imshow('{window_name}', frame) is being called.")

            # 6. Process window events using cv2.waitKey(1) - executes immediately after cv2.imshow()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("[INFO] 'Q' pressed. Exiting...")
                break

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    finally:
        # Proper resource cleanup only after the video loop finishes
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Video loop finished. Webcam released and OpenCV windows closed successfully.")


if __name__ == "__main__":
    test_camera()
