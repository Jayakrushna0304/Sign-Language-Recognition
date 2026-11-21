# Real-Time Sign Language Recognition Using Machine Learning

A Final-Year B.Tech Capstone Project for recognizing hand sign gestures in real time using computer vision and machine learning techniques.

---

## 📌 Project Overview
Sign language is the primary medium of communication for the deaf and hard-of-hearing community. This project aims to bridge the communication gap by translating hand gestures into readable text and speech in real time using standard webcam input.

### Key Objectives
- Capture real-time hand gestures via webcam.
- Extract hand landmark keypoints (using MediaPipe / OpenCV).
- Classify gestures using trained Machine Learning / Deep Learning models.
- Display predictions with high accuracy and low latency on a clean user interface.

---

## 🗂️ Project Structure

```text
Sign_Language_Recognition/
│
├── dataset/             # Raw and processed gesture datasets / landmark coordinates
├── preprocessing/       # Scripts for data extraction, landmark extraction, and augmentation
├── models/              # Model architectures, training scripts, and evaluation routines
├── saved_model/         # Exported weights, trained model binaries (.pkl, .h5, etc.)
├── realtime/            # Real-time webcam inference and live gesture prediction scripts
├── app/                 # User interface application (e.g., Streamlit / web / desktop dashboard)
├── utils/               # Helper utilities, visualization, audio feedback, and metrics
├── results/             # Plots, confusion matrices, evaluation reports, and demo recordings
│
├── config.py            # Central project configurations, paths, and hyperparameters
├── requirements.txt     # Python dependencies tailored for Python 3.12
└── README.md            # Comprehensive project documentation
```

---

## ⚙️ Environment & Setup
- **Python Version**: 3.12
- *(Dependencies will be installed in subsequent steps)*

---

## 🚀 Development Roadmap
1. [x] Step 1: Project requirement analysis & folder structure initialization
2. [ ] Step 2: Dataset selection, collection & landmark extraction pipeline
3. [ ] Step 3: Data preprocessing, normalization & feature engineering
4. [ ] Step 4: Model architecture design, training & validation
5. [ ] Step 5: Model evaluation, metrics calculation & confusion matrix generation
6. [ ] Step 6: Real-time webcam inference pipeline implementation
7. [ ] Step 7: Application UI development (Interactive dashboard)
8. [ ] Step 8: Final integration, testing & documentation
