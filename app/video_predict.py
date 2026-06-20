"""
Deepfake Video Detection - Inference Pipeline
Approach:
  1. Extract evenly-spaced frames from the video
  2. Detect & crop the face region in each frame (OpenCV Haar Cascade)
  3. Run each face crop through MesoNet (Meso4) - a pretrained CNN
     specifically trained for deepfake artifact detection
  4. Aggregate per-frame scores -> final video-level verdict

MesoNet reference: Afchar et al., 2018 - "MesoNet: a Compact Facial
Video Forgery Detection Network" (pretrained weights via DariusAf/MesoNet)
"""
import os
import cv2
import numpy as np
from mesonet_classifiers import Meso4

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "models", "mesonet_weights", "Meso4_DF.h5")
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = 256

_model = None
_face_cascade = None


def get_model():
    global _model
    if _model is None:
        _model = Meso4()
        _model.load(WEIGHTS_PATH)
    return _model


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    return _face_cascade


def extract_frames(video_path, max_frames=20):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    step = max(1, total // max_frames)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(frames) < max_frames:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def crop_face(frame, cascade, margin=0.3):
    """Detect the largest face in a frame and return a margin-padded crop.
    Falls back to the full frame if no face is detected."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return frame  # fallback: use the whole frame

    # pick the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    return frame[y0:y1, x0:x1]


def preprocess(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    return face_img


def predict_video(video_path, max_frames=20, threshold=0.5):
    """
    Returns a dict with:
      verdict: 'REAL' or 'FAKE'
      confidence: float (0-1)
      frame_scores: list of per-frame scores
      frames_analyzed: int
    Score interpretation (MesoNet Meso4_DF convention): closer to 1 = REAL, closer to 0 = FAKE
    """
    model = get_model()
    cascade = get_face_cascade()

    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        return {"error": "Could not read frames from video. Check the file format."}

    batch = []
    for frame in frames:
        face = crop_face(frame, cascade)
        batch.append(preprocess(face))

    batch = np.array(batch)
    scores = model.predict(batch).flatten().tolist()

    avg_score = float(np.mean(scores))
    verdict = "REAL" if avg_score >= threshold else "FAKE"
    confidence = avg_score if verdict == "REAL" else (1 - avg_score)

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 2),
        "raw_avg_score": round(avg_score, 4),
        "frame_scores": [round(s, 4) for s in scores],
        "frames_analyzed": len(frames),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_predict.py <path_to_video>")
        sys.exit(1)
    result = predict_video(sys.argv[1])
    print(result)
