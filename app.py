import atexit
import os
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from gesture_store import GestureStore
from hand_features import create_detector, extract_features

# ------------------- CONFIG -------------------
UPLOAD_FOLDER = "uploads"
GESTURES_FILE = "gestures.json"
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MIN_CONFIDENCE = 0.5          # used when reading videos
MAX_DISTANCE = 1.6            # farther than this from every taught sample = "not sure"
MAX_LABEL_LENGTH = 40
# On a public website set the environment variable ALLOW_TEACHING=0 so visitors cannot change the words
TEACHING_ENABLED = os.environ.get("ALLOW_TEACHING", "1") != "0"
IGNORED_WORDS = {"rest", "idle", "nothing", "neutral"}   # taught poses that never become words

app = Flask(__name__)
CORS(app)  # lets the React website (a different port) call this backend
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = create_detector()
detector_lock = threading.Lock()   # MediaPipe is not safe to use from two requests at once
store = GestureStore(GESTURES_FILE)
atexit.register(store.save)


# ------------------- HELPERS -------------------
def teaching_off():
    return jsonify({"error": "Teaching is turned off on this site"}), 403


def features_from_frame(frame):
    with detector_lock:
        return extract_features(detector, frame)


def features_from_upload():
    """Reads the 'file' field of the request. Returns (features, error_response)."""
    if "file" not in request.files:
        return None, (jsonify({"error": "No file uploaded"}), 400)
    data = np.frombuffer(request.files["file"].read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return None, (jsonify({"error": "Could not read that image"}), 400)
    return features_from_frame(frame), None


def classify(features):
    """Turns a hand shape into a JSON-friendly answer."""
    if features is None:
        return {"status": "no_hand", "prediction": None, "confidence": 0.0, "ignore": False}
    result = store.predict(features)
    if result is None:
        return {"status": "no_gestures", "prediction": None, "confidence": 0.0, "ignore": False}
    label, confidence, distance = result
    if distance > MAX_DISTANCE:
        return {"status": "unsure", "prediction": None, "confidence": 0.0, "ignore": False}
    return {
        "status": "ok",
        "prediction": label,
        "confidence": round(confidence, 3),
        "ignore": label.lower() in IGNORED_WORDS,
    }


def words_from_video(path, step=5):
    cap = cv2.VideoCapture(path)
    seen = []   # one entry per checked frame: a word, or None when nothing was recognised
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number % step == 0:
            answer = classify(features_from_frame(frame))
            usable = (
                answer["status"] == "ok"
                and not answer["ignore"]
                and answer["confidence"] >= MIN_CONFIDENCE
            )
            seen.append(answer["prediction"] if usable else None)
        frame_number += 1
    cap.release()

    # a word counts once it is seen in 2 checks in a row; it can repeat after a short gap
    words, run_label, run_length, gap, last_added = [], None, 0, 0, None
    for label in seen:
        if label is None:
            run_label, run_length = None, 0
            gap += 1
            if gap >= 2:
                last_added = None
            continue
        gap = 0
        if label == run_label:
            run_length += 1
        else:
            run_label, run_length = label, 1
        if run_length >= 2 and label != last_added:
            words.append(label)
            last_added = label
    return " ".join(words)


# ------------------- ROUTES -------------------
@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Sign language API is running", "words": store.counts()})


@app.route("/api/gestures", methods=["GET"])
def list_gestures():
    return jsonify({"gestures": store.counts(), "teaching": TEACHING_ENABLED})


@app.route("/api/sample", methods=["POST"])
def add_sample():
    if not TEACHING_ENABLED:
        return teaching_off()
    label = request.form.get("label", "").strip()
    if not label or len(label) > MAX_LABEL_LENGTH or "/" in label or "\\" in label:
        return jsonify({"error": "Give the word a name (up to 40 characters, no slashes)"}), 400
    features, error = features_from_upload()
    if error:
        return error
    if features is None:
        return jsonify({"ok": False, "reason": "no_hand"})
    count = store.add(label, features)
    return jsonify({"ok": True, "count": count})


@app.route("/api/save", methods=["POST"])
def save_now():
    if not TEACHING_ENABLED:
        return teaching_off()
    store.save()
    return jsonify({"ok": True})


@app.route("/api/gestures/<path:label>", methods=["DELETE"])
def delete_gesture(label):
    if not TEACHING_ENABLED:
        return teaching_off()
    if not store.delete(label):
        return jsonify({"error": "No such word"}), 404
    return jsonify({"ok": True})


@app.route("/predict_image", methods=["POST"])
def predict_image():
    features, error = features_from_upload()
    if error:
        return error
    return jsonify(classify(features))


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    extension = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if extension not in VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video type"}), 400
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    try:
        sentence = words_from_video(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return jsonify({"prediction": sentence})


# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
