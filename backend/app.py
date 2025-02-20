import os
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
import librosa
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.playback import play
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
face_emotion_model = load_model(r'C:\Users\aksha\Desktop\emotion-detection-frontend\ml_models\facial detection model\emotion_recognition_model.h5')
audio_emotion_model = load_model(r'C:\Users\aksha\Desktop\emotion-detection-frontend\ml_models\Audio\best_voice_emotion_model.h5')
label_encoder = joblib.load(r'C:\Users\aksha\Desktop\emotion-detection-frontend\ml_models\Audio\voice_emotion_label_encoder.pkl')

face_emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
predicted_audio_emotion = "Listening..."
predicted_facial_emotion = "No Face Detected"
emotion_lock = threading.Lock()

# Video processing parameters
target_fps = 12
frame_skip_factor = 2

def extract_audio_from_video(video_path):
    """Extracts and saves audio from the video as a WAV file."""
    video = VideoFileClip(video_path)
    audio_path = os.path.join(UPLOAD_FOLDER, "extracted_audio.wav")
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path

def play_audio(audio_path):
    """Plays the extracted audio in sync with the video."""
    audio = AudioSegment.from_wav(audio_path)
    play(audio)

def extract_mfcc(audio, sample_rate, max_pad_length=100):
    """Extracts MFCC and other audio features."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        [np.mean(zcr)]
    ])

    if len(features) > max_pad_length:
        features = features[:max_pad_length]
    else:
        features = np.pad(features, (0, max_pad_length - len(features)), "constant")

    return features

def process_audio(audio_path):
    """Processes extracted audio in real-time for emotion detection."""
    global predicted_audio_emotion

    y, sr = librosa.load(audio_path, sr=22050)
    chunk_size = sr * 5
    num_chunks = len(y) // chunk_size

    for i in range(num_chunks):
        chunk = y[i * chunk_size : (i + 1) * chunk_size]
        features = extract_mfcc(chunk, sr)
        features = np.expand_dims(features, axis=0)

        prediction = audio_emotion_model.predict(features)
        predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        with emotion_lock:
            predicted_audio_emotion = predicted_emotion

        print(f"Audio Emotion: {predicted_audio_emotion}")
        time.sleep(5)

def facial_emotion_detection(video_path):
    """Plays video naturally while detecting facial emotions in real-time."""
    global predicted_facial_emotion

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / target_fps
    frame_skip = int(original_fps // target_fps) * frame_skip_factor

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_count = 0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        detected_emotion = "No Face Detected"
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype("float32") / 255.0
                face_roi = np.expand_dims(face_roi, axis=[0, -1])

                prediction = face_emotion_model.predict(face_roi)
                detected_emotion = face_emotion_labels[np.argmax(prediction)]

                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            with emotion_lock:
                predicted_facial_emotion = detected_emotion

        with emotion_lock:
            cv2.putText(frame, f"Audio Emotion: {predicted_audio_emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Emotion Detection", frame)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload and starts emotion detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        extracted_audio_path = extract_audio_from_video(video_path)

        video_thread = threading.Thread(target=facial_emotion_detection, args=(video_path,))
        audio_thread = threading.Thread(target=process_audio, args=(extracted_audio_path,))
        play_audio_thread = threading.Thread(target=play_audio, args=(extracted_audio_path,))

        video_thread.start()
        audio_thread.start()
        play_audio_thread.start()

        return jsonify({"message": "File uploaded successfully", "video_path": video_path})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
