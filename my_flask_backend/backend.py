from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import mediapipe as mp
from deepface import DeepFace
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, pipeline, T5ForConditionalGeneration, T5Tokenizer
import os
from moviepy import VideoFileClip
import torchaudio
import torchaudio.transforms as T
import speech_recognition as sr
from pydub import AudioSegment
from collections import Counter
import difflib
import string

app = Flask(__name__)

posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
posenet_model = hub.load(posenet_model_url)

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

num_labels = 6
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model.eval()

pronunciation_pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

gesture_to_body_language = {
    "Open Palm": "Honesty",
    "Closed Fist": "Determination",
    "Pointing Finger": "Confidence",
    "Thumbs Up": "Encouragement",
    "Thumbs Down": "Criticism",
    "Victory Sign": "Peace",
    "OK Sign": "Agreement",
    "Rock Sign": "Excitement",
    "Call Me": "Friendly"
}

posture_meanings = {
    "Head Up": "Confidence",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Arms on Hips": "Confidence",
    "Crossed Arms": "Defensive",
    "Hands in Pockets": "Casual"
}

posture_counter = {key: 0 for key in posture_meanings}
gesture_counter = {key: 0 for key in gesture_to_body_language}


def run_inference(frame):
    resized_frame = cv2.resize(frame, (192, 192))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
    rgb_frame = rgb_frame / 255.0
    rgb_frame = tf.cast(rgb_frame, dtype=tf.int32)
    model_input = {"input": rgb_frame}
    return posenet_model.signatures['serving_default'](**model_input)

def extract_keypoints(results):
    keypoints = []
    for i in range(17):  
        x = results['output_0'][0][0][i][1].numpy()
        y = results['output_0'][0][0][i][0].numpy()
        confidence = results['output_0'][0][0][i][2].numpy()
        keypoints.append({"x": x, "y": y, "confidence": confidence})
    return keypoints

def classify_posture(keypoints):
    shoulder_y = keypoints[5]['y']
    hip_y = keypoints[11]['y']
    knee_y = keypoints[13]['y']
    ankle_y = keypoints[15]['y']
    head_y = keypoints[0]['y']

    if abs(head_y - shoulder_y) < 0.05:
        return "Head Up"
    if shoulder_y > hip_y and knee_y > shoulder_y and ankle_y > knee_y:
        return "Slouching"
    if head_y < shoulder_y and knee_y > hip_y:
        return "Leaning Forward"
    if head_y > shoulder_y:
        return "Head Down"
    if shoulder_y < hip_y and knee_y < hip_y and ankle_y < knee_y:
        return "Leaning Back"
    if abs(keypoints[5]['x'] - keypoints[6]['x']) < 0.1 and abs(keypoints[11]['x'] - keypoints[12]['x']) < 0.1:
        return "Arms on Hips"
    if keypoints[9]['y'] < keypoints[3]['y'] and keypoints[10]['y'] < keypoints[4]['y']:
        return "Crossed Arms"
    if keypoints[9]['y'] > hip_y and keypoints[10]['y'] > hip_y:
        return "Hands in Pockets"
    return "Unknown Posture"

def predict_emotions(image):
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        return "Unknown"

def detect_eye_contact(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    return "Eye Contact" if len(eyes) > 0 else "No Eye Contact"

def classify_hand_gesture(hand_landmarks):
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19
    THUMB_MCP = 2

    def is_finger_extended(tip, dip):
        return landmarks[tip][1] < landmarks[dip][1]

    thumb_extended = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]
    index_extended = is_finger_extended(INDEX_TIP, INDEX_DIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_DIP)
    ring_extended = is_finger_extended(RING_TIP, RING_DIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_DIP)

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm"
    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist"
    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger"
    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up"
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down"
    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign"
    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign"
    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign"
    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me"
    return "Unknown Gesture"

def extract_audio_from_video(video_file_path):
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)
    return audio_file_path

def process_audio_for_speech_analysis(audio_file_path):
    recognizer = sr.Recognizer()
    audio_segment = AudioSegment.from_file(audio_file_path)
    audio_duration = len(audio_segment) / 1000
    transcription = ""
    current_position = 0
    with sr.AudioFile(audio_file_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        while current_position < audio_duration:
            try:
                audio_chunk = recognizer.record(source, duration=30)
                chunk_text = recognizer.recognize_google(audio_chunk)
                transcription += chunk_text + " "
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                return None
            current_position += 30
    return transcription

def correct_grammar(sentence, t5_model, tokenizer, max_length=512):
    corrected_text = ""
    i = 0
    while i < len(sentence):
        chunk = sentence[i:i + max_length]
        input_text = "grammar: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)
        outputs = t5_model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        corrected_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_text += corrected_chunk + " "
        i += max_length
    return corrected_text.strip()

def calculate_speech_pace(audio_file):
    audio_segment = AudioSegment.from_file(audio_file)
    duration = len(audio_segment) / 1000
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return None, None
    word_count = len(transcription.split())
    minutes = duration / 60
    pace = word_count / minutes if minutes > 0 else 0
    return pace, transcription

@app.route('/')
def home():
    return "Flask server is running"

@app.route('/process_video', methods=['POST'])
def process_video():
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    video_path = os.path.join("temp_video.mp4")
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_emotions = []
    all_eye_contacts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            results = run_inference(frame)
            keypoints = extract_keypoints(results)
            posture = classify_posture(keypoints)
            gesture = classify_hand_gesture(mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks[0])
            emotion = predict_emotions(frame)
            eye_contact = detect_eye_contact(frame)

            all_emotions.append(emotion)
            all_eye_contacts.append(eye_contact)

        frame_count += 1
    cap.release()

    dominant_emotion = Counter(all_emotions).most_common(1)[0][0]
    dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0]

    audio_file_path = "temp_audio.wav"
    extract_audio_from_video(video_path)

    transcription = process_audio_for_speech_analysis(audio_file_path)
    grammar_corrected = correct_grammar(transcription, t5_model, tokenizer)
    pace, pace_feedback = calculate_speech_pace(audio_file_path)

    final_feedback = {
        'dominant_emotion': dominant_emotion,
        'dominant_eye_contact': dominant_eye_contact,
        'speech_transcription': transcription,
        'grammar_feedback': grammar_corrected,
        'speech_pace': pace,
        'speech_feedback': pace_feedback,
        'gesture_summary': gesture_counter,
        'posture_summary': posture_counter
    }

    return jsonify(final_feedback)

if __name__ == '__main__':
    app.run(debug=True)
