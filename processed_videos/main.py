import os
import tempfile
import cv2
import torch
import torchaudio
import asyncio
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import multiprocessing
import time
from collections import Counter
from moviepy import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from celery import Celery
import subprocess
from Finalmodel import load_models, classify_hand_gesture, predict_emotions, refine_emotion_prediction, detect_eye_contact

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Sample function to simulate CPU-bound task (e.g., model inference)
def cpu_bound_task(input_data):
    time.sleep(2)  # Simulate long task
    return f"Processed {input_data}"

# Initialize FastAPI app
app = FastAPI()

# Initialize multiprocessing pool
pool = multiprocessing.Pool(processes=4)

# Shared memory manager for multiprocessing
manager = multiprocessing.Manager()
shared_data = manager.dict()

# Producer function to update shared data at regular intervals
def producer(shared_data):
    count = 0
    while True:
        shared_data['value'] = count
        time.sleep(10)
        count += 1

# Start long-running tasks like loading models
@app.on_event("startup")
def startup_event():
    load_models()  # Load models only once on startup
    # Start the producer process to update shared data
    p = multiprocessing.Process(target=producer, args=(shared_data,))
    p.start()

@app.get("/")
def read_root():
    return {"message": "Models loaded successfully"}    

@app.post("/analyze-video/")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    return await analyze_video(file)

# Function to extract audio from video using FFmpeg
async def extract_audio_from_video(video_file_path):
    try:
        base_name = os.path.splitext(os.path.basename(video_file_path))[0]
        audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
        
        # Use FFmpeg for faster audio extraction
        command = ['ffmpeg', '-i', video_file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_file_path]
        subprocess.run(command, check=True)
        
        logger.info(f"Audio extracted successfully to {audio_file_path}")
        return audio_file_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe the audio file
async def transcribe_audio(audio_file_path):
    try:
        recognizer = sr.Recognizer()
        audio_segment = AudioSegment.from_file(audio_file_path)
        audio_duration = len(audio_segment) / 1000
        audio_file = sr.AudioFile(audio_file_path)

        transcription = ""
        current_position = 0
        chunk_size = 30  # in seconds

        with audio_file as source:
            recognizer.adjust_for_ambient_noise(source)

            while current_position < audio_duration:
                try:
                    audio_chunk = recognizer.record(source, duration=chunk_size)
                    chunk_text = recognizer.recognize_google(audio_chunk)
                    transcription += chunk_text + " "
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    logger.error(f"Error during recognition: {e}")
                    break
                current_position += chunk_size

        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

# Video analysis for gesture, emotion, and eye contact
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(file.file.read())
            video_file_path = tmp_video.name

        audio_file_path = await asyncio.get_event_loop().run_in_executor(None, extract_audio_from_video, video_file_path)
        transcription = await asyncio.get_event_loop().run_in_executor(None, transcribe_audio, audio_file_path)

        cap = cv2.VideoCapture(video_file_path)
        frame_count = 0
        all_emotions = []
        all_eye_contacts = []
        gesture_counter = {"Open Palm": 0, "Closed Fist": 0, "Pointing Finger": 0, "Thumbs Up": 0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                gesture = "Unknown Gesture"
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if mp_hands is not None:
                    results_hands = mp_hands.process(image_rgb)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            gesture = await asyncio.get_event_loop().run_in_executor(None, classify_hand_gesture, hand_landmarks)

                emotion = await asyncio.get_event_loop().run_in_executor(None, predict_emotions, frame)
                eye_contact = await asyncio.get_event_loop().run_in_executor(None, detect_eye_contact, frame)
                refined_emotion = refine_emotion_prediction(emotion, eye_contact)

                all_emotions.append(refined_emotion)
                all_eye_contacts.append(eye_contact)

                if gesture != "Unknown Gesture":
                    gesture_counter[gesture] += 1

            frame_count += 1

        cap.release()

        dominant_emotion = Counter(all_emotions).most_common(1)[0][0]
        dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0]

        combined_feedback = {
            "transcription": transcription,
            "dominant_emotion": dominant_emotion,
            "dominant_eye_contact": dominant_eye_contact,
            "gesture_summary": gesture_counter
        }

        os.remove(video_file_path)
        os.remove(audio_file_path)

        return JSONResponse(content=combined_feedback)
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return JSONResponse(content={"error": "An error occurred during video processing"}, status_code=500)

@app.get("/shared-value")
def get_shared_value():
    value = shared_data.get('value', 'Not Available')
    return {"value": value}

@app.get("/start-task")
async def start_task():
    # Start a CPU-bound task using the multiprocessing pool
    result = await asyncio.get_event_loop().run_in_executor(None, pool.apply, cpu_bound_task, ('data',))
    return {"result": result}
@app.post("/async-endpoint")
async def test_endpoint():
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            results = await loop.run_in_executor(pool,
                                                 subject_extract_check_process,
                                                  *(args, subject_check_rules, docx_paras, pdf_paras))  # wait result
    except Exception as err:
        logger.error(f"Subprocess execute fail: {str(err)}")
        results = []
    return results